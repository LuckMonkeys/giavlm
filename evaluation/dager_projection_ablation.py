"""Paired Q/K/V/QKV attacks on one public observation, then offline evaluation.

Only the selected projection changes. All four attacks commit before references
are read. Reports retain aggregate metrics, not private texts or sample IDs.
"""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import torch

from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import TrainingSpec, load_config
from evaluation.dager_ablation import compare_token_filters, validate_token_filters
from evaluation.reconstruction import evaluate


PROJECTIONS = ("q", "k", "v", "qkv")


def evaluate_committed(output, capture, evaluation_spec):
    """Check every public result before allowing any private-reference access."""
    directories = {name: output / name for name in PROJECTIONS}
    results = {name: read_json(directory / "result.json")
               for name, directory in directories.items()}
    observation = read_json(capture / "public" / "observation.json")
    expected_options = None
    for name, result in results.items():
        if result["observation_id"] != observation["observation_id"]:
            raise ValueError("Compared attacks must use the same observation")
        options = dict(result["provenance"]["options"])
        if options.pop("projections") != name:
            raise ValueError("Attack projection does not match its condition")
        if expected_options is not None and options != expected_options:
            raise ValueError("Non-projection DAGER options differ")
        expected_options = options
        if not (directories[name] / "token_candidates.safetensors").exists():
            raise ValueError("Every vocabulary scan must finish before comparison")

    training = TrainingSpec(**observation["training"])
    truth = capture / "private"
    comparison = compare_token_filters(directories, truth, training, topk=50)
    rankings = validate_token_filters(
        directories, truth, training,
        topks=(1, 5, 10, 20, 50, 100, 200, 256, 500, 1000),
        random_draws=100_000, seed=20261002)
    write_json(output / "top50_comparison.json", comparison)
    write_json(output / "ranking_validation.json", rankings)
    summary = {
        "schema_version": 1, "sample_count": training.sample_count,
        "observation_id": observation["observation_id"],
        "condition": {key: getattr(training, key) for key in (
            "knowledge", "token_lengths_known", "fine_tuning_strategy",
            "server_round", "algorithm", "lora_rank")},
        "shared_dager_options": expected_options, "projections": {},
        "top50_pairs": comparison["pairs"],
        "limitations": ["One paired development sample; no generalization claim.",
                        "Fixed thresholds and budgets; no reference-based tuning.",
                        "Token-set detection is distinct from sequence recovery.",
                        "Uniform random control does not control corpus frequency."],
    }
    for name, directory in directories.items():
        report = evaluate(directory, truth, evaluation_spec, "cpu")
        write_json(directory / "evaluation.json", report)
        result = results[name]
        summary["projections"][name] = {
            "status": result["status"],
            "subspaces": [{key: span[key] for key in (
                "rank", "public_rank", "input_dimension", "saturated")}
                for span in result["provenance"]["subspaces"]],
            "token_filter": result["provenance"]["token_filter"],
            "token_detection": report["token_detection"],
            "top50": comparison["modes"][name],
            "ranking": rankings["modes"][name]["ranking"],
            "text_metrics": [row["metrics"] for row in report["samples"]],
            "costs": result["costs"],
            "result_sha256": file_hash(directory / "result.json"),
        }
    write_json(output / "summary.json", summary)
    return summary


def run(args):
    config = load_config(args.config)
    if config.attack.method != "dager_adapted":
        raise ValueError("Supply the frozen DAGER protocol configuration")
    output, capture = Path(args.output).resolve(), Path(args.capture).resolve()
    model, config_path = Path(args.model).resolve(), Path(args.config).resolve()
    output.mkdir(parents=True, exist_ok=False)
    stage = "plan"
    try:
        public = capture / "public"
        plan = {
            "schema_version": 1, "projections": list(PROJECTIONS),
            "source_sha256": source_fingerprint(),
            "config_sha256": file_hash(config_path),
            "public_sha256": {name: file_hash(public / name) for name in (
                "observation.json", "update.safetensors", "public_images.safetensors",
                "upload.json")},
            "model_sha256": {name: file_hash(model / name) for name in (
                "model.json", "model.safetensors")},
            "attack": asdict(config.attack), "evaluation": asdict(config.evaluation),
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "policy": "Change projections only; commit all attacks before evaluation.",
        }
        write_json(output / "plan.json", plan)
        for name in PROJECTIONS:
            stage = name
            if source_fingerprint() != plan["source_sha256"]:
                raise ValueError("Source changed during the paired experiment")
            write_json(output / "progress.json", {"stage": "attack", "projection": name})
            command = [sys.executable, "-m", "core.commands", "attack",
                       "--observation", str(public), "--model", str(model),
                       "--config", str(config_path), "--output", str(output / name),
                       "--device", args.device, "--set", f"attack.dager.projections={name}"]
            print(f"Starting projection {name}", flush=True)
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])
        stage = "evaluation"
        for name, expected in plan["public_sha256"].items():
            if file_hash(public / name) != expected:
                raise ValueError("Public observation changed during the paired experiment")
        write_json(output / "progress.json", {"stage": stage})
        evaluate_committed(output, capture, config.evaluation)
        write_json(output / "complete.json", {
            "status": "completed", "summary_sha256": file_hash(output / "summary.json")})
        print("All projections and post-commit evaluations completed", flush=True)
    except Exception as error:
        write_json(output / "failed.json", {
            "status": "failed", "stage": stage,
            "error_type": type(error).__name__, "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
