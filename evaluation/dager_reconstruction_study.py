"""Paired multi-sample DAGER text reconstruction with known and surrogate images.

Every attack commits before this module reads any private reference.  The
image-private conditions receive an attacker-generated random image through the
DAGER configuration; their public observations never contain the true image.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import os
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import digest, load_config
from evaluation.dager_ablation import compare_token_filters, validate_token_filters
from evaluation.reconstruction import evaluate


CONDITIONS = {
    "known_residual": ("image_known", "declared", "public_residual"),
    "known_raw": ("image_known", "declared", "raw"),
    "random_residual": ("private", "random", "public_residual"),
    "random_raw": ("private", "random", "raw"),
}
TOPKS = (1, 5, 10, 20, 50, 100, 200, 500, 1000)
ROOT = Path(__file__).resolve().parents[1]


def _protocol(base, knowledge, image_source, mode, candidates, beam_width, max_evaluations,
              seconds, seed):
    dager = replace(base.attack.dager, mode=mode, image_source=image_source,
                    token_selection="topk", max_candidates=candidates,
                    beam_width=beam_width, prefix_batch_size=32,
                    rerank_candidates=0)
    attack = replace(base.attack, dager=dager, seed=seed, max_evaluations=max_evaluations,
                     seconds=seconds, checkpoint_interval=64)
    training = replace(base.training, knowledge=knowledge, token_lengths_known=True)
    return replace(base, attack=attack, training=training)


def _run(command):
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def _capture(config, data, model, output, split, client, offset, device):
    if (output / "public" / "observation.json").exists():
        return
    _run([sys.executable, "-m", "core.commands", "capture",
          "--data", str(data), "--model", str(model), "--config", str(config),
          "--output", str(output), "--split", split, "--client", str(client),
          "--offset", str(offset), "--set", f"model.device={device}"])


def _attack(config, capture, model, output, device):
    if (output / "result.json").exists():
        return
    _run([sys.executable, "-m", "core.commands", "attack",
          "--observation", str(capture / "public"), "--model", str(model),
          "--config", str(config), "--output", str(output), "--device", device,
          "--resume"])


def _mean(values):
    return float(statistics.fmean(values)) if values else None


def _curve_row(report, topk):
    return next(row for row in report["ranking"]["curve"] if row["topk"] == topk)


def _aggregate(run_reports, candidates):
    grouped = {name: {"statuses": Counter(), "candidate_recall": [],
                      "ranking_recall": [], "average_precision": [], "text": defaultdict(list),
                      "full_candidate_coverage": 0, "full_ranking_coverage": 0,
                      "conditional_candidate_exact": [], "conditional_ranking_exact": [],
                      "outputs": []}
               for name in CONDITIONS}
    for run in run_reports:
        for name in CONDITIONS:
            row = run["conditions"][name]
            target = grouped[name]
            target["statuses"][row["status"]] += 1
            target["candidate_recall"].append(row["candidate_recall"])
            target["ranking_recall"].append(row["ranking_at_k"]["recall"])
            target["average_precision"].append(row["average_precision"])
            target["outputs"].append({"run": run["run"], **row["reconstruction"]})
            for metric, value in row["text_metrics"].items():
                target["text"][metric].append(value)
            if row["candidate_recall"] == 1:
                target["full_candidate_coverage"] += 1
                target["conditional_candidate_exact"].append(row["text_exact"])
            if row["ranking_at_k"]["recall"] == 1:
                target["full_ranking_coverage"] += 1
                target["conditional_ranking_exact"].append(row["text_exact"])
    summary = {}
    for name, values in grouped.items():
        summary[name] = {
            "statuses": dict(values["statuses"]),
            "candidate_recall_mean": _mean(values["candidate_recall"]),
            "ranking_recall_at_k_mean": _mean(values["ranking_recall"]),
            "average_precision_mean": _mean(values["average_precision"]),
            "full_candidate_coverage": values["full_candidate_coverage"],
            "conditional_sequence_exact_rate_given_full_candidates": _mean(
                values["conditional_candidate_exact"]),
            "full_ranking_coverage_at_k": values["full_ranking_coverage"],
            "conditional_sequence_exact_rate_given_full_ranking_at_k": _mean(
                values["conditional_ranking_exact"]),
            "text_metrics_mean": {metric: _mean(samples)
                                  for metric, samples in values["text"].items()},
            "reconstructions": values["outputs"],
        }
    return {"candidate_k": candidates, "sample_count": len(run_reports),
            "conditions": summary}


def evaluate_committed(output, configs, candidates):
    """Read references only after every planned attack has result.json."""
    run_dirs = sorted(path for path in output.glob("run-*") if path.is_dir())
    for run_dir in run_dirs:
        for name in CONDITIONS:
            if not (run_dir / "attacks" / name / "result.json").exists():
                raise ValueError("Every attack must commit before evaluation starts")

    reports = []
    for index, run_dir in enumerate(run_dirs):
        known_capture = run_dir / "captures" / "known"
        private_capture = run_dir / "captures" / "private"
        known_meta = read_json(known_capture / "public" / "observation.json")
        private_meta = read_json(private_capture / "public" / "observation.json")
        if known_meta["update_sha256"] != private_meta["update_sha256"]:
            raise ValueError("Paired knowledge captures produced different updates")
        for filename in ("images.safetensors", "text_tokens.safetensors"):
            if file_hash(known_capture / "private" / filename) != file_hash(
                    private_capture / "private" / filename):
                raise ValueError("Paired captures selected different private examples")

        directories = {name: run_dir / "attacks" / name for name in CONDITIONS}
        training = load_config(configs["known_residual"]).training
        comparison = compare_token_filters(
            directories, known_capture / "private", training, topk=candidates)
        rankings = validate_token_filters(
            directories, known_capture / "private", training, topk=candidates,
            topks=TOPKS, random_draws=100_000, seed=20261003 + index)
        write_json(run_dir / "filter_comparison.json", comparison)
        write_json(run_dir / "ranking_validation.json", rankings)

        run_report = {"run": index, "paired_update_sha256": known_meta["update_sha256"],
                      "conditions": {}}
        for name, (knowledge, _, _) in CONDITIONS.items():
            truth = (known_capture if knowledge == "image_known" else private_capture) / "private"
            report = evaluate(directories[name], truth, load_config(configs[name]).evaluation, "cpu")
            write_json(directories[name] / "evaluation.json", report)
            result = read_json(directories[name] / "result.json")
            ranking = rankings["modes"][name]
            curve = _curve_row(ranking, candidates)
            token_filter = result.get("provenance", {}).get("token_filter", {})
            metrics = report["samples"][0]["metrics"] if report["samples"] else {}
            text_metrics = {key: value for key, value in metrics.items()
                            if key.startswith(("question_", "target_"))}
            exact_fields = [value for key, value in text_metrics.items()
                            if key.endswith("_exact_match")]
            run_report["conditions"][name] = {
                "status": result["status"],
                "image_context": result.get("provenance", {}).get("image_context"),
                "selected_candidates": (token_filter.get("informative_candidates", 0)
                                        + token_filter.get("ambiguous_candidates", 0)),
                "ambiguous_candidates": token_filter.get("ambiguous_candidates"),
                "candidate_recall": report["token_detection"]["candidates"]["recall"],
                "ranking_at_k": curve,
                "average_precision": ranking["ranking"]["average_precision"],
                "text_exact": float(bool(exact_fields) and all(value == 1 for value in exact_fields)),
                "text_metrics": text_metrics,
                "reconstruction": {"questions": result["questions"],
                                   "targets": result["targets"]},
                "costs": result.get("costs", {}),
            }
        write_json(run_dir / "summary.json", run_report)
        reports.append(run_report)
    aggregate = _aggregate(reports, candidates)
    aggregate.update({
        "schema_version": 1,
        "conditions": aggregate["conditions"],
        "reference_policy": "private references read only after all attacks committed",
        "limitations": [
            "Random-surrogate images are fixed attacker-generated pixels; no image optimization.",
            "DAGER adapted sequence search is not a paper-faithful implementation of original DAGER.",
            "The four conditions share samples and victim updates; treat comparisons as paired.",
        ],
    })
    write_json(output / "summary.json", aggregate)
    return aggregate


def run(args):
    base = load_config(args.config)
    if base.attack.method != "dager_adapted":
        raise ValueError("The base protocol must select dager_adapted")
    output = Path(args.output).resolve()
    data, model = Path(args.data).resolve(), Path(args.model).resolve()
    base_path = Path(args.config).resolve()
    protocols = {name: _protocol(base, *settings, args.candidates, args.beam_width,
                                 args.max_evaluations, args.seconds, args.seed)
                 for name, settings in CONDITIONS.items()}
    study = {
        "schema_version": 1, "conditions": CONDITIONS, "samples": args.samples,
        "start_offset": args.start_offset, "split": args.split, "client": args.client,
        "candidates": args.candidates, "beam_width": args.beam_width,
        "max_evaluations": args.max_evaluations, "seconds": args.seconds, "seed": args.seed,
        "data_sha256": file_hash(data), "model_sha256": {
            name: file_hash(model / name) for name in ("model.json", "model.safetensors")},
        "base_config_sha256": file_hash(base_path), "source_sha256": source_fingerprint(),
        "device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    signature = digest(study)
    if output.exists():
        if not args.resume:
            raise FileExistsError("Study output exists; pass --resume for the identical plan")
        plan = read_json(output / "plan.json")
        if plan["signature"] != signature:
            raise ValueError("Study resume plan differs")
    else:
        output.mkdir(parents=True)
        write_json(output / "plan.json", {**study, "signature": signature})
    configs = {}
    for name, protocol in protocols.items():
        path = output / "configs" / f"{name}.json"
        if path.exists() and read_json(path) != asdict(protocol):
            raise ValueError("Saved condition protocol differs")
        if not path.exists():
            write_json(path, protocol)
        configs[name] = path

    stage = "capture"
    try:
        for run_id in range(args.samples):
            offset = args.start_offset + run_id
            run_dir = output / f"run-{run_id:05d}"
            known = run_dir / "captures" / "known"
            private = run_dir / "captures" / "private"
            write_json(output / "progress.json", {"stage": "capture", "run": run_id,
                                                   "offset": offset})
            _capture(configs["known_residual"], data, model, known, args.split,
                     args.client, offset, args.device)
            _capture(configs["random_residual"], data, model, private, args.split,
                     args.client, offset, args.device)
            known_meta = read_json(known / "public" / "observation.json")
            private_meta = read_json(private / "public" / "observation.json")
            if known_meta["update_sha256"] != private_meta["update_sha256"]:
                raise ValueError("Paired captures produced different victim updates")
            for name, (knowledge, _, _) in CONDITIONS.items():
                stage = "attack"
                write_json(output / "progress.json", {"stage": stage, "run": run_id,
                                                       "condition": name})
                capture = known if knowledge == "image_known" else private
                _attack(configs[name], capture, model, run_dir / "attacks" / name, args.device)

        stage = "evaluation"
        write_json(output / "progress.json", {"stage": stage})
        evaluate_committed(output, configs, args.candidates)
        write_json(output / "complete.json", {
            "status": "completed", "summary_sha256": file_hash(output / "summary.json")})
        write_json(output / "progress.json", {"stage": "completed"})
    except Exception as error:
        write_json(output / "failed.json", {"status": "failed", "stage": stage,
                                             "error_type": type(error).__name__,
                                             "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--split", choices=("tune", "eval"), default="tune")
    parser.add_argument("--client", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--max-evaluations", type=int, default=200_000)
    parser.add_argument("--seconds", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for field in ("samples", "candidates", "beam_width", "max_evaluations"):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
