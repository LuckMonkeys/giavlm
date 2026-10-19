"""Frozen paired text benchmark; truth is opened for scoring after all commits."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import gc
from pathlib import Path
import statistics
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import AttackSpec, Config, DAGEROptions, EvalSpec, ModelSpec, TrainingSpec, digest
from core.fl import restore_model
from evaluation.dager_ablation import validate_token_filters
from evaluation.dager_rank_study import attack_public, capture_pair
from evaluation.reconstruction import evaluate
from metrics.text import text_metrics


def attack_spec(family, image):
    return AttackSpec(
        method="dager_adapted", text_method="none", seed=0, restarts=1,
        checkpoint_interval=1, seconds=3600., max_evaluations=200000,
        dager=DAGEROptions(mode="public_residual", image_source="declared" if image == "gt" else "random",
                           projections="linear_inputs" if family == "qwen3_5" else "qkv",
                           token_selection="topk", max_candidates=100, beam_width=16,
                           vocab_chunk_size=8192, prefix_batch_size=32, lm_prior_weight=.01))


def capture_training(training):
    return replace(training, algorithm="fedsgd", local_optimizer="sgd", batch_size=1,
                   local_steps=1, gradient_accumulation_steps=1, lr=.01, weight_decay=0.,
                   knowledge="image_known", token_lengths_known=True)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        for length_bin in (row["length_bin"], "All"):
            groups[(row["model"], row["dataset"], row["image"], length_bin)].append(row)
    result = []
    metrics = ("question_em", "answer_em", "joint_em", "rouge1", "rouge2", "rougeL", "word_recall",
               "recall_at_100", "average_precision", "full_coverage", "candidate_recall", "seconds")
    for key, values in sorted(groups.items()):
        summary = dict(zip(("model", "dataset", "image", "length_bin"), key, strict=True))
        summary.update(n=len(values), completed=sum(r["status"] == "completed" for r in values),
                       statuses=dict(Counter(r["status"] for r in values)))
        for metric in metrics:
            available = [r[metric] for r in values if r.get(metric) is not None]
            summary[metric] = statistics.fmean(available) if available else None
            summary[f"{metric}_n"] = len(available)
        result.append(summary)
    return result


def evaluate_committed(output, frozen, model_label):
    """All planned reconstructions must commit before any evaluation reads truth."""
    selected = frozen["selected"]
    for index in range(len(selected)):
        for image in ("gt", "random"):
            read_json(output / f"run-{index:05d}" / "attacks" / image / "result.json")
    rows = []
    for index, sample in enumerate(selected):
        run = output / f"run-{index:05d}"
        obs = read_json(run / "captures/known/public/observation.json")
        training = TrainingSpec(**obs["training"])
        directories = {image: run / "attacks" / image for image in ("gt", "random")}
        with_filter = {key: path for key, path in directories.items()
                       if (path / "token_candidates.safetensors").exists()}
        ranking = (validate_token_filters(with_filter, run / "captures/known/private", training,
                                          topk=100, topks=(100,), random_draws=1000, seed=42 + index)
                   if with_filter else {"modes": {}})
        write_json(run / "ranking.json", ranking)
        for image, directory in directories.items():
            truth_dir = run / "captures" / ("known" if image == "gt" else "private") / "private"
            report = evaluate(directory, truth_dir, EvalSpec())
            write_json(directory / "evaluation.json", report)
            result = read_json(directory / "result.json")
            row = {"model": model_label, "dataset": frozen["dataset"], "image": image,
                   "run": index, "length_bin": sample["length_bin"], "status": result["status"],
                   "seconds": result.get("costs", {}).get("seconds"),
                   "native_lengths": sample["native_lengths"][model_label],
                   "peak_memory_bytes": result.get("benchmark_peak_memory_bytes")}
            if report["samples"]:
                metrics = report["samples"][0]["metrics"]
                row.update(question_em=metrics["question_exact_match"], answer_em=metrics["target_exact_match"])
                row["joint_em"] = row["question_em"] * row["answer_em"]
                truth = read_json(truth_dir / "truth.json")["samples"][0]
                reference = truth["model_question"] + " " + truth["model_target"]
                prediction = result["questions"][0] + " " + result["targets"][0]
                row.update({key: value for key, value in text_metrics(reference, prediction).items()
                            if key in ("rouge1", "rouge2", "rougeL", "word_recall")})
            elif result["status"] == "budget_exhausted":
                # No complete sequence is output. Count this against recovery rate.
                row.update({key: 0. for key in ("question_em", "answer_em", "joint_em", "rouge1",
                                                "rouge2", "rougeL", "word_recall")})
            if image in ranking["modes"]:
                ranked = ranking["modes"][image]["ranking"]
                row.update(recall_at_100=ranked["curve"][0]["recall"],
                           average_precision=ranked["average_precision"])
                candidates = report["token_detection"]["candidates"]
                row.update(candidate_recall=candidates["recall"], full_coverage=candidates["recall"] == 1.)
            rows.append(row)
    summary = {"schema_version": 1, "uses_private_reference": True,
               "reference_use": "post-commit evaluation only; no candidate selection or tuning",
               "rows": rows, "aggregate": aggregate(rows),
               "source_sha256": source_fingerprint(),
               "metric_policy": "Full-QA ROUGE/word recall; Joint EM requires both fields exact; "
                                "budget exhaustion counts as zero recovery; unsupported is undefined"}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-label", required=True, choices=("llava", "qwen35"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.output
    frozen = read_json(args.frozen)
    metadata = read_json(args.model / "model.json")
    if metadata["training"]["server_round"] != 10:
        raise ValueError("Text benchmark requires the Round 10 victim")
    if file_hash(frozen["manifest"]) != frozen["manifest_sha256"]:
        raise ValueError("Frozen dataset changed")
    for path, expected in frozen["tokenizer_files"][args.model_label].items():
        if file_hash(path) != expected:
            raise ValueError("Tokenizer files differ from the frozen length audit")
    expected_family = "llava" if args.model_label == "llava" else "qwen3_5"
    if metadata["model"]["family"] != expected_family:
        raise ValueError("Model label differs from checkpoint architecture")
    if any(metadata["model"][key] != value for key, value in frozen["capacity"].items()):
        raise ValueError("Victim input capacity differs from frozen sample audit")
    signature = digest({"frozen_sha256": file_hash(args.frozen), "model_sha256": file_hash(args.model / "model.json"),
                        "source_sha256": source_fingerprint(),
                        "attacks": {image: asdict(attack_spec(expected_family, image)) for image in ("gt", "random")}})
    if output.exists():
        if not args.resume or read_json(output / "state.json")["signature"] != signature:
            raise ValueError("Output exists; resume requires identical source/configuration")
    else:
        output.mkdir(parents=True)
    state = {"status": "running", "phase": "load_model", "signature": signature,
             "model": str(args.model), "frozen": str(args.frozen)}
    def progress(phase, **fields):
        state.update(phase=phase, **fields)
        write_json(output / "state.json", state)
        print(phase, fields, flush=True)
    try:
        progress("load_model")
        adapter = restore_model(args.model, "cuda:0")
        fingerprint = metadata["fingerprint"]
        training = capture_training(adapter.training_spec)
        from core.data import read_manifest
        by_id = {row["sample_id"]: row for row in read_manifest(frozen["manifest"], task="vqa", split="eval")}
        for index, sample in enumerate(frozen["selected"]):
            run = output / f"run-{index:05d}"
            progress("capture", run=index)
            if (run / "capture_complete.json").exists():
                continue
            if run.exists():
                raise ValueError("Incomplete capture; inspect artifacts before creating a replacement")
            capture_pair(adapter, by_id[sample["sample_id"]], training, run, args.model)
            write_json(run / "capture_complete.json", {"status": "completed"})
        del by_id
        gc.collect()
        for index in range(len(frozen["selected"])):
            run = output / f"run-{index:05d}"
            for image in ("gt", "random"):
                directory = run / "attacks" / image
                if (directory / "result.json").exists():
                    continue
                if directory.exists():
                    raise ValueError("Incomplete attack; preserve and audit before a new attempt")
                progress("attack", run=index, image=image)
                config = Config(model=ModelSpec(**metadata["model"]), training=training,
                                attack=attack_spec(expected_family, image))
                torch.cuda.reset_peak_memory_stats()
                public = run / "captures" / ("known" if image == "gt" else "private") / "public"
                started = time.monotonic()
                status, _ = attack_public(adapter, config, public, directory, fingerprint)
                result = read_json(directory / "result.json")
                result.update(benchmark_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                              benchmark_wall_seconds=time.monotonic() - started)
                write_json(directory / "result.json", result)
                if status not in {"completed", "budget_exhausted"}:
                    raise ValueError(f"Unexpected unsupported attack condition: {status}")
        progress("evaluate")
        evaluate_committed(output, frozen, args.model_label)
        state["status"] = "completed"
        progress("finished")
    except BaseException:
        state.update(status="failed", traceback=traceback.format_exc())
        progress("failed")
        raise


if __name__ == "__main__":
    main()
