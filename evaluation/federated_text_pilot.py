"""Paired benchmark for pooled SGD uploads and upload perturbations.

Capture, attack, and post-commit evaluation are separate processes. The attack
worker reads only saved public observations, never the frozen dataset or truth.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import math
from pathlib import Path
import statistics
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import Config, EvalSpec, ModelSpec
from core.data import load_batch, read_manifest
from core.fl import capture, load_observation, restore_model, save_observation
from defenses.factory import create_defense
from evaluation.dager_rank_study import attack_public
from evaluation.reconstruction import evaluate
from evaluation.text_benchmark import attack_spec, capture_training
from metrics.text import text_metrics


CONDITIONS = (
    ("fedsgd-b1", "fedsgd", 1, 1, {"name": "none"}),
    ("fedsgd-b2", "fedsgd", 2, 1, {"name": "none"}),
    ("fedavg-b2-s1", "fedavg", 2, 1, {"name": "none"}),
    ("fedavg-b1-s2", "fedavg", 1, 2, {"name": "none"}),
    ("clipping", "fedsgd", 1, 1, {"name": "clipping"}),
    ("noise", "fedsgd", 1, 1, {"name": "gaussian_noise", "noise_ratio": .01}),
    ("sparsify", "fedsgd", 1, 1, {"name": "topk_sparsify", "ratio": .1}),
)


def select_pilot(frozen):
    """Two short and two medium QA, selected before reading attack results."""
    bins = [[row for row in frozen["selected"] if row["length_bin"] == key][:2]
            for key in ("1-16", "17-32")]
    if any(len(rows) != 2 for rows in bins):
        raise ValueError("Pilot requires two frozen samples in each of the first two bins")
    return [row for pair in zip(*bins, strict=True) for row in pair]


def select_cohort(frozen, cohort):
    if cohort == "pilot4":
        return select_pilot(frozen)
    if cohort != "slake30" or frozen.get("dataset") != "slake":
        raise ValueError("Expected the frozen SLAKE 30-sample cohort")
    selected = frozen["selected"]
    if (len(selected) != 30 or len({r["sample_id"] for r in selected}) != 30
            or any(sum(r["length_bin"] == b for r in selected) != 10
                   for b in ("1-16", "17-32", "33-64"))):
        raise ValueError("SLAKE cohort must contain 30 distinct QA, ten in each original length bin")
    return list(selected)


def validate_upload_groups(rows):
    if len(rows) % 2:
        raise ValueError("The paired upload conditions require an even cohort size")
    for start in range(0, len(rows), 2):
        pair = rows[start:start + 2]
        if any("client" not in row for row in pair) or pair[0]["client"] != pair[1]["client"]:
            raise ValueError("A single-client upload cannot mix rows from different clients")


def pilot_attack(image, n):
    base = attack_spec("llava", image)
    # Equal budget per content sample; one joint beam per observed upload.
    return replace(base, seconds=base.seconds * n, max_evaluations=base.max_evaluations * n,
                   dager=replace(base.dager, pooled_updates=True, defense_mode="unaware"))


def update_norm(values):
    return sum(value.detach().double().square().sum().item() for value in values.values()) ** .5


def save_pair(adapter, batch, rows, observed, folder, model, defense):
    private = replace(observed, training=replace(observed.training, knowledge="private"), public_images=None)
    records = [dict(row, model_question=q, model_target=a) for row, q, a in zip(
        rows, adapter.decode(batch.questions), adapter.decode(batch.targets), strict=True)]
    for image, obs in (("gt", observed), ("random", private)):
        root = folder / "captures" / image
        observation_id = save_observation(root / "public", obs)
        write_json(root / "public/model_ref.json", {"path": str(model.resolve())})
        write_json(root / "public/upload.json", {"defense": defense})
        write_tensors(root / "private/images.safetensors", {"images": batch.images})
        write_tensors(root / "private/text_tokens.safetensors", {
            "questions": batch.questions, "targets": batch.targets})
        write_json(root / "private/truth.json", {
            "observation_id": observation_id, "training": asdict(obs.training), "samples": records})
    left = read_json(folder / "captures/gt/public/observation.json")
    right = read_json(folder / "captures/random/public/observation.json")
    if left["update_sha256"] != right["update_sha256"]:
        raise ValueError("GT and Random must receive exactly the same defended upload")


def capture_pilot(output, frozen_path, model, cohort="pilot4", clip_max_norm=None):
    if (output / "plan.json").exists() or (output / "runs").exists():
        raise FileExistsError("Pilot capture is immutable; use a fresh directory")
    if clip_max_norm is not None and (not math.isfinite(clip_max_norm) or clip_max_norm <= 0):
        raise ValueError("Fixed clipping norm must be finite and positive")
    if cohort == "slake30" and clip_max_norm is None:
        raise ValueError("SLAKE expansion requires a clipping threshold fixed before this cohort")
    frozen = read_json(frozen_path)
    if file_hash(frozen["manifest"]) != frozen["manifest_sha256"]:
        raise ValueError("Frozen dataset changed")
    selected = select_cohort(frozen, cohort)
    rows_by_id = {row["sample_id"]: row for row in read_manifest(frozen["manifest"], task="vqa", split="eval")}
    rows = [rows_by_id[row["sample_id"]] for row in selected]
    validate_upload_groups(rows)
    metadata = read_json(model / "model.json")
    if metadata["round"] != 10:
        raise ValueError("This benchmark requires the existing Round 10 model")
    adapter = restore_model(model, "cuda:0")
    if adapter.spec.family != "llava" or adapter.training_spec.lora_rank != 32:
        raise ValueError("Pilot is frozen to LLaVA rank 32")
    base = capture_training(adapter.training_spec)
    jobs, baseline_norms = [], []
    for condition, algorithm, batch_size, steps, defense_template in CONDITIONS:
        training = replace(base, algorithm=algorithm, batch_size=batch_size, local_steps=steps)
        n = training.sample_count
        for start in range(0, len(rows), n):
            job_id = f"{condition}-{start // n:02d}"
            folder = output / "runs" / job_id
            print("capture", job_id, flush=True)
            adapter.set_training_spec(training)
            batch = load_batch(rows[start:start + n], adapter)
            observed = capture(adapter, batch, [], [])
            clean_norm = update_norm(observed.tensors)
            if condition == "fedsgd-b1":
                baseline_norms.append(clean_norm)
            defense = dict(defense_template)
            if defense["name"] == "clipping":
                # One common C, calibrated before attacks from the four raw uploads.
                # This is a pilot calibration, not a claimed private DP mechanism.
                defense["max_norm"] = (clip_max_norm if clip_max_norm is not None
                                       else .5 * statistics.median(baseline_norms))
            seed = 4200 + start
            defended = create_defense(defense).apply(
                observed.tensors, generator=torch.Generator().manual_seed(seed))
            difference = {key: value.double() - observed.tensors[key].double() for key, value in defended.items()}
            perturbation = {"clean_norm": clean_norm, "uploaded_norm": update_norm(defended),
                            "relative_l2_change": update_norm(difference) / max(clean_norm, 1e-30),
                            "nonzero_fraction": sum(v.count_nonzero().item() for v in defended.values()) /
                            sum(v.numel() for v in defended.values()), "defense_seed": seed}
            save_pair(adapter, batch, rows[start:start + n], replace(observed, tensors=defended),
                      folder, model, defense)
            write_json(folder / "victim_diagnostics.json", perturbation)
            jobs.append({"id": job_id, "condition": condition, "n": n,
                         "batch_size": batch_size, "local_steps": steps,
                         "algorithm": algorithm, "defense": defense,
                         "length_bins": [row["length_bin"] for row in selected[start:start + n]]})
            write_json(output / "progress-capture.json", {
                "status": "running", "completed": len(jobs), "job": job_id,
                "total": sum(len(rows) // (batch * steps) for _, _, batch, steps, _ in CONDITIONS)})
    write_json(output / "plan.json", {
        "schema_version": 1, "status": "captured", "source_sha256": source_fingerprint(),
        "model": str(model.resolve()), "model_metadata_sha256": file_hash(model / "model.json"),
        "model_fingerprint": metadata["fingerprint"], "frozen_sha256": file_hash(frozen_path),
        "cohort_n": len(rows), "cohort": cohort,
        "selection": ("first two short and first two medium frozen QA, interleaved"
                      if cohort == "pilot4" else "all 30 frozen SLAKE QA in original order; adjacent pairs"),
        "n_unique_per_upload": "batch_size * local_steps; one pass, no sample reuse",
        "clip_calibration": ("fixed before cohort capture" if clip_max_norm is not None else
                             "C = half median global norm of baseline uploads, frozen before attacks"),
        "clip_max_norm": clip_max_norm if clip_max_norm is not None else .5 * statistics.median(baseline_norms),
        "matching": "post-commit one-to-one text Hungarian assignment; no reference in attack",
        "jobs": jobs, "attack_count": len(jobs) * 2,
        "attack_specs": {str(n): {image: asdict(pilot_attack(image, n)) for image in ("gt", "random")}
                         for n in (1, 2)}})
    write_json(output / "progress-capture.json", {
        "status": "completed", "completed": len(jobs), "total": len(jobs)})


def attack_pilot(output, image):
    plan = read_json(output / "plan.json")
    if plan["source_sha256"] != source_fingerprint():
        raise ValueError("Pilot source differs from frozen capture")
    model = Path(plan["model"])
    if file_hash(model / "model.json") != plan["model_metadata_sha256"]:
        raise ValueError("Model metadata changed")
    adapter = restore_model(model, "cuda:0")
    for index, job in enumerate(plan["jobs"]):
        run = output / "runs" / job["id"]
        directory = run / "attacks" / image
        # No implicit resume/retry, including OOM or partly completed searches.
        if directory.exists():
            raise FileExistsError(directory)
        public = run / "captures" / image / "public"
        obs = load_observation(public)
        config = Config(model=ModelSpec(**asdict(obs.model)), training=obs.training,
                        attack=pilot_attack(image, job["n"]))
        write_json(output / f"progress-{image}.json", {
            "status": "running", "completed": index, "total": len(plan["jobs"]), "job": job["id"]})
        print("attack", image, job["id"], flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        status, _ = attack_public(adapter, config, public, directory, plan["model_fingerprint"],
                                  upload_metadata=read_json(public / "upload.json"))
        write_json(directory / "runtime.json", {"seconds": time.monotonic() - started,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated()})
        if status not in {"completed", "budget_exhausted", "no_signal", "no_candidates"}:
            raise ValueError(f"Unsupported pilot condition: {status}")
        print("finished", image, job["id"], status, flush=True)
    write_json(output / f"progress-{image}.json", {
        "status": "completed", "completed": len(plan["jobs"]), "total": len(plan["jobs"])})


def aggregate_rows(rows, by_length=False):
    groups = defaultdict(list)
    for row in rows:
        key = (row["condition"], row["image"], row["length_bin"] if by_length else "All")
        groups[key].append(row)
    aggregate = []
    keys = ("question_em", "answer_em", "joint_em", "rouge1", "rouge2", "rougeL", "word_recall",
            "full_coverage", "candidate_recall", "relative_l2_change", "seconds_per_sample")
    for (condition, image, length_bin), values in groups.items():
        aggregate.append({"condition": condition, "image": image, "length_bin": length_bin,
                          "n": len(values), "uploads": len({v["job"] for v in values}),
                          "completed_samples": sum(v["status"] == "completed" for v in values),
                          **{key: statistics.fmean(v[key] for v in values if v.get(key) is not None)
                             if any(v.get(key) is not None for v in values) else None for key in keys}})
    return aggregate


def evaluate_pilot(output):
    plan = read_json(output / "plan.json")
    for job in plan["jobs"]:
        for image in ("gt", "random"):
            read_json(output / "runs" / job["id"] / "attacks" / image / "result.json")
    rows = []
    for job in plan["jobs"]:
        run = output / "runs" / job["id"]
        perturbation = read_json(run / "victim_diagnostics.json")
        for image in ("gt", "random"):
            directory = run / "attacks" / image
            truth_dir = run / "captures" / image / "private"
            report = evaluate(directory, truth_dir, EvalSpec())
            write_json(directory / "evaluation.json", report)
            result = read_json(directory / "result.json")
            truth = read_json(truth_dir / "truth.json")["samples"]
            candidates = report.get("token_detection", {}).get("candidates", {})
            token_data = read_tensors(directory / "token_candidates.safetensors") if (
                directory / "token_candidates.safetensors").exists() else None
            references = read_tensors(truth_dir / "text_tokens.safetensors")
            forbidden = set(read_json(directory / "token_candidates.json")["forbidden_ids"]) if token_data else set()
            selected = set(token_data["selected_ids"].tolist()) if token_data else set()
            for index, target in enumerate(truth):
                metrics = {key: 0. for key in (
                    "question_em", "answer_em", "joint_em", "rouge1", "rouge2", "rougeL", "word_recall")}
                if report["samples"]:
                    item = report["samples"][index]
                    predicted = item["prediction_index"]
                    metrics.update(question_em=item["metrics"]["question_exact_match"],
                                   answer_em=item["metrics"]["target_exact_match"])
                    metrics["joint_em"] = metrics["question_em"] * metrics["answer_em"]
                    full = text_metrics(target["model_question"] + " " + target["model_target"],
                                        result["questions"][predicted] + " " + result["targets"][predicted])
                    metrics.update({key: full[key] for key in ("rouge1", "rouge2", "rougeL", "word_recall")})
                tokens = set(references["questions"][index].tolist() + references["targets"][index].tolist()) - forbidden
                rows.append({"condition": job["condition"], "image": image, "job": job["id"],
                             "slot": index, "status": result["status"], "n_per_upload": job["n"],
                             "length_bin": job["length_bins"][index], **metrics,
                             "full_coverage": tokens <= selected if token_data else None,
                             "candidate_recall": len(tokens & selected) / len(tokens) if token_data and tokens else None,
                             "union_candidate_recall": candidates.get("recall"),
                             "relative_l2_change": perturbation["relative_l2_change"],
                             "seconds_per_sample": result["costs"]["seconds"] / job["n"]})
    summary = {"schema_version": 1, "cohort_n": plan["cohort_n"], "attack_count": plan["attack_count"],
               "source_sha256": plan["source_sha256"], "aggregate": aggregate_rows(rows), "rows": rows,
               "by_length": aggregate_rows(rows, by_length=True),
               "note": "Frozen cohort; group text recovery uses one-to-one matching. No DP guarantee."}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "attack", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--image", choices=("gt", "random"))
    parser.add_argument("--cohort", choices=("pilot4", "slake30"), default="pilot4")
    parser.add_argument("--clip-max-norm", type=float)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state_path = args.output / f"state-{args.phase}-{args.image or 'all'}.json"
    write_json(state_path, {"status": "running", "phase": args.phase})
    try:
        if args.phase == "capture":
            if args.model is None or args.frozen is None:
                raise ValueError("Capture requires --model and --frozen")
            capture_pilot(args.output, args.frozen, args.model, args.cohort, args.clip_max_norm)
        elif args.phase == "attack":
            if args.image is None:
                raise ValueError("Attack requires --image")
            attack_pilot(args.output, args.image)
        else:
            evaluate_pilot(args.output)
        write_json(state_path, {"status": "completed", "phase": args.phase})
    except BaseException:
        write_json(state_path, {"status": "failed", "phase": args.phase, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
