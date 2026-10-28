"""Compare SGD minibatch partitions of the same four-example client upload.

Victim capture, public-only attack and post-commit evaluation are separate CLI
phases. All conditions preserve the pretrained Round-10 values and use bf16
compute with FP32 LoRA parameters and uploads.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import Config
from core.data import load_batch, read_manifest
from core.fl import load_observation
from evaluation.dager_rank_study import attack_public
from evaluation.federated_text_pilot import evaluate_pilot, pilot_attack, save_pair, select_cohort
from evaluation.fp32_multistep_study import (
    captured_observation, precision_context, precision_settings, restore_mixed, trace_local_update,
)
from evaluation.text_benchmark import capture_training


CONDITIONS = (("fixed4-b1-s4", 1, 4), ("fixed4-b2-s2", 2, 2), ("fixed4-b4-s1", 4, 1))
SELECTION_RULE = "exclude last frozen short and last frozen medium QA; preserve remaining order"


def select_fixed_cohort(frozen, manifest_rows):
    """Selection uses only frozen order, length bins and client ownership."""
    original = select_cohort(frozen, "slake30")
    excluded = {next(i for i in range(len(original) - 1, -1, -1)
                     if original[i]["length_bin"] == length_bin)
                for length_bin in ("1-16", "17-32")}
    selected = [row for i, row in enumerate(original) if i not in excluded]
    lookup = {row["sample_id"]: row for row in manifest_rows}
    rows = [lookup[row["sample_id"]] for row in selected]
    validate_groups(rows)
    if len(rows) != 28 or Counter(row["length_bin"] for row in selected) != {
            "1-16": 9, "17-32": 9, "33-64": 10}:
        raise ValueError("Expected the predetermined 28 QA cohort")
    return selected, rows


def validate_groups(rows):
    if not rows or len(rows) % 4 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("Each upload needs four distinct QA; no partial or repeated groups")
    for start in range(0, len(rows), 4):
        clients = {row["client"] for row in rows[start:start + 4]}
        if len(clients) != 1:
            raise ValueError("Four-QA uploads cannot mix different clients")


def attack_spec(image):
    spec = pilot_attack(image, 4)
    if (spec.dager.rerank_candidates != 0 or spec.dager.lm_prior_weight != .01
            or spec.dager.beam_width != 16 or spec.dager.max_candidates != 100):
        raise ValueError("Frozen SGD attack settings changed")
    return spec


def check_trace(update, trace, training):
    if training.sample_count != 4 or len(trace) != training.local_steps:
        raise ValueError("Expected four examples, one fresh forward per local step")
    if trace[0]["selected_A_distance_from_initial"] != 0 or any(
            row["selected_A_distance_from_initial"] <= 0 for row in trace[1:]):
        raise ValueError("Every later step must use updated parameters")
    if any(row["parameter_dtype"] != "torch.float32" or
           row["selected_projection_output_dtype"] != "torch.bfloat16" for row in trace):
        raise ValueError("Expected bf16 compute and FP32 LoRA parameters")
    if not update or any(v.dtype != torch.float32 or not torch.isfinite(v).all()
                         for v in update.values()):
        raise ValueError("Upload must be finite FP32")


def native_smoke(adapter, training, names):
    """Compare functional replay with native SGD on four public synthetic QA."""
    adapter.set_training_spec(training)
    size = adapter.spec.image_size
    batch = adapter.batch(torch.full((4, 3, size, size), .5),
                          ["what is shown?", "what color?", "what is shown?", "what color?"],
                          ["unknown", "red", "unknown", "blue"])
    before = {name: p.detach().clone() for name, p in adapter.trainable().items()}
    with precision_context("mixed", adapter.device.type):
        expected, trace = trace_local_update(adapter, batch, training, names)
    check_trace(expected, trace, training)
    optimizer = torch.optim.SGD(adapter.trainable().values(), lr=training.lr,
                                foreach=False, fused=False)
    try:
        for step in range(training.local_steps):
            optimizer.zero_grad(set_to_none=True)
            part = batch.slice(step * training.batch_size, (step + 1) * training.batch_size)
            with precision_context("mixed", adapter.device.type):
                loss = adapter(part.images, part.questions, part.targets)
            loss.backward()
            if any(p.grad is not None and p.grad.dtype != torch.float32
                   for p in adapter.trainable().values()):
                raise ValueError("Native LoRA gradients lost FP32 precision")
            optimizer.step()
        for name, p in adapter.trainable().items():
            torch.testing.assert_close(p.detach() - before[name], expected[name], rtol=0, atol=0)
    finally:
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in adapter.trainable().items():
                p.copy_(before[name])
    return {"status": "passed", "batch": training.batch_size, "steps": training.local_steps,
            "native_sgd_exact": True, "trace": trace, "benchmark_result": False,
            "input": "four public constant images and synthetic QA"}


def prepare(output, model, frozen_path):
    if (output / "selection.json").exists():
        raise FileExistsError("Selection is immutable; use a new output directory")
    frozen = read_json(frozen_path)
    if file_hash(frozen["manifest"]) != frozen["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    selected, rows = select_fixed_cohort(frozen, read_manifest(frozen["manifest"], "vqa", "eval"))
    metadata = read_json(model / "model.json")
    if metadata["round"] != 10 or metadata["training"]["lora_rank"] != 32:
        raise ValueError("Expected the original Round-10 rank-32 public model")
    # IDs and client ownership are victim-side scheduling data. Attack workers
    # never open selection.json or the original frozen manifest.
    write_json(output / "selection.json", {
        "schema_version": 1, "source_sha256": source_fingerprint(),
        "model": str(model.resolve()), "model_metadata_sha256": file_hash(model / "model.json"),
        "frozen": str(frozen_path.resolve()), "frozen_sha256": file_hash(frozen_path),
        "manifest_sha256": frozen["manifest_sha256"], "rule": SELECTION_RULE,
        "selected": selected, "cohort_n": len(rows), "upload_size": 4,
        "length_counts": dict(Counter(row["length_bin"] for row in selected)),
        "groups": [{"sample_ids": [r["sample_id"] for r in rows[start:start + 4]],
                    "client": rows[start]["client"]} for start in range(0, len(rows), 4)],
        "conditions": [{"condition": name, "batch": batch, "steps": steps}
                       for name, batch, steps in CONDITIONS],
        "attack_specs": {image: asdict(attack_spec(image)) for image in ("gt", "random")}})


def capture_all(output):
    if (output / "runs").exists() or (output / "plan.json").exists():
        raise FileExistsError("No implicit capture retry")
    selection = read_json(output / "selection.json")
    if source_fingerprint() != selection["source_sha256"]:
        raise ValueError("Source changed after sample selection")
    model, frozen_path = Path(selection["model"]), Path(selection["frozen"])
    if (file_hash(model / "model.json") != selection["model_metadata_sha256"] or
            file_hash(frozen_path) != selection["frozen_sha256"]):
        raise ValueError("Parent model or frozen dataset changed")
    frozen = read_json(frozen_path)
    if file_hash(frozen["manifest"]) != selection["manifest_sha256"]:
        raise ValueError("Victim manifest changed")
    selected, rows = select_fixed_cohort(frozen, read_manifest(frozen["manifest"], "vqa", "eval"))
    if selected != selection["selected"]:
        raise ValueError("Predetermined cohort changed")
    settings = precision_settings("mixed")
    adapter = restore_mixed(model, "cuda:0")
    fingerprint = adapter.fingerprint()
    base = replace(capture_training(adapter.training_spec), algorithm="fedavg", local_optimizer="sgd",
                   lr=.01, weight_decay=0., gradient_accumulation_steps=1)
    names = [name for name in adapter.trainable() if any(
        f"layers.{layer}.self_attn.{projection}_proj.lora_A." in name
        for layer in (0, 1) for projection in "qkv")]
    if len(names) != 6:
        raise ValueError("Expected six selected LoRA-A matrices")
    write_json(output / "derived-model.json", {
        "parent_model_metadata_sha256": selection["model_metadata_sha256"],
        "model": asdict(adapter.spec), "fingerprint": fingerprint, "precision": settings,
        "conversion": "restore original bf16 snapshot; promote only trainable LoRA to FP32",
        "new_training_rounds": 0})
    smokes = []
    for condition, batch_size, steps in CONDITIONS:
        training = replace(base, batch_size=batch_size, local_steps=steps)
        torch.cuda.reset_peak_memory_stats()
        smoke = native_smoke(adapter, training, names)
        smoke.update(condition=condition, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     peak_reserved_bytes=torch.cuda.max_memory_reserved())
        smokes.append(smoke)
        write_json(output / "public-smoke.json", {"status": "running", "conditions": smokes,
                                                  "benchmark_result": False})
        torch.cuda.empty_cache()
    if adapter.fingerprint() != fingerprint:
        raise ValueError("Public smoke did not restore initial model")
    write_json(output / "public-smoke.json", {"status": "passed", "conditions": smokes,
                                              "benchmark_result": False})
    versions = {name: p._version for name, p in adapter.named_parameters()}
    jobs = []
    for condition, batch_size, steps in CONDITIONS:
        training = replace(base, batch_size=batch_size, local_steps=steps)
        adapter.set_training_spec(training)
        for group_index, start in enumerate(range(0, len(rows), 4)):
            job_id = f"{condition}-{group_index:02d}"
            print("capture", job_id, flush=True)
            group = rows[start:start + 4]
            batch = load_batch(group, adapter)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with precision_context("mixed"):
                update, trace = trace_local_update(adapter, batch, training, names)
            check_trace(update, trace, training)
            if any(p._version != versions[name] for name, p in adapter.named_parameters()):
                raise ValueError("Capture mutated initial model")
            observed = captured_observation(adapter, batch, training, update, fingerprint)
            root = output / "runs" / job_id
            save_pair(adapter, batch, group, observed, root, model, {"name": "none"})
            write_json(root / "local-trace.json", {
                "trace": trace, "steps": steps, "batch": batch_size, "unique_samples": 4,
                "fresh_forward_backward_per_step": True, "precision": settings})
            write_json(root / "capture-runtime.json", {
                "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            write_json(root / "victim_diagnostics.json", {"relative_l2_change": 0.})
            jobs.append({"id": job_id, "condition": condition, "n": 4, "group_index": group_index,
                         "batch_size": batch_size, "local_steps": steps, "algorithm": "fedavg",
                         "length_bins": [r["length_bin"] for r in selected[start:start + 4]]})
            write_json(output / "progress-capture.json", {
                "status": "running", "completed": len(jobs), "total": 21, "job": job_id})
            del batch, update, observed
            torch.cuda.empty_cache()
    if adapter.fingerprint() != fingerprint:
        raise ValueError("Initial model changed across captures")
    write_json(output / "plan.json", {
        "schema_version": 1, "source_sha256": source_fingerprint(), "cohort_n": 28,
        "attack_count": 42, "jobs": jobs, "model": str(model.resolve()),
        "model_fingerprint": fingerprint, "parent_model_metadata_sha256": selection["model_metadata_sha256"],
        "derived_model_sha256": file_hash(output / "derived-model.json"), "precision": settings,
        "initial_model_immutable": True, "group_size": 4, "selection_rule": SELECTION_RULE,
        "note": "Same four distinct QA per upload, same order, one pass without reuse. "
                "Fixed SGD lr=0.01: partitions change loss weighting and cumulative update scale. "
                "No new training rounds; public token lengths; fixed K=100, beam=16, rerank=0.",
        "attack_specs": selection["attack_specs"]})
    write_json(output / "progress-capture.json", {"status": "completed", "completed": 21, "total": 21})


def verify_plan(output):
    plan = read_json(output / "plan.json")
    if source_fingerprint() != plan["source_sha256"]:
        raise ValueError("Frozen source changed")
    if file_hash(Path(plan["model"]) / "model.json") != plan["parent_model_metadata_sha256"]:
        raise ValueError("Parent model changed")
    if file_hash(output / "derived-model.json") != plan["derived_model_sha256"]:
        raise ValueError("Model derivation changed")
    return plan


def attack_all(output, image):
    plan = verify_plan(output)
    precision_settings("mixed")
    adapter = restore_mixed(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Attacker model differs from capture model")
    for index, job in enumerate(plan["jobs"]):
        root = output / "runs" / job["id"]
        public, directory = root / "captures" / image / "public", root / "attacks" / image
        if directory.exists():
            raise FileExistsError("No implicit attack retry or resume")
        observed = load_observation(public, adapter.spec.device)
        if (observed.model.dtype != "bfloat16" or observed.training.sample_count != 4
                or observed.training.local_optimizer != "sgd" or observed.training.lr != .01
                or any(v.dtype != torch.float32 for v in observed.tensors.values())):
            raise ValueError("Unexpected public SGD condition or upload precision")
        spec = attack_spec(image)
        if asdict(spec) != plan["attack_specs"][image]:
            raise ValueError("Attack settings changed after selection")
        config = Config(model=observed.model, training=observed.training, attack=spec)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        write_json(output / f"progress-{image}.json", {
            "status": "running", "completed": index, "total": 21, "job": job["id"]})
        print("attack", image, job["id"], flush=True)
        with precision_context("mixed"):
            status, _ = attack_public(adapter, config, public, directory, plan["model_fingerprint"],
                                      upload_metadata={"defense": {"name": "none"}})
        if status not in {"completed", "budget_exhausted", "no_signal", "no_candidates"}:
            raise ValueError("Unsupported fixed-N condition: " + status)
        result = read_json(directory / "result.json")
        result["condition"].update(precision_mode="mixed", trainable_dtype="float32",
                                   optimizer_dtype="float32", upload_dtype="float32", tf32=False,
                                   autocast_bf16=True, autocast_cache_enabled=False, unique_samples=4)
        write_json(directory / "result.json", result)
        write_json(directory / "runtime.json", {
            "seconds": time.monotonic() - started, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print("finished", image, job["id"], status, flush=True)
    write_json(output / f"progress-{image}.json", {"status": "completed", "completed": 21, "total": 21})


def evaluate_all(output):
    verify_plan(output)
    return evaluate_pilot(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "capture", "attack", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--image", choices=("gt", "random"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}-{args.image or 'all'}.json"
    if state.exists():
        raise FileExistsError("Inspect existing phase state; no automatic retry")
    write_json(state, {"status": "running", "phase": args.phase})
    try:
        if args.phase == "prepare":
            if args.model is None or args.frozen is None:
                raise ValueError("Preparation requires parent model and frozen cohort")
            prepare(args.output, args.model, args.frozen)
        elif args.phase == "capture":
            capture_all(args.output)
        elif args.phase == "attack":
            if args.image is None:
                raise ValueError("Attack requires an image condition")
            attack_all(args.output, args.image)
        else:
            evaluate_all(args.output)
        write_json(state, {"status": "completed", "phase": args.phase})
    except BaseException:
        write_json(state, {"status": "failed", "phase": args.phase, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
