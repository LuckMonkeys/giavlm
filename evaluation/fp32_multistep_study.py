"""Fresh FP32 forward/backward and local SGD on the frozen SLAKE cohort.

All model values originate from the same public Round-10 bf16 snapshot, promoted
exactly to FP32. No new training rounds or higher-precision pretrained weights
are substituted. Each local step evaluates a fresh gradient at current weights.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json
from core.config import Config
from core.data import load_batch, read_manifest
from core.fl import load_observation, restore_model, simulate_update
from core.types import Observation
from evaluation.dager_rank_study import attack_public
from evaluation.federated_text_pilot import (
    evaluate_pilot, pilot_attack, save_pair, select_cohort, validate_upload_groups,
)
from evaluation.text_benchmark import capture_training


CONDITIONS = (
    ("fp32-gradient", "fedsgd", 2, 1),
    ("fp32-one-step", "fedavg", 2, 1),
    ("fp32-two-steps", "fedavg", 1, 2),
)
MIXED_CONDITIONS = tuple((name.replace("fp32-", "mixed-"), algorithm, batch, steps)
                         for name, algorithm, batch, steps in CONDITIONS)


def require_mixed(adapter):
    trainable = adapter.trainable()
    if not trainable or any(".lora_" not in name or p.dtype != torch.float32
                            for name, p in trainable.items()):
        raise ValueError("Mixed mode requires every trainable LoRA parameter in FP32")
    if any(p.dtype != torch.bfloat16 for p in adapter.parameters() if not p.requires_grad):
        raise ValueError("Mixed mode requires frozen parameters in bf16")
    if adapter.spec.dtype != "bfloat16" or adapter.dtype != torch.bfloat16:
        raise ValueError("Mixed mode public model must declare bf16 backbone")


def restore_mixed(model, device):
    adapter = restore_model(model, device)
    for parameter in adapter.trainable().values():
        parameter.data = parameter.data.float()
    require_mixed(adapter)
    return adapter


def precision_context(mode, device_type="cuda"):
    # Functional local steps replace parameters. Never cache casts of the old
    # FP32 LoRA tensors across steps (or across read-only attack forwards).
    return (torch.autocast(device_type, dtype=torch.bfloat16, cache_enabled=False)
            if mode == "mixed" else nullcontext())


def require_fp32(adapter):
    parameters = list(adapter.parameters())
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("Every model parameter must be FP32")
    if any(b.is_floating_point() and b.dtype != torch.float32 for b in adapter.buffers()):
        raise ValueError("All floating model buffers must be FP32")
    if adapter.spec.dtype != "float32" or adapter.dtype != torch.float32:
        raise ValueError("Public model dtype differs from actual parameters")


def restore_fp32(model, device):
    # Restore and verify the original bf16 overlay first: loading its original
    # checkpoint directly as FP32 could substitute different frozen base values.
    adapter = restore_model(model, device)
    adapter.float()
    adapter.spec = replace(adapter.spec, dtype="float32")
    require_fp32(adapter)
    return adapter


def precision_settings(mode="fp32"):
    if mode not in {"fp32", "mixed"}:
        raise ValueError("Unknown precision mode")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {"parameters": "float32" if mode == "fp32" else "frozen bf16; trainable LoRA float32",
            "forward_backward": "float32" if mode == "fp32" else "bf16 autocast; FP32 LoRA gradients",
            "upload": "float32", "optimizer": "float32", "mode": mode,
            "autocast": mode == "mixed", "autocast_cache_enabled": False,
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def trace_local_update(adapter, batch, training, selected_names):
    """Observe the existing functional replay; do not replace any gradients."""
    baseline = {name: p.detach().clone() for name, p in adapter.named_parameters()
                if name in selected_names}
    trace = []

    def observe(module, args):
        current = dict(module.named_parameters())
        difference = sum((current[name].detach().double() - value.double()).square().sum()
                         for name, value in baseline.items())
        trace.append({"forward_index": len(trace), "selected_A_distance_from_initial":
                      float(difference.sqrt()), "parameter_dtype": str(current[selected_names[0]].dtype)})

    handle = adapter.register_forward_pre_hook(observe)
    projection_name = selected_names[0].split(".lora_A.")[0]
    projection = dict(adapter.named_modules())[projection_name]

    def probe(module, args, result):
        trace[-1]["selected_projection_output_dtype"] = str(result.dtype)

    probe_handle = projection.register_forward_hook(probe)
    try:
        update = simulate_update(adapter, batch, training, differentiable=False)
    finally:
        handle.remove()
        probe_handle.remove()
    expected = training.local_steps * training.gradient_accumulation_steps
    if len(trace) != expected or trace[0]["selected_A_distance_from_initial"] != 0:
        raise ValueError("Unexpected local forward/gradient count or initial weights")
    if training.local_steps > 1 and trace[1]["selected_A_distance_from_initial"] <= 0:
        raise ValueError("Second local step did not use changed FP32 parameters")
    if any(value.dtype != torch.float32 for value in update.values()):
        raise ValueError("Local upload lost FP32 precision")
    return {name: value.detach() for name, value in update.items()}, trace


def captured_observation(adapter, batch, training, update, fingerprint):
    # The fingerprint is computed once for the immutable public initial model,
    # then independently checked again after all captures.
    result = Observation(model=replace(adapter.spec), training=replace(training),
        tensors=update, model_fingerprint=fingerprint,
        public_questions=[], public_targets=[], public_question_ids=[], public_target_ids=[],
        public_question_lengths=adapter.content_lengths(batch.questions),
        public_target_lengths=adapter.content_lengths(batch.targets),
        public_images=batch.images.detach().float().cpu().clone())
    result.validate()
    return result


def capture_all(output, model, frozen_path, precision="fp32"):
    if (output / "runs").exists() or (output / "plan.json").exists():
        raise FileExistsError("Use a fresh FP32 multi-step output directory")
    settings = precision_settings(precision)
    conditions = MIXED_CONDITIONS if precision == "mixed" else CONDITIONS
    conversion = ("restore original verified bf16 snapshot, promote only trainable LoRA to FP32"
                  if precision == "mixed" else
                  "restore original verified bf16 snapshot, then promote every value to FP32")
    frozen = read_json(frozen_path)
    if file_hash(frozen["manifest"]) != frozen["manifest_sha256"]:
        raise ValueError("Frozen cohort manifest changed")
    selected = select_cohort(frozen, "slake30")
    lookup = {r["sample_id"]: r for r in read_manifest(frozen["manifest"], task="vqa", split="eval")}
    rows = [lookup[r["sample_id"]] for r in selected]
    validate_upload_groups(rows)
    parent = read_json(model / "model.json")
    if parent["round"] != 10 or parent["training"]["lora_rank"] != 32:
        raise ValueError("Expected original Round-10 rank-32 public model")
    adapter = (restore_mixed if precision == "mixed" else restore_fp32)(model, "cuda:0")
    fingerprint = adapter.fingerprint()
    versions = {name: p._version for name, p in adapter.named_parameters()}
    base = capture_training(adapter.training_spec)
    names = [name for name in adapter.trainable() if any(
        f"layers.{layer}.self_attn.{projection}_proj.lora_A." in name
        for layer in (0, 1) for projection in "qkv")]
    if len(names) != 6:
        raise ValueError("Expected six first-layer LoRA-A blocks")
    write_json(output / "derived-model.json", {
        "parent_model_metadata_sha256": file_hash(model / "model.json"),
        "parent_fingerprint": parent["fingerprint"], "fingerprint": fingerprint,
        "model": asdict(adapter.spec), "precision": settings,
        "conversion": conversion,
        "new_training_rounds": 0})
    # Public synthetic input checks fresh second-step gradients and FP32 memory
    # before any real-data capture. It is not a reconstruction benchmark result.
    smoke_training = replace(base, algorithm="fedavg", batch_size=1, local_steps=2)
    adapter.set_training_spec(smoke_training)
    size = adapter.spec.image_size
    smoke_batch = adapter.batch(torch.full((2, 3, size, size), .5),
                                ["what is shown?"] * 2, ["unknown"] * 2)
    torch.cuda.reset_peak_memory_stats()
    with precision_context(precision):
        smoke_update, smoke_trace = trace_local_update(adapter, smoke_batch, smoke_training, names)
    if precision == "mixed" and any(t["selected_projection_output_dtype"] != "torch.bfloat16"
                                    for t in smoke_trace):
        raise ValueError("Mixed smoke did not use bf16 projection outputs")
    write_json(output / "public-smoke.json", {
        "status": "passed", "input": "two public constant images and public synthetic QA",
        "trace": smoke_trace, "precision": settings,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "benchmark_result": False})
    del smoke_batch, smoke_trace, smoke_update
    torch.cuda.empty_cache()
    jobs = []
    for condition, algorithm, batch_size, steps in conditions:
        training = replace(base, algorithm=algorithm, batch_size=batch_size, local_steps=steps)
        adapter.set_training_spec(training)
        for start in range(0, 30, 2):
            job_id = f"{condition}-{start // 2:02d}"
            print("capture", job_id, flush=True)
            batch = load_batch(rows[start:start + 2], adapter)
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            with precision_context(precision):
                update, trace = trace_local_update(adapter, batch, training, names)
            if any(p._version != versions[name] for name, p in adapter.named_parameters()):
                raise ValueError("Functional local training mutated the shared initial model")
            observed = captured_observation(adapter, batch, training, update, fingerprint)
            root = output / "runs" / job_id
            save_pair(adapter, batch, rows[start:start + 2], observed, root, model, {"name": "none"})
            for image in ("gt", "random"):
                write_json(root / "captures" / image / "public/model_ref.json", {
                    "path": str(model.resolve()), "derivation": conversion,
                    "derived_model_sha256": file_hash(output / "derived-model.json")})
            write_json(root / "local-trace.json", {
                "trace": trace, "steps": steps, "batch": batch_size, "unique_samples": 2,
                "fresh_forward_backward_per_step": True, "parameter_dtype": "float32",
                "gradient_dtype": "float32", "upload_dtype": "float32", "precision": settings})
            if condition.endswith("-one-step"):
                reference = output / "runs" / f"{precision}-gradient-{start // 2:02d}" / "captures/gt/public"
                gradients = read_tensors(reference / "update.safetensors")
                error = norm = 0.
                for name in names:
                    ideal = -training.lr * gradients[name].to(update[name].device).double()
                    error += (update[name].double() - ideal).square().sum().item()
                    norm += ideal.square().sum().item()
                relative = (error / max(norm, 1e-30)) ** .5
                write_json(root / "one-step-audit.json", {
                    "selected_A_relative_l2_delta_vs_gradient": relative,
                    "scope": "six first-layer LoRA-A blocks, freshly recomputed gradient",
                    "precision": settings})
                if relative > .001:
                    raise ValueError("FP32 one-step control differs materially from the gradient")
            write_json(root / "capture-runtime.json", {
                "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            write_json(root / "victim_diagnostics.json", {"relative_l2_change": 0.})
            jobs.append({"id": job_id, "condition": condition, "n": 2, "batch_size": batch_size,
                         "local_steps": steps, "algorithm": algorithm,
                         "length_bins": [r["length_bin"] for r in selected[start:start + 2]]})
            write_json(output / "progress-capture.json", {
                "status": "running", "completed": len(jobs), "total": 45, "job": job_id})
    if adapter.fingerprint() != fingerprint:
        raise ValueError("Initial model state changed across captures")
    write_json(output / "plan.json", {
        "schema_version": 1, "source_sha256": source_fingerprint(), "cohort_n": 30,
        "attack_count": 90, "jobs": jobs, "model": str(model.resolve()),
        "model_fingerprint": fingerprint, "parent_model_metadata_sha256": file_hash(model / "model.json"),
        "derived_model_sha256": file_hash(output / "derived-model.json"), "precision": settings,
        "precision_mode": precision,
        "frozen_sha256": file_hash(frozen_path), "initial_model_immutable": True,
        "pairing": "original frozen order, adjacent two samples from same client",
        "note": conversion + "; " + settings["forward_backward"] + ". "
                "B2/S1 vs B1/S2 also changes minibatch loss weighting; not a pure step-count isolation."})
    write_json(output / "progress-capture.json", {"status": "completed", "completed": 45, "total": 45})


def verify_plan(output):
    plan = read_json(output / "plan.json")
    if source_fingerprint() != plan["source_sha256"]:
        raise ValueError("Frozen source changed")
    if file_hash(Path(plan["model"]) / "model.json") != plan["parent_model_metadata_sha256"]:
        raise ValueError("Parent snapshot changed")
    if file_hash(output / "derived-model.json") != plan["derived_model_sha256"]:
        raise ValueError("FP32 derivation metadata changed")
    return plan


def attack_all(output, image):
    plan = verify_plan(output)
    mode = plan.get("precision_mode", "fp32")
    precision_settings(mode)
    adapter = (restore_mixed if mode == "mixed" else restore_fp32)(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("FP32 attacker model differs from capture model")
    versions = {name: p._version for name, p in adapter.named_parameters()}
    # With FP32 parameters, release unused cache between uploads so NVML remains
    # meaningful; no batch size, beam, precision or score is changed.
    for index, job in enumerate(plan["jobs"]):
        root = output / "runs" / job["id"]
        public = root / "captures" / image / "public"
        directory = root / "attacks" / image
        if directory.exists():
            raise FileExistsError("No implicit attack retry or resume")
        observed = load_observation(public, adapter.spec.device)
        expected_dtype = "bfloat16" if mode == "mixed" else "float32"
        if observed.model.dtype != expected_dtype or any(v.dtype != torch.float32 for v in observed.tensors.values()):
            raise ValueError("Expected FP32 public model and upload")
        config = Config(model=observed.model, training=observed.training,
                        attack=pilot_attack(image, 2))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        write_json(output / f"progress-{image}.json", {
            "status": "running", "phase": "search", "completed": index, "total": 45, "job": job["id"]})
        print("attack", image, job["id"], flush=True)
        started = time.monotonic()
        with precision_context(mode):
            status, _ = attack_public(adapter, config, public, directory, plan["model_fingerprint"],
                                      upload_metadata={"defense": {"name": "none"}})
        if status not in {"completed", "budget_exhausted", "no_signal", "no_candidates"}:
            raise ValueError("Unsupported FP32 multi-step condition: " + status)
        result = read_json(directory / "result.json")
        result["condition"].update(full_fp32_forward_backward=mode == "fp32", tf32=False,
                                   precision_mode=mode, trainable_dtype="float32",
                                   optimizer_dtype="float32", upload_dtype="float32",
                                   autocast_bf16=mode == "mixed", autocast_cache_enabled=False)
        write_json(directory / "result.json", result)
        write_json(directory / "runtime.json", {
            "seconds": time.monotonic() - started, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print("finished", image, job["id"], status, flush=True)
    if any(p._version != versions[name] for name, p in adapter.named_parameters()):
        raise ValueError("Text attack mutated the initial FP32 model")
    write_json(output / f"progress-{image}.json", {"status": "completed", "completed": 45, "total": 45})


def evaluate_all(output):
    verify_plan(output)
    return evaluate_pilot(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("capture", "attack", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--image", choices=("gt", "random"))
    parser.add_argument("--precision", choices=("fp32", "mixed"), default="fp32")
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}-{args.image or 'all'}.json"
    write_json(state, {"status": "running", "phase": args.phase})
    try:
        if args.phase == "capture":
            if args.model is None or args.frozen is None:
                raise ValueError("Model and frozen cohort are required")
            capture_all(args.output, args.model, args.frozen, args.precision)
        elif args.phase == "attack":
            if args.image is None:
                raise ValueError("Image context is required")
            attack_all(args.output, args.image)
        else:
            evaluate_all(args.output)
        write_json(state, {"status": "completed", "phase": args.phase})
    except BaseException:
        write_json(state, {"status": "failed", "phase": args.phase, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
