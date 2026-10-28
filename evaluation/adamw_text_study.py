"""Small AdamW text-recovery diagnostic with native-optimizer replay selection.

Delta row spaces are proposal heuristics, never described as gradient subspaces.
All completed reconstructions precede reference scoring. No new training rounds.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import Config
from core.data import load_batch, read_manifest
from core.fl import load_observation, restore_model, simulate_update
from evaluation.dager_rank_study import attack_public
from evaluation.federated_text_pilot import evaluate_pilot, pilot_attack, save_pair, select_pilot, validate_upload_groups
from evaluation.fp32_multistep_study import captured_observation, precision_context, precision_settings
from evaluation.text_benchmark import capture_training


CONDITIONS = (
    ("adamw-gradient-control", "mixed", "fedsgd", "sgd", 2, 1),
    ("adamw-fp32-s1", "mixed", "fedavg", "adamw", 2, 1),
    ("adamw-fp32-s2", "mixed", "fedavg", "adamw", 1, 2),
    ("adamw-bf16-s1", "bf16", "fedavg", "adamw", 2, 1),
    ("adamw-bf16-s2", "bf16", "fedavg", "adamw", 1, 2),
)


def set_precision(adapter, mode):
    dtype = torch.float32 if mode == "mixed" else torch.bfloat16
    for p in adapter.trainable().values():
        p.data = p.data.to(dtype)
    if any(p.dtype != dtype for p in adapter.trainable().values()):
        raise ValueError("Trainable precision mismatch")
    if any(p.dtype != torch.bfloat16 for p in adapter.parameters() if not p.requires_grad):
        raise ValueError("Frozen backbone must stay bf16")
    return dtype


def attack_spec(image, adamw):
    base = pilot_attack(image, 2)
    return replace(base, dager=replace(base.dager, rerank_candidates=16,
                                       adamw_delta_proposals=adamw))


def traced_update(adapter, batch, training, names):
    original = {k: p.detach().clone() for k, p in adapter.trainable().items() if k in names}
    trace = []

    def before(module, args):
        current = dict(module.named_parameters())
        norm = sum((current[k].detach().double() - v.double()).square().sum() for k, v in original.items())
        trace.append({"forward_index": len(trace), "selected_A_distance_from_initial": float(norm.sqrt()),
                      "parameter_dtype": str(current[names[0]].dtype)})

    def after(module, args, result):
        trace[-1]["projection_output_dtype"] = str(result.dtype)

    h1 = adapter.register_forward_pre_hook(before)
    h2 = dict(adapter.named_modules())[names[0].split(".lora_A.")[0]].register_forward_hook(after)
    try:
        # Both modes use bf16 autocast; only master weights and moments differ.
        with precision_context("mixed", adapter.device.type):
            update = simulate_update(adapter, batch, training, False)
    finally:
        h1.remove()
        h2.remove()
    if len(trace) != training.local_steps or trace[0]["selected_A_distance_from_initial"] != 0:
        raise ValueError("Wrong forward count or initial parameters")
    if training.local_steps > 1 and trace[1]["selected_A_distance_from_initial"] <= 0:
        raise ValueError("Second step did not use updated parameters")
    if any(t["projection_output_dtype"] != "torch.bfloat16" for t in trace):
        raise ValueError("Expected bf16 forward outputs")
    dtype = next(iter(adapter.trainable().values())).dtype
    if any(v.dtype != dtype or not torch.isfinite(v).all() for v in update.values()):
        raise ValueError("Upload dtype or finiteness mismatch")
    return {k: v.detach() for k, v in update.items()}, trace


def native_smoke(adapter, training, names):
    """Public synthetic two-step GPU check against actual torch.optim.AdamW."""
    adapter.set_training_spec(training)
    size = adapter.spec.image_size
    batch = adapter.batch(torch.full((2, 3, size, size), .5),
                          ["what is shown?"] * 2, ["unknown"] * 2)
    expected, trace = traced_update(adapter, batch, training, names)
    original = {k: p.detach().clone() for k, p in adapter.trainable().items()}
    optimizer = torch.optim.AdamW(adapter.trainable().values(), lr=training.lr,
                                 betas=(training.adam_beta1, training.adam_beta2),
                                 eps=training.adam_epsilon, weight_decay=0., foreach=False, fused=False)
    try:
        for index in range(2):
            optimizer.zero_grad(set_to_none=True)
            part = batch.slice(index, index + 1)
            with precision_context("mixed", adapter.device.type):
                loss = adapter(part.images, part.questions, part.targets)
            loss.backward()
            optimizer.step()
        for name, p in adapter.trainable().items():
            torch.testing.assert_close(p.detach() - original[name], expected[name], rtol=0, atol=0)
        states = {str(state[k].dtype) for state in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq")}
        assert states == {str(next(iter(adapter.trainable().values())).dtype)}
    finally:
        with torch.no_grad():
            for name, p in adapter.trainable().items():
                p.copy_(original[name])
                p.grad = None
    return {"status": "passed", "trace": trace, "native_optimizer_exact_match": True,
            "moment_dtypes": sorted(states), "benchmark_result": False,
            "input": "public constant images and public synthetic QA"}


def capture_all(output, model, frozen_path):
    if (output / "runs").exists():
        raise FileExistsError("Use a fresh AdamW diagnostic output")
    precision_settings("mixed")
    frozen = read_json(frozen_path)
    if file_hash(frozen["manifest"]) != frozen["manifest_sha256"]:
        raise ValueError("Frozen data changed")
    selected = select_pilot(frozen)
    lookup = {r["sample_id"]: r for r in read_manifest(frozen["manifest"], task="vqa", split="eval")}
    rows = [lookup[r["sample_id"]] for r in selected]
    validate_upload_groups(rows)
    metadata = read_json(model / "model.json")
    assert metadata["round"] == 10 and metadata["training"]["lora_rank"] == 32
    adapter = restore_model(model, "cuda:0")
    base = capture_training(adapter.training_spec)
    names = [k for k in adapter.trainable() if any(f"layers.{layer}.self_attn.{q}_proj.lora_A." in k
                                                for layer in (0, 1) for q in "qkv")]
    assert len(names) == 6
    jobs, fingerprints = [], {}
    active = None
    for condition, mode, algorithm, optimizer, batch_size, steps in CONDITIONS:
        if mode != active:
            dtype = set_precision(adapter, mode)
            training = replace(base, algorithm="fedavg", local_optimizer="adamw", lr=1e-4,
                               local_steps=2, batch_size=1, weight_decay=0.)
            torch.cuda.reset_peak_memory_stats()
            smoke = native_smoke(adapter, training, names)
            smoke.update(parameter_dtype=str(dtype), peak_allocated_bytes=torch.cuda.max_memory_allocated())
            write_json(output / f"public-smoke-{mode}.json", smoke)
            torch.cuda.empty_cache()
            fingerprints[mode] = adapter.fingerprint()
            active = mode
        training = replace(base, algorithm=algorithm, local_optimizer=optimizer, lr=1e-4 if optimizer == "adamw" else .01,
                           batch_size=batch_size, local_steps=steps, weight_decay=0.,
                           adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8)
        adapter.set_training_spec(training)
        versions = {k: p._version for k, p in adapter.named_parameters()}
        for start in (0, 2):
            job_id = f"{condition}-{start // 2:02d}"
            print("capture", job_id, flush=True)
            torch.cuda.reset_peak_memory_stats()
            begun = time.monotonic()
            batch = load_batch(rows[start:start + 2], adapter)
            update, trace = traced_update(adapter, batch, training, names)
            if any(p._version != versions[k] for k, p in adapter.named_parameters()):
                raise ValueError("Capture mutated initial model")
            observed = captured_observation(adapter, batch, training, update, fingerprints[mode])
            run = output / "runs" / job_id
            save_pair(adapter, batch, rows[start:start + 2], observed, run, model, {"name": "none"})
            for image in ("gt", "random"):
                write_json(run / "captures" / image / "public/model_ref.json", {
                    "path": str(model.resolve()), "precision_mode": mode,
                    "conversion": "verified original bf16 values; promote only trainable LoRA for mixed"})
            write_json(run / "local-trace.json", {"trace": trace, "parameter_dtype": str(dtype),
                "moment_dtype": str(dtype) if optimizer == "adamw" else "none", "fresh_gradient_per_step": True,
                "initial_moments": "zero", "lr": training.lr, "optimizer": optimizer})
            write_json(run / "capture-runtime.json", {"seconds": time.monotonic() - begun,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            write_json(run / "victim_diagnostics.json", {"relative_l2_change": 0.})
            jobs.append({"id": job_id, "condition": condition, "precision_mode": mode,
                "n": 2, "batch_size": batch_size, "local_steps": steps, "optimizer": optimizer,
                "length_bins": [r["length_bin"] for r in selected[start:start + 2]]})
            write_json(output / "progress-capture.json", {"status": "running", "completed": len(jobs), "total": 10})
        if adapter.fingerprint() != fingerprints[mode]:
            raise ValueError("Initial model fingerprint changed")
    write_json(output / "plan.json", {"schema_version": 1, "cohort_n": 4, "attack_count": 20,
        "jobs": jobs, "model": str(model.resolve()), "model_metadata_sha256": file_hash(model / "model.json"),
        "model_fingerprints": fingerprints, "source_sha256": source_fingerprint(),
        "frozen_sha256": file_hash(frozen_path), "initial_model_immutable": True,
        "attack_specs": {"adamw": asdict(attack_spec("random", True)),
                         "control": asdict(attack_spec("random", False))},
        "note": "AdamW delta proposals are heuristic, without gradient row-space premise; actual optimizer replay "
                "selects among 16 candidates. Four preselected QA; not full FedVLMBench retraining. "
                "B2/S1 vs B1/S2 also changes loss weighting. Random replay suffers unknown-image mismatch."})
    write_json(output / "progress-capture.json", {"status": "completed", "completed": 10, "total": 10})


def verify_plan(output):
    plan = read_json(output / "plan.json")
    if source_fingerprint() != plan["source_sha256"]:
        raise ValueError("Frozen source changed")
    if file_hash(Path(plan["model"]) / "model.json") != plan["model_metadata_sha256"]:
        raise ValueError("Public model changed")
    return plan


def evaluate_all(output):
    verify_plan(output)
    return evaluate_pilot(output)


def attack_all(output, image):
    precision_settings("mixed")
    plan = verify_plan(output)
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    active = None
    for index, job in enumerate(plan["jobs"]):
        mode = job["precision_mode"]
        if active != mode:
            set_precision(adapter, mode)
            if adapter.fingerprint() != plan["model_fingerprints"][mode]:
                raise ValueError("Attacker precision/model fingerprint differs")
            active = mode
        run = output / "runs" / job["id"]
        public, directory = run / "captures" / image / "public", run / "attacks" / image
        if directory.exists():
            raise FileExistsError("No implicit retry or resume")
        observed = load_observation(public, adapter.spec.device)
        config = Config(model=observed.model, training=observed.training,
                        attack=attack_spec(image, job["optimizer"] == "adamw"))
        write_json(output / f"progress-{image}.json", {"status": "running", "completed": index,
                   "total": 10, "job": job["id"]})
        print("attack", image, job["id"], flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        begun = time.monotonic()
        with precision_context("mixed"):
            status, _ = attack_public(adapter, config, public, directory, plan["model_fingerprints"][mode],
                                      upload_metadata={"defense": {"name": "none"}})
        if status not in {"completed", "budget_exhausted", "no_signal", "no_candidates"}:
            raise ValueError("Unexpected AdamW attack status: " + status)
        result = read_json(directory / "result.json")
        result["condition"].update(precision_mode=mode, local_optimizer=job["optimizer"],
            trainable_dtype="float32" if mode == "mixed" else "bfloat16", autocast_bf16=True,
            optimizer_state_reset="zero per upload", proposal_type="adamw_delta_heuristic"
            if job["optimizer"] == "adamw" else "gradient_subspace", rerank_candidates=16)
        write_json(directory / "result.json", result)
        write_json(directory / "runtime.json", {"seconds": time.monotonic() - begun,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated()})
        print("finished", image, job["id"], status, flush=True)
    if adapter.fingerprint() != plan["model_fingerprints"][active]:
        raise ValueError("Attack mutated public model")
    write_json(output / f"progress-{image}.json", {"status": "completed", "completed": 10, "total": 10})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("capture", "attack", "evaluate"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--frozen", type=Path)
    p.add_argument("--image", choices=("gt", "random"))
    args = p.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}-{args.image or 'all'}.json"
    write_json(state, {"status": "running"})
    try:
        if args.phase == "capture":
            capture_all(args.output, args.model, args.frozen)
        elif args.phase == "attack":
            if args.image is None:
                raise ValueError("Image condition required")
            attack_all(args.output, args.image)
        else:
            evaluate_all(args.output)
        write_json(state, {"status": "completed"})
    except BaseException:
        write_json(state, {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
