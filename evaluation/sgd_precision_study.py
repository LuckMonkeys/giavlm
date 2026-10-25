"""One-step SGD precision controls from identical, already observed gradients.

This is a fixed-gradient diagnostic, not FP32 retraining or multi-step FedAvg.
Only the evaluate phase reads the original private reference artifacts.
"""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import shutil
import time
import traceback

import torch

from attacks.analytic.dager_adapted import DAGERSearch, gradient_groups
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json
from core.config import Config
from core.fl import load_observation, restore_model, save_observation
from evaluation.dager_rank_study import attack_public
from evaluation.federated_text_pilot import evaluate_pilot, pilot_attack, update_norm


VARIANTS = ("ideal-fp64", "sgd-bf16", "sgd-fp32")


def fixed_gradient_update(weights, gradients, variant, lr):
    """The actual torch optimizer is the reference for each parameter dtype."""
    if variant == "ideal-fp64":
        return {name: -lr * gradient.double() for name, gradient in gradients.items()}
    if variant not in VARIANTS:
        raise ValueError("Unknown precision control")
    dtype = torch.bfloat16 if variant == "sgd-bf16" else torch.float32
    parameters = {name: torch.nn.Parameter(weights[name].to(dtype).clone()) for name in gradients}
    for name, parameter in parameters.items():
        parameter.grad = gradients[name].to(dtype).clone()
    optimizer = torch.optim.SGD(parameters.values(), lr=lr, foreach=False, fused=False)
    optimizer.step()
    return {name: parameter.detach() - weights[name].to(dtype)
            for name, parameter in parameters.items()}


def numeric_audit(gradients, update, names, lr):
    ideal = {name: -lr * gradients[name].double() for name in names}
    difference = {name: update[name].double() - ideal[name] for name in names}
    count = sum(gradients[name].numel() for name in names)
    return {"selected_A_relative_l2_vs_ideal": update_norm(difference) / update_norm(ideal),
            "selected_A_gradient_nonzero_fraction": sum(
                gradients[name].count_nonzero().item() for name in names) / count,
            "selected_A_delta_nonzero_fraction": sum(
                update[name].count_nonzero().item() for name in names) / count}


def prepare(output, source):
    if (output / "runs").exists() or (output / "plan.json").exists():
        raise FileExistsError("Use a fresh precision-control directory")
    previous = read_json(source / "plan.json")
    source_jobs = [job for job in previous["jobs"] if job["condition"] == "fedsgd-b2"]
    if len(source_jobs) != 15 or previous["cohort_n"] != 30:
        raise ValueError("Expected all 15 original SLAKE paired gradient uploads")
    model = Path(previous["model"])
    if file_hash(model / "model.json") != previous["model_metadata_sha256"]:
        raise ValueError("Original model metadata changed")
    weights = read_tensors(model / "model.safetensors")
    jobs = []
    for index, original in enumerate(source_jobs):
        public = source / "runs" / original["id"] / "captures/gt/public"
        obs = load_observation(public)
        legacy = load_observation(source / "runs" / f"fedavg-b2-s1-{index:02d}" / "captures/gt/public")
        assert replace(obs.training, algorithm="fedavg") == legacy.training
        assert obs.model_fingerprint == legacy.model_fingerprint == previous["model_fingerprint"]
        torch.testing.assert_close(obs.public_images, legacy.public_images, rtol=0, atol=0)
        names = sum(read_json(source / "runs" / original["id"] / "attacks/gt/result.json")[
            "provenance"]["gradient_names"], [])
        for variant in VARIANTS:
            job_id = f"{variant}-{index:02d}"
            print("prepare", job_id, flush=True)
            update = fixed_gradient_update(weights, obs.tensors, variant, obs.training.lr)
            root = output / "runs" / job_id
            diagnostics = numeric_audit(obs.tensors, update, names, obs.training.lr)
            diagnostics.update(variant=variant, source_gradient_sha256=file_hash(public / "update.safetensors"),
                               lr=obs.training.lr, update_dtype=str(next(iter(update.values())).dtype),
                               fixed_gradient=True, original_pair_index=index)
            write_json(root / "numeric-audit.json", diagnostics)
            for image in ("gt", "random"):
                training = replace(obs.training, algorithm="fedavg",
                                   knowledge="image_known" if image == "gt" else "private")
                current = replace(obs, tensors=update, training=training,
                                  public_images=obs.public_images if image == "gt" else None)
                directory = root / "captures" / image / "public"
                save_observation(directory, current)
                write_json(directory / "upload.json", {"defense": {"name": "none"},
                           "precision_variant": variant, "fixed_gradient": True})
            left, right = [read_json(root / "captures" / im / "public/observation.json")
                           for im in ("gt", "random")]
            if left["update_sha256"] != right["update_sha256"]:
                raise ValueError("Paired upload values differ")
            # The generic evaluator uses this field for defense strength only.
            write_json(root / "victim_diagnostics.json", {"relative_l2_change": 0.})
            jobs.append({"id": job_id, "condition": variant, "n": 2,
                         "length_bins": original["length_bins"], "source_job": original["id"],
                         "pair_index": index})
        write_json(output / "progress-prepare.json", {"status": "running", "completed": index + 1,
                                                     "total": len(source_jobs)})
    write_json(output / "plan.json", {
        "schema_version": 1, "cohort_n": 30, "attack_count": 90, "jobs": jobs,
        "source_sha256": source_fingerprint(), "reference_source": str(source.resolve()),
        "reference_plan_sha256": file_hash(source / "plan.json"),
        "model": str(model), "model_metadata_sha256": file_hash(model / "model.json"),
        "model_fingerprint": previous["model_fingerprint"], "variants": list(VARIANTS),
        "method_note": "Same observed bf16 gradient; one-step public-weight arithmetic only. "
                       "FP32 master arithmetic here does not change forward/backward precision.",
        "original_comparison_conditions": ["fedsgd-b2", "fedavg-b2-s1"],
        "attack_specs": {image: asdict(pilot_attack(image, 2)) for image in ("gt", "random")}})
    write_json(output / "progress-prepare.json", {"status": "completed", "completed": 15, "total": 15})


def verify_cuda_sgd(adapter, observation, source):
    names = sum(gradient_groups(adapter, observation, "qkv"), [])
    weights = dict(adapter.named_parameters())
    gradients = read_tensors(source / "runs/fedsgd-b2-00/captures/gt/public/update.safetensors")
    for dtype in (torch.bfloat16, torch.float32):
        for name in names:
            gradient = gradients[name].to(adapter.device, dtype)
            original = weights[name].detach().to(dtype)
            parameter = torch.nn.Parameter(original.clone())
            parameter.grad = gradient.clone()
            optimizer = torch.optim.SGD([parameter], lr=observation.training.lr, foreach=False, fused=False)
            optimizer.step()
            actual = torch.add(original, gradient, alpha=-observation.training.lr)
            torch.testing.assert_close(parameter, actual, rtol=0, atol=0)


def attack(output, image):
    plan = read_json(output / "plan.json")
    if source_fingerprint() != plan["source_sha256"]:
        raise ValueError("Frozen source differs from upload preparation")
    if file_hash(Path(plan["model"]) / "model.json") != plan["model_metadata_sha256"]:
        raise ValueError("Model metadata changed")
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    versions = {name: p._version for name, p in adapter.named_parameters()}
    # Compute every token filter first; no private labels are read here.
    for index, job in enumerate(plan["jobs"]):
        root = output / "runs" / job["id"]
        directory = root / "attacks" / image
        if directory.exists():
            raise FileExistsError("No implicit retries or resume")
        obs = load_observation(root / "captures" / image / "public", adapter.spec.device)
        if index == 0:
            verify_cuda_sgd(adapter, obs, Path(plan["reference_source"]))
            write_json(output / f"cuda-sgd-{image}.json", {"status": "passed", "dtypes": ["bf16", "fp32"],
                       "scope": "torch.add exactly matches actual torch.optim.SGD on selected CUDA blocks"})
        adapter.set_training_spec(obs.training)
        spec = pilot_attack(image, job["n"])
        write_json(output / f"progress-{image}.json", {
            "phase": "filter", "status": "running", "completed": index, "total": len(plan["jobs"]),
            "job": job["id"]})
        print("filter", image, job["id"], flush=True)
        search = DAGERSearch(adapter, obs, spec, directory)
        search.prepare()
        search.filter_tokens()
        write_json(directory / "filter-complete.json", {
            "status": "completed", "precision_variant": job["condition"],
            "stage": search.state["stage"], "saved_seconds": search.elapsed(),
            "source_sha256": source_fingerprint()})
    write_json(output / f"filters-complete-{image}.json", {"status": "completed", "count": len(plan["jobs"])})
    for index, job in enumerate(plan["jobs"]):
        root = output / "runs" / job["id"]
        public = root / "captures" / image / "public"
        directory = root / "attacks" / image
        obs = load_observation(public)
        config = Config(model=obs.model, training=obs.training, attack=pilot_attack(image, job["n"]))
        write_json(output / f"progress-{image}.json", {
            "phase": "search", "status": "running", "completed": index, "total": len(plan["jobs"]),
            "job": job["id"]})
        print("search", image, job["id"], flush=True)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        status, _ = attack_public(adapter, config, public, directory, plan["model_fingerprint"],
                                  upload_metadata=read_json(public / "upload.json"), resume=True)
        if status not in {"completed", "budget_exhausted", "no_candidates", "no_signal"}:
            raise ValueError("Unsupported precision condition: " + status)
        result = read_json(directory / "result.json")
        result["condition"].update(update_precision_variant=job["condition"], fixed_gradient=True)
        write_json(directory / "result.json", result)
        write_json(directory / "runtime.json", {"seconds": time.monotonic() - started,
                   "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                   "filter_was_checkpointed": True})
        print("finished", image, job["id"], status, flush=True)
    if any(p._version != versions[name] for name, p in adapter.named_parameters()):
        raise ValueError("Precision diagnostic mutated the public model")
    write_json(output / f"progress-{image}.json", {
        "status": "completed", "phase": "search", "completed": len(plan["jobs"]), "total": len(plan["jobs"])})


def evaluate(output):
    plan = read_json(output / "plan.json")
    # Do not touch private reference artifacts before all reconstructions commit.
    for job in plan["jobs"]:
        for image in ("gt", "random"):
            read_json(output / "runs" / job["id"] / "attacks" / image / "result.json")
    source = Path(plan["reference_source"])
    if file_hash(source / "plan.json") != plan["reference_plan_sha256"]:
        raise ValueError("Reference cohort plan changed")
    for job in plan["jobs"]:
        for image in ("gt", "random"):
            capture = output / "runs" / job["id"] / "captures" / image
            truth = source / "runs" / job["source_job"] / "captures" / image / "private"
            reference = read_json(truth / "truth.json")
            observed = read_json(capture / "public/observation.json")
            reference.update(observation_id=observed["observation_id"], training=observed["training"])
            private = capture / "private"
            if private.exists():
                raise FileExistsError("No implicit evaluation retry")
            private.mkdir()
            for name in ("images.safetensors", "text_tokens.safetensors"):
                shutil.copyfile(truth / name, private / name)
            write_json(private / "truth.json", reference)
    summary = evaluate_pilot(output)
    summary["note"] = plan["method_note"]
    write_json(output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "attack", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--image", choices=("gt", "random"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}-{args.image or 'all'}.json"
    write_json(state, {"status": "running", "phase": args.phase})
    try:
        if args.phase == "prepare":
            if args.source is None:
                raise ValueError("Source cohort required")
            prepare(args.output, args.source)
        elif args.phase == "attack":
            if args.image is None:
                raise ValueError("Image context required")
            attack(args.output, args.image)
        else:
            evaluate(args.output)
        write_json(state, {"status": "completed", "phase": args.phase})
    except BaseException:
        write_json(state, {"status": "failed", "phase": args.phase, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
