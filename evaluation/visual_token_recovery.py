"""Public D0 double-backward gate; never reads private targets or labels."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import time
import traceback

import torch
from PIL import Image

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import restore_model
from evaluation.visual_concept_probe import inject_logits, interface_check, native_hard_loss


def block_split(names):
    """Hold out one in five layers, stratified by module and LoRA factor."""
    import re
    held = []
    for name in names:
        match = re.search(r"layers\.(\d+)\.(.*)", name)
        if match is None:
            raise ValueError(f"Unexpected uploaded parameter: {name}")
        offset = int(hashlib.sha256(match[2].encode()).hexdigest()[:8], 16) % 5
        held.append(int(match[1]) % 5 == offset)
    if not any(held) or all(held):
        raise ValueError("Empty fit or heldout upload blocks")
    return held


def normalized_errors(candidate, observed, floor_relative=1e-6):
    energy = torch.stack([g.detach().float().square().sum() for g in observed])
    floor = energy.mean() * floor_relative
    denominator = energy + floor.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.stack([(a.float() - b.float()).square().sum()
                        for a, b in zip(candidate, observed, strict=True)]) / denominator


def toy_direction_check():
    """FP32 exact-small-network directional finite difference of double backward."""
    gen = torch.Generator().manual_seed(20261028)
    weight = torch.randn(7, 5, generator=gen, requires_grad=True)
    x = torch.randn(3, 5, generator=gen, requires_grad=True)
    reference = torch.randn(7, 5, generator=gen)
    direction = torch.randn(3, 5, generator=gen)
    direction /= direction.norm()

    def objective(value):
        loss = (value @ weight.T).tanh().square().mean()
        grad, = torch.autograd.grad(loss, weight, create_graph=True)
        return normalized_errors([grad], [reference]).mean()

    value = objective(x)
    derivative, = torch.autograd.grad(value, x)
    analytic = float((derivative * direction).sum())
    step = .01
    difference = float((objective(x + step * direction) - objective(x - step * direction)) / (2 * step))
    error = abs(analytic - difference) / max(abs(analytic), abs(difference), 1e-6)
    if error > .02:
        raise ValueError(f"Small-network double-backward check failed: {error}")
    return {"analytic": analytic, "finite_difference": difference,
            "relative_error": error, "step": step, "tolerance": .02, "dtype": "float32"}


def parameter_grads(adapter, visual, question, target, create_graph=False):
    loss = native_hard_loss(inject_logits(adapter, visual.to(adapter.dtype), question, target),
                            target, adapter.pad)
    return torch.autograd.grad(loss, tuple(adapter.trainable().values()), create_graph=create_graph)


def d0(root, stage="d0"):
    if (root / f"{stage}/started.json").exists():
        raise FileExistsError("Do not retry or overwrite an attempted gate")
    plan = read_json(root / "plan.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly the selected GPU")
    write_json(root / f"{stage}/started.json", {"pid": os.getpid(), "host": os.uname().nodename})
    start = time.monotonic()
    rates = [.005] if stage == "d0" else read_json(root / "d0-calibration/plan.json")["rates"]
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError(f"Frozen input changed: {path}")
        source_manifest = root / stage / "runtime_source.json"
        if not source_manifest.exists():
            source_manifest = root / "runtime_source.json"
        for path, sha in read_json(source_manifest)["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError(f"Frozen source changed: {path}")
        if torch.cuda.mem_get_info()[0] < 60 * 2**30:
            raise RuntimeError("D0 requires 60 GiB free; no silent budget or dtype change")
        toy = toy_direction_check()
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        initial = adapter.fingerprint()
        names = list(adapter.trainable())
        held = torch.tensor(block_split(names), device="cuda:0")
        question = adapter.encode([plan["training_question"]], adapter.spec.question_length)
        target = adapter.encode([plan["training_answer"]], adapter.spec.target_length)
        rows = []
        for case_index, case in enumerate(plan["engineering_cases"]):
            pixels = read_tensors(Path(case["bank"]) / "public/bank.safetensors")[case["key"]]
            image = adapter.prepare_image(Image.fromarray(pixels.numpy())).unsqueeze(0).to(adapter.device, adapter.dtype)
            with torch.no_grad():
                truth = adapter.visual_embeddings(image)[0].detach()
            exact = interface_check(adapter, image, truth, question, target)
            observed = [g.detach() for g in parameter_grads(adapter, truth, question, target)]
            generator = torch.Generator(device="cuda:0").manual_seed(20261028 + case_index)
            # This is a labeled public interface check, NOT a recovery initialization.
            scale = float(truth.float().std())
            visual = (truth.float() + .05 * scale * torch.randn(truth.shape, generator=generator,
                      device="cuda:0", dtype=torch.float32)).detach().requires_grad_(True)
            grads = parameter_grads(adapter, visual, question, target, create_graph=True)
            errors = normalized_errors(grads, observed)
            objective = errors[~held].mean()
            before = {"fit": float(objective.detach()), "heldout": float(errors[held].mean().detach())}
            derivative, = torch.autograd.grad(objective, visual)
            if not torch.isfinite(derivative).all() or float(derivative.norm()) <= 0:
                raise ValueError("No finite nonzero second-order visual gradient")
            derivative_norm = float(derivative.norm())
            direction = derivative / derivative.square().mean().sqrt().clamp_min(1e-30)
            del grads, errors, objective, derivative
            gc.collect()
            trials = []
            for rate in rates:
                with torch.no_grad():
                    updated = visual - rate * scale * direction
                after_errors = normalized_errors(parameter_grads(adapter, updated, question, target), observed)
                after = {"fit": float(after_errors[~held].mean()), "heldout": float(after_errors[held].mean())}
                trials.append({"rate": rate, **after})
            if all(before["fit"] == trial["fit"] for trial in trials):
                raise ValueError("An actual FP32 visual update did not change the matching objective")
            item = {"index": case_index, "injection": exact, "before": before, "after": after,
                    "second_order_norm": derivative_norm, "trials": trials,
                    "fit_decreased": min(t["fit"] for t in trials) < before["fit"],
                    "visual_shape": list(truth.shape), "master_dtype": "float32"}
            if stage == "d0-calibration":
                bank = read_tensors(Path(plan["engineering_cases"][0]["bank"]) / "public/bank.safetensors")
                dictionary = []
                for key in ["candidate-04", "candidate-16"]:
                    public_image = adapter.prepare_image(Image.fromarray(bank[key].numpy())).unsqueeze(0).to(adapter.device, adapter.dtype)
                    with torch.no_grad():
                        dictionary.append(adapter.visual_embeddings(public_image)[0].detach().float())
                dictionary = torch.stack(dictionary)
                logits = torch.zeros(2, device="cuda:0", requires_grad=True)
                mix = torch.einsum("m,mbnd->bnd", logits.softmax(0), dictionary)
                candidate = parameter_grads(adapter, mix, question, target, create_graph=True)
                errs = normalized_errors(candidate, observed)
                previous = float(errs[~held].mean().detach())
                derivative_alpha, = torch.autograd.grad(errs[~held].mean(), logits)
                optimizer = torch.optim.Adam([logits], lr=.05)
                logits.grad = derivative_alpha
                optimizer.step()
                with torch.no_grad():
                    next_mix = torch.einsum("m,mbnd->bnd", logits.softmax(0), dictionary)
                next_err = normalized_errors(parameter_grads(adapter, next_mix, question, target), observed)
                item["convex_engineering"] = {"before_fit": previous,
                    "after_fit": float(next_err[~held].mean()), "after_heldout": float(next_err[held].mean()),
                    "coefficient_gradient_norm": float(derivative_alpha.norm()), "adam_learning_rate": .05,
                    "dictionary_size": 2, "dictionary_uses_query": False}
                del candidate, errs, next_err, dictionary, mix, next_mix, logits, optimizer, derivative_alpha
            rows.append(item)
            write_json(root / stage / f"case-{case_index}.json", item)
            write_tensors(root / stage / f"case-{case_index}.safetensors", {"updated_visual": updated.cpu()})
            write_json(root / "progress.json", {"phase": stage, "completed": len(rows), "total": 4,
                       "seconds": time.monotonic() - start})
            if case_index == 0:
                write_json(root / stage / "first_checkpoint.json", item)
                if stage == "d0":
                    write_json(root / "first_checkpoint.json", item)
            print(f"D0 {case_index}: {before} -> {after}; second-order norm {derivative_norm}", flush=True)
            del observed, visual, updated, truth, image, after_errors, direction
            gc.collect()
            torch.cuda.empty_cache()
        if adapter.fingerprint() != initial:
            raise ValueError("Public model mutated")
        write_json(root / f"{stage}/complete.json", {"status": "complete", "toy": toy, "cases": rows,
                   "all_fit_decreased": all(r["fit_decreased"] for r in rows),
                   "backend": adapter.backend.language_model.config._attn_implementation,
                   "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "fit_blocks": int((~held).sum()), "heldout_blocks": int(held.sum()),
                   "model_fingerprint": initial, "parameters_unchanged": True})
    except BaseException:
        write_json(root / f"{stage}/failed.json", {"traceback": traceback.format_exc(),
                   "seconds": time.monotonic() - start})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["d0", "d0-calibration"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    d0(args.root, args.stage)


if __name__ == "__main__":
    main()
