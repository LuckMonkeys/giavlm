"""Recover convex visual tokens from Observations and a disjoint public dictionary."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation, restore_model
from evaluation.public_style_study import check_source
from evaluation.visual_concept_probe import inject_logits, native_hard_loss
from evaluation.visual_token_recovery import block_split, normalized_errors, parameter_grads


def recover(root):
    out = root / "convex"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retries")
    write_json(out / "started.json", {"pid": os.getpid()})
    start = time.monotonic()
    try:
        plan = check_source(root, "convex")
        if not read_json(root / "d0-calibration/gate.json")["passes"]:
            raise ValueError("D0 calibration gate did not pass")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.cuda.mem_get_info()[0] < 60 * 2**30:
            raise RuntimeError("Need 60 GiB free; no silent budget changes")
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        fingerprint = adapter.fingerprint()
        held = torch.tensor(block_split(list(adapter.trainable())), device="cuda:0")
        dictionary = torch.stack([read_tensors(path)["visual"].float() for path in plan["dictionary"]]).to("cuda:0")
        if len(dictionary) != 32 or dictionary.shape[2:] != (576, 4096):
            raise ValueError("Expected fixed 32-image dictionary and native visual token layout")
        hashes, manifests = {}, []
        for query_index, directory in enumerate(plan["observations"]):
            observation = load_observation(directory, "cpu")
            if (observation.model_fingerprint != fingerprint or observation.training.knowledge != "text_known"
                    or observation.public_images is not None or observation.sample_count != 1
                    or observation.training.algorithm != "fedsgd"):
                raise ValueError("Unexpected controlled text_known observation")
            adapter.set_training_spec(observation.training)
            q = torch.tensor(observation.public_question_ids, device="cuda:0")
            a = torch.tensor(observation.public_target_ids, device="cuda:0")
            observed = [observation.tensors[name].to("cuda:0") for name in adapter.trainable()]

            def losses(visual, graph=False, reference=observed):
                gradients = parameter_grads(adapter, visual, q, a, create_graph=graph)
                errors = normalized_errors(gradients, reference)
                return errors[~held].mean(), errors[held].mean()

            # Baseline selection sees only the upload and heldout block errors.
            single = []
            for visual in dictionary:
                fit, val = losses(visual)
                single.append({"fit": float(fit), "heldout": float(val)})
            best_single = min(range(len(single)), key=lambda i: single[i]["heldout"])
            uniform = torch.full((32,), 1 / 32, device="cuda:0")
            mean = torch.einsum("m,mbnd->bnd", uniform, dictionary)
            fit, val = losses(mean)
            mean_errors = {"fit": float(fit), "heldout": float(val)}
            del mean
            runs = []
            for seed in plan["seeds"]:
                gen = torch.Generator(device="cuda:0").manual_seed(seed)
                logits = torch.randn(32, generator=gen, device="cuda:0", requires_grad=True)
                initial = logits.detach().softmax(0)
                optimizer = torch.optim.Adam([logits], lr=plan["learning_rate"])
                best, best_weights, trace = float("inf"), None, []
                for step in range(plan["steps"] + 1):
                    step_start = time.monotonic()
                    visual = torch.einsum("m,mbnd->bnd", logits.softmax(0), dictionary)
                    train, validation = losses(visual, step < plan["steps"])
                    train_value, val_value = float(train.detach()), float(validation.detach())
                    trace.append({"step": step, "fit": train_value, "heldout": val_value})
                    if val_value < best:
                        best = val_value
                        best_weights = logits.detach().softmax(0).clone()
                    if step < plan["steps"]:
                        derivative, = torch.autograd.grad(train, logits)
                        if not torch.isfinite(derivative).all():
                            raise ValueError("Nonfinite coefficient gradient")
                        optimizer.zero_grad(set_to_none=True)
                        logits.grad = derivative
                        optimizer.step()
                    del visual, train, validation
                    trace[-1]["seconds"] = time.monotonic() - step_start
                    trace[-1]["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
                    write_json(out / "progress.json", {"query": query_index, "queries": 12,
                               "seed": seed, "step": step, "steps": plan["steps"],
                               "seconds": time.monotonic() - start})
                    if query_index == 0 and step in [0, 1, 5]:
                        write_json(out / f"timing-step-{step}.json", {"seconds": time.monotonic() - start,
                                   "seed": seed, "fit": train_value, "heldout": val_value,
                                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20})
                    print(f"Convex q={query_index} seed={seed} step={step} fit={train_value:.6g} held={val_value:.6g}", flush=True)
                weights_path = out / "weights" / f"q{query_index:02d}-s{seed}.safetensors"
                write_tensors(weights_path, {"best": best_weights.cpu(), "initial": initial.cpu(),
                                            "final": logits.detach().softmax(0).cpu()})
                hashes[str(weights_path)] = file_hash(weights_path)
                runs.append({"seed": seed, "heldout": best, "weights": str(weights_path), "trace": trace})
                del logits, optimizer, best_weights
                gc.collect()
            selected = min(range(len(runs)), key=lambda i: runs[i]["heldout"])
            near = [i for i, row in enumerate(runs) if row["heldout"] <= runs[selected]["heldout"] * 1.05 + 1e-6]
            chosen = read_tensors(runs[selected]["weights"])["best"].to("cuda:0")
            visual = torch.einsum("m,mbnd->bnd", chosen, dictionary)
            selected_grads = parameter_grads(adapter, visual, q, a)
            all_errors = normalized_errors(selected_grads, observed)
            energy_candidate = sum(float(g.detach().float().double().square().sum()) for g in selected_grads)
            energy_observed = sum(float(g.detach().float().double().square().sum()) for g in observed)
            inner = sum(float((x.detach().double() * y.double()).sum())
                        for x, y in zip(selected_grads, observed, strict=True))
            match_stats = {"fit": float(all_errors[~held].mean()), "heldout": float(all_errors[held].mean()),
                           "all_blocks": float(all_errors.mean()),
                           "global_cosine_loss": 1 - inner / max((energy_candidate * energy_observed) ** .5, 1e-60),
                           "norm_ratio": (energy_candidate / max(energy_observed, 1e-60)) ** .5}
            del visual, selected_grads, all_errors, chosen
            row = {"query": query_index, "runs": runs, "selected": selected, "near_optimal": near,
                   "mean_errors": mean_errors, "single_errors": single, "best_single": best_single,
                   "selected_matching": match_stats}
            path = out / "predictions" / f"q{query_index:02d}.json"
            write_json(path, row)
            hashes[str(path)] = file_hash(path)
            manifests.append(row)
            del observed
            torch.cuda.empty_cache()
        # Property question and all attribute readout are deliberately delayed.
        write_json(out / "predictions_complete.json", {"hashes": hashes, "queries": 12,
                   "selection": "Minimum heldout upload-block matching loss, no property or image truth",
                   "seconds": time.monotonic() - start})
        read_q = adapter.encode([plan["readout_question"]], adapter.spec.question_length)
        answers = [adapter.encode([answer], adapter.spec.target_length) for answer in ["yes", "no"]]

        @torch.no_grad()
        def readout(weights):
            visual = torch.einsum("m,mbnd->bnd", weights.to("cuda:0"), dictionary)
            return [float(native_hard_loss(inject_logits(adapter, visual, read_q, y), y, adapter.pad)) for y in answers]

        outputs = []
        for row in manifests:
            scores = {"mean": readout(uniform),
                      "best_single": readout(torch.nn.functional.one_hot(torch.tensor(row["best_single"]), 32).float())}
            optimized, initial_scores = [], []
            for run in row["runs"]:
                weights = read_tensors(run["weights"])
                optimized.append(readout(weights["best"]))
                initial_scores.append(readout(weights["initial"]))
            scores["optimized"] = optimized[row["selected"]]
            scores["random_initial_mean"] = torch.tensor(initial_scores).mean(0).tolist()
            outputs.append({"query": row["query"], "scores": scores,
                            "initial_scores": initial_scores, "optimized_scores": optimized,
                            "near_optimal": row["near_optimal"], "selected": row["selected"]})
        path = out / "readout_predictions.json"
        write_json(path, outputs)
        if adapter.fingerprint() != fingerprint:
            raise ValueError("Model mutated")
        write_json(out / "complete.json", {"readout_sha256": file_hash(path),
                   "predictions_sha256": file_hash(out / "predictions_complete.json"),
                   "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "optimization_steps": 12 * len(plan["seeds"]) * plan["steps"],
                   "readout_uses_attribute_for_optimization": False})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    recover(args.root)


if __name__ == "__main__":
    main()
