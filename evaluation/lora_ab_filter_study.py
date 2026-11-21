"""A/B coupling under fixed defended FedSGD uploads: score, then evaluate."""
import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids, gradient_groups
from attacks.analytic.dager_subspace import residual, row_basis
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.config import DAGEROptions
from core.fl import load_observation, restore_model
from evaluation.lora_ab_math import METHODS, NULL_QUANTILE, NULL_SAMPLES, module_blocks, score_blocks
from evaluation.soft_residual_filter_study import prepare as prepare_uploads, selected_ids, verify_plan


CONDITIONS = ("fedsgd-b1", "noise")
MODULES = ("q", "k", "v")


def prepare(output, source):
    prepare_uploads(output, source)
    plan = read_json(output / "plan.json")
    del plan["alphas"]
    original = read_json(source / "plan.json")
    bins = {int(j["id"].rsplit("-", 1)[1]): j["length_bins"][0]
            for j in original["jobs"] if j["condition"] == "fedsgd-b1"}
    plan.update(conditions=list(CONDITIONS), methods=list(METHODS), score_groups=120,
                score_conditions=480, length_bins_by_index=bins,
                null_samples=NULL_SAMPLES, null_quantile=NULL_QUANTILE,
                null_seed_rule="1729 + qkv module index; same draws for every upload and image",
                noise_proxy="0.01 * observed tensor maximum; no paired clean upload in scoring",
                joint_weights="observed A/B maxima squared; common ratio cancels",
                source_boundary="All scores commit before token references or paired clean-upload diagnosis",
                common_ambiguity="same original hard-residual ambiguous IDs for all methods")
    plan["sources"] = {k: v for k, v in plan["sources"].items()
                       if k.split(":")[0].rsplit("-", 1)[0] in CONDITIONS}
    write_json(output / "plan.json", plan)


@torch.no_grad()
def score_all(output, image):
    plan = verify_plan(output)
    if plan["methods"] != list(METHODS) or plan["null_samples"] != NULL_SAMPLES:
        raise ValueError("Prespecified method matrix changed")
    source = Path(plan["source"])
    options = DAGEROptions(**plan["options"])
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Model differs from original uploads")
    versions = {n: p._version for n, p in adapter.named_parameters()}
    parameters = dict(adapter.named_parameters())
    features = adapter.backend.language_model.model.layers[0].input_layernorm(
        adapter.embedding().weight).detach().double()
    forbidden = forbidden_token_ids(adapter)
    completed = 0
    for index in range(30):
        observations = {}
        for condition in CONDITIONS:
            public = source / "runs" / f"{condition}-{index:02d}" / "captures" / image / "public"
            expected = plan["sources"][f"{condition}-{index:02d}:{image}"]
            if file_hash(public / "observation.json") != expected["metadata_sha256"]:
                raise ValueError("Observation metadata changed")
            obs = load_observation(public, "cuda:0")
            if obs.model_fingerprint != plan["model_fingerprint"]:
                raise ValueError("Observation fingerprint mismatch")
            observations[condition] = obs
        obs = observations["fedsgd-b1"]
        adapter.set_training_spec(obs.training)
        if image == "gt":
            images = obs.public_images
            if any(plan["sources"][f"{c}-{index:02d}:{image}"]["public_images_sha256"] !=
                   plan["sources"][f"fedsgd-b1-{index:02d}:{image}"]["public_images_sha256"]
                   for c in CONDITIONS):
                raise ValueError("Image context differs across conditions")
        else:
            if any(o.public_images is not None for o in observations.values()):
                raise ValueError("Random scoring received a private image")
            images = torch.rand(1, 3, obs.model.image_size, obs.model.image_size,
                                generator=torch.Generator().manual_seed(plan["seed"]))
        view = LlavaTextView(adapter, replace(obs, public_images=images))
        public_basis, _ = row_basis(view.public_layer0.double(), options.rank_rtol, options.rank_atol)
        projected = torch.empty_like(features)
        for start in range(0, len(features), options.vocab_chunk_size):
            x = features[start:start + options.vocab_chunk_size]
            projected[start:start + len(x)] = residual(x, public_basis)
        flags = projected.norm(dim=-1) <= options.zero_tolerance * features.norm(dim=-1)
        for condition, observation in observations.items():
            directory = output / "scores" / image / f"{condition}-{index:02d}"
            if directory.exists():
                raise FileExistsError("No implicit scoring retry")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            blocks, diagnostics, artifacts = {m: [] for m in METHODS}, {}, {}
            names = gradient_groups(adapter, observation, "qkv")[0]
            length = observation.public_question_lengths[0] + observation.public_target_lengths[0]
            for module_index, (module, name_a) in enumerate(zip(MODULES, names, strict=True)):
                name_b = name_a.replace(".lora_A.", ".lora_B.")
                if name_b not in parameters or name_b not in observation.tensors:
                    raise ValueError("Matching B weight or gradient is missing")
                # Observation tensors remain on CPU even when public images
                # are loaded onto CUDA. Match the explicitly selected device.
                ga, gb = (observation.tensors[n].to(device=public_basis.device, dtype=torch.float64)
                          for n in (name_a, name_b))
                a, b = (parameters[n].detach().double() for n in (name_a, name_b))
                # This function receives only this condition's observation.
                values, diagnostic, tensors = module_blocks(ga, gb, a, b, public_basis, length,
                    .01 if condition == "noise" else 0., seed=1729 + module_index)
                diagnostics[module] = diagnostic
                for method in METHODS:
                    blocks[method].append(values[method])
                artifacts.update({f"{module}_{k}": v.cpu() for k, v in tensors.items()})
            scores, subspaces = {}, {}
            for method in METHODS:
                if method != "baseline" and all(torch.equal(x, y) for x, y in
                        zip(blocks[method], blocks["baseline"], strict=True)):
                    scores[method] = scores["baseline"].clone()
                    subspaces[method] = dict(subspaces["baseline"], identical_matrix_to_baseline=True)
                else:
                    values, subspaces[method] = score_blocks(blocks[method], projected, flags, options)
                    scores[method] = values.cpu()
            historical = read_tensors(source / "runs" / f"{condition}-{index:02d}" /
                                      "attacks" / image / "token_candidates.safetensors")
            delta = float((scores["baseline"] - historical["scores"]).abs().max())
            if (not torch.equal(flags.cpu(), historical["ambiguous"]) or delta > 1e-6 or
                    selected_ids(scores["baseline"], flags.cpu(), forbidden) !=
                    set(historical["selected_ids"].tolist())):
                raise ValueError(f"Original filter differs: max score delta {delta}")
            write_tensors(directory / "scores.safetensors", {**scores, "ambiguous": flags.cpu()})
            write_tensors(directory / "diagnostics.safetensors", artifacts)
            write_json(directory / "complete.json", {
                "status": "completed", "condition": condition, "image": image, "index": index,
                "methods": list(METHODS), "forbidden_ids": sorted(forbidden),
                "baseline_candidates_exact": True, "baseline_max_score_delta": delta,
                "diagnostics": diagnostics, "subspaces": subspaces, "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "score_sha256": file_hash(directory / "scores.safetensors"),
                "diagnostic_sha256": file_hash(directory / "diagnostics.safetensors"),
                "source_sha256": plan["source_sha256"], "model_fingerprint": plan["model_fingerprint"],
                "observation_id": plan["sources"][f"{condition}-{index:02d}:{image}"]["observation_id"]})
            completed += 1
            write_json(output / f"progress-{image}.json", {
                "status": "running", "completed": completed, "total": 60,
                "last_group": f"{condition}-{index:02d}"})
            print("committed", image, condition, index, "baseline delta", delta, flush=True)
        del observations, observation, obs, view, projected, images, blocks
        torch.cuda.empty_cache()
    if any(p._version != versions[n] for n, p in adapter.named_parameters()):
        raise ValueError("Read-only scoring mutated model parameters")
    write_json(output / f"scores-complete-{image}.json", {
        "status": "completed", "groups": 60, "score_conditions": 240,
        "source_sha256": plan["source_sha256"], "initial_model_immutable": True})
    write_json(output / f"progress-{image}.json", {"status": "completed", "completed": 60, "total": 60})


def aggregate(rows, keys, fields):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in buckets.items():
        entry = dict(zip(keys, key, strict=True), n=len(values))
        for field in fields:
            present = [v[field] for v in values if v.get(field) is not None]
            entry[field] = statistics.fmean(present) if present else None
        if "full_coverage" in fields:
            entry["full_coverage_count"] = sum(v["full_coverage"] for v in values)
        result.append(entry)
    return result


def paired_clean_diagnosis(output, plan):
    """Additional clean uploads are evaluation references, never score inputs."""
    rows = []
    for image in ("gt", "random"):
        for index in range(30):
            base = output / "scores" / image
            clean = read_tensors(base / f"fedsgd-b1-{index:02d}" / "diagnostics.safetensors")
            noisy = read_tensors(base / f"noise-{index:02d}" / "diagnostics.safetensors")
            cm = read_json(base / f"fedsgd-b1-{index:02d}" / "complete.json")["diagnostics"]
            nm = read_json(base / f"noise-{index:02d}" / "complete.json")["diagnostics"]
            for module in MODULES:
                reference, observed = clean[f"{module}_raw_residual"], noisy[f"{module}_raw_residual"]
                estimates = {"baseline": observed, "ab_consistency": noisy[f"{module}_joint_residual"],
                    "residual_left": noisy[f"{module}_left_projection"] @ observed,
                    "residual_left_gated": noisy[f"{module}_gate_projection"] @ observed}
                for method, value in estimates.items():
                    projection = noisy.get(f"{module}_" + ("left_projection" if method == "residual_left"
                                           else "gate_projection")) if method.startswith("residual_left") else None
                    rows.append({"image": image, "index": index, "module": module, "method": method,
                        "relative_residual_l2_error": float((value - reference).norm() / reference.norm()),
                        "retained_clean_energy": (float((projection @ reference).square().sum() /
                                                         reference.square().sum()) if projection is not None else None),
                        "sigma_a_proxy_relative_error": abs(nm[module]["sigma_a_proxy"] /
                                                           (.01 * cm[module]["observed_a_max"]) - 1),
                        "sigma_b_proxy_relative_error": abs(nm[module]["sigma_b_proxy"] /
                                                           (.01 * cm[module]["observed_b_max"]) - 1)})
    report = {"status": "completed", "uses_private_reference": True,
              "private_reference_note": "仅全部分数提交后，使用额外配对干净上传评价残差误差；不进入 noisy 打分或参数选择",
              "all_scores_committed_before_reference": True,
              "aggregate": aggregate(rows, ("image", "method"), ("relative_residual_l2_error",
                  "retained_clean_energy", "sigma_a_proxy_relative_error", "sigma_b_proxy_relative_error")),
              "rows": rows, "contains_private_tokens": False}
    write_json(output / "paired-clean-diagnosis.json", report)


def evaluate_all(output):
    plan = verify_plan(output)
    source, groups = Path(plan["source"]), []
    for image in ("gt", "random"):
        completion = read_json(output / f"scores-complete-{image}.json")
        if completion["status"] != "completed" or completion["groups"] != 60:
            raise ValueError("Incomplete scoring")
        for condition in CONDITIONS:
            for index in range(30):
                folder = output / "scores" / image / f"{condition}-{index:02d}"
                meta = read_json(folder / "complete.json")
                if (file_hash(folder / "scores.safetensors") != meta["score_sha256"] or
                        file_hash(folder / "diagnostics.safetensors") != meta["diagnostic_sha256"] or
                        not meta["baseline_candidates_exact"] or meta["methods"] != list(METHODS)):
                    raise ValueError("Incomplete or changed score commitment")
                groups.append((image, condition, index, folder, meta))
    write_json(output / "evaluation-barrier.json", {
        "groups_verified": len(groups), "score_conditions": 480, "time": time.time(),
        "private_reference_read": False, "paired_clean_comparison_started": False})
    rows, module_rows = [], []
    for image, condition, index, folder, meta in groups:
        refs = read_tensors(source / "runs" / f"{condition}-{index:02d}" /
                            "captures" / image / "private/text_tokens.safetensors")
        forbidden = set(meta["forbidden_ids"])
        truth = set(refs["questions"].flatten().tolist() + refs["targets"].flatten().tolist()) - forbidden
        values = read_tensors(folder / "scores.safetensors")
        flags = values["ambiguous"].numpy()
        valid = np.ones(len(flags), dtype=bool)
        valid[list(forbidden)] = False
        allowed = np.flatnonzero(valid & ~flags)
        informative_truth = truth - set(np.flatnonzero(valid & flags).tolist())
        for method in METHODS:
            scores = values[method]
            selected = selected_ids(scores, values["ambiguous"], forbidden)
            order = allowed[np.lexsort((allowed, scores.numpy()[allowed]))].tolist()
            positions = {token: rank + 1 for rank, token in enumerate(order)}
            ranks = sorted(positions[t] for t in informative_truth)
            ap = statistics.fmean((i + 1) / rank for i, rank in enumerate(ranks)) if ranks else None
            rows.append({"condition": condition, "image": image, "index": index, "method": method,
                         "length_bin": plan["length_bins_by_index"][str(index)],
                         "candidate_recall": len(selected & truth) / len(truth),
                         "full_coverage": truth <= selected, "informative_ap": ap,
                         "selected_count": len(selected)})
        for module, diagnostic in meta["diagnostics"].items():
            module_rows.append({"condition": condition, "image": image, "index": index,
                                "module": module, **diagnostic})
    fields = ("candidate_recall", "full_coverage", "informative_ap", "selected_count")
    overall = aggregate(rows, ("condition", "image", "method"), fields)
    if len(rows) != 480 or len(overall) != 16 or any(r["n"] != 30 for r in overall):
        raise ValueError("Incomplete fixed matrix")
    diagnostic_fields = ("raw_identity_relative_error", "residual_identity_relative_error",
        "corrected_identity_relative_error", "relative_a_correction", "raw_numerical_rank",
        "residual_numerical_rank", "sketch_numerical_rank", "length_rank", "gate_rank",
        "length_fallback", "gate_fallback", "null_threshold")
    write_json(output / "summary.json", {"status": "completed", "cohort_n": 30,
        "score_groups": 120, "score_conditions": 480, "source_sha256": plan["source_sha256"],
        "aggregate": overall, "by_length": aggregate(rows, ("condition", "image", "method", "length_bin"), fields),
        "module_aggregate": aggregate(module_rows, ("condition", "image"), diagnostic_fields),
        "rows": rows, "module_rows": module_rows,
        "uses_private_reference": False, "contains_private_tokens": False,
        "all_scores_committed_before_truth": True, "no_method_selected_by_truth": True})
    paired_clean_diagnosis(output, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--image", choices=("gt", "random"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}-{args.image or 'all'}.json"
    if state.exists():
        raise FileExistsError("No implicit retry")
    write_json(state, {"status": "running"})
    try:
        if args.phase == "prepare":
            if args.source is None:
                raise ValueError("Source is required")
            prepare(args.output, args.source)
        elif args.phase == "score":
            if args.image is None:
                raise ValueError("Image condition required")
            score_all(args.output, args.image)
        else:
            evaluate_all(args.output)
        write_json(state, {"status": "completed"})
    except BaseException:
        write_json(state, {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
