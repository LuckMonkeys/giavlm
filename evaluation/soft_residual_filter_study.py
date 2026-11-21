"""Fixed-upload soft public-input suppression; score first, evaluate later.

The production attack is unchanged. Every alpha transforms gradients and token
features identically. A common hard-projection ambiguity policy fixes the token
budget across alpha; truth is read only after all score files commit.
"""
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
from core.artifacts import file_hash, read_json, read_tensors, source_fingerprint, write_json, write_tensors
from core.config import DAGEROptions
from core.fl import load_observation, restore_model


ALPHAS = (0., .5, .9, 1.)
CONDITIONS = ("fedsgd-b1", "noise", "sparsify")


def alpha_key(alpha):
    return "alpha_" + str(alpha).replace(".", "p")


def common_flags(features, public_basis, tolerance):
    projected = residual(features, public_basis)
    return projected.norm(dim=-1) <= tolerance * features.norm(dim=-1).clamp_min(
        torch.finfo(features.dtype).tiny)


@torch.no_grad()
def weighted_scores(gradients, features, public_basis, alpha, options, *, projected_features=None,
                    ambiguous=None):
    if not 0 <= alpha <= 1:
        raise ValueError("Suppression alpha must be in [0, 1]")
    blocks = [g.to(features).detach() / g.to(features).norm()
              for g in gradients if g.norm() > 0]
    if not blocks:
        raise ValueError("Zero observed gradient")
    matrix = torch.cat(blocks)
    matrix = matrix - alpha * (matrix @ public_basis.T) @ public_basis
    basis, singular = row_basis(matrix, options.rank_rtol, options.rank_atol)
    if ambiguous is None:
        ambiguous = common_flags(features, public_basis, options.zero_tolerance)
    scores = features.new_empty(len(features))
    for start in range(0, len(features), options.vocab_chunk_size):
        x = features[start:start + options.vocab_chunk_size]
        component = ((x @ public_basis.T) @ public_basis if projected_features is None else
                     projected_features[start:start + options.vocab_chunk_size])
        x = x - alpha * component
        norm = x.norm(dim=-1)
        distance = residual(x, basis).norm(dim=-1) / norm.clamp_min(torch.finfo(x.dtype).tiny)
        scores[start:start + len(x)] = distance.clamp(0, 1)
    scores = torch.where(ambiguous, torch.zeros_like(scores), scores)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite soft residual scores")
    return scores, {"rank": len(basis), "public_rank": len(public_basis),
                    "singular_values": singular.cpu().tolist()}


def selected_ids(scores, ambiguous, forbidden, k=100):
    values = scores.detach().cpu().numpy()
    flags = ambiguous.detach().cpu().numpy()
    valid = np.ones(len(values), dtype=bool)
    valid[list(forbidden)] = False
    allowed = np.flatnonzero(valid & ~flags)
    ranked = allowed[np.lexsort((allowed, values[allowed]))]
    return set(ranked[:k].tolist()) | set(np.flatnonzero(valid & flags).tolist())


def prepare(output, source):
    if (output / "plan.json").exists():
        raise FileExistsError("Use a new study directory")
    original = read_json(source / "plan.json")
    jobs = [j for j in original["jobs"] if j["condition"] in CONDITIONS]
    if len(jobs) != 90 or any(j["n"] != 1 for j in jobs):
        raise ValueError("Expected 30 single-example uploads per condition")
    # These names identify public capture artifacts, never private sample IDs.
    sources = {}
    for job in jobs:
        for image in ("gt", "random"):
            public = source / "runs" / job["id"] / "captures" / image / "public"
            meta = read_json(public / "observation.json")
            sources[f"{job['id']}:{image}"] = {"observation_id": meta["observation_id"],
                "metadata_sha256": file_hash(public / "observation.json"),
                "update_sha256": meta["update_sha256"],
                "public_images_sha256": meta["public_images_sha256"]}
    write_json(output / "plan.json", {
        "schema_version": 1, "source_sha256": source_fingerprint(),
        "source": str(source.resolve()), "source_plan_sha256": file_hash(source / "plan.json"),
        "model": original["model"], "model_fingerprint": original["model_fingerprint"],
        "model_metadata_sha256": original["model_metadata_sha256"], "sources": sources,
        "alphas": list(ALPHAS), "conditions": list(CONDITIONS), "cohort_n": 30,
        "length_bins": {"1-16": 10, "17-32": 10, "33-64": 10},
        "score_groups": 180, "score_conditions": 720,
        "options": original["attack_specs"]["1"]["gt"]["dager"],
        "common_ambiguity": "same hard-projection public-overlap IDs for every alpha",
        "informative_k": 100, "seed": 0, "new_gradients_or_noise": False,
        "score_before_truth": True, "initial_model_immutable": True})


def verify_plan(output):
    plan = read_json(output / "plan.json")
    if source_fingerprint() != plan["source_sha256"]:
        raise ValueError("Frozen source changed")
    if file_hash(Path(plan["model"]) / "model.json") != plan["model_metadata_sha256"]:
        raise ValueError("Public parent model changed")
    return plan


@torch.no_grad()
def score_all(output, image):
    plan = verify_plan(output)
    source = Path(plan["source"])
    options = DAGEROptions(**plan["options"])
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Model differs from original capture")
    versions = {name: p._version for name, p in adapter.named_parameters()}
    layernorm = adapter.backend.language_model.model.layers[0].input_layernorm
    features = layernorm(adapter.embedding().weight).detach().double()
    forbidden = forbidden_token_ids(adapter)
    completed = []
    for index in range(30):
        roots = {c: source / "runs" / f"{c}-{index:02d}" for c in CONDITIONS}
        observations = {}
        for condition, root in roots.items():
            public = root / "captures" / image / "public"
            expected = plan["sources"][f"{condition}-{index:02d}:{image}"]
            if file_hash(public / "observation.json") != expected["metadata_sha256"]:
                raise ValueError("Public observation metadata changed")
            obs = load_observation(public, "cuda:0")
            if obs.model_fingerprint != plan["model_fingerprint"]:
                raise ValueError("Observed model mismatch")
            observations[condition] = obs
        obs = observations["fedsgd-b1"]
        adapter.set_training_spec(obs.training)
        if image == "gt":
            images = obs.public_images
            if any(plan["sources"][f"{c}-{index:02d}:{image}"]["public_images_sha256"] !=
                   plan["sources"][f"fedsgd-b1-{index:02d}:{image}"]["public_images_sha256"]
                   for c in CONDITIONS):
                raise ValueError("Image context differs between defense conditions")
        else:
            if any(o.public_images is not None for o in observations.values()):
                raise ValueError("Random worker received private image")
            images = torch.rand(1, 3, obs.model.image_size, obs.model.image_size,
                                generator=torch.Generator(device="cpu").manual_seed(plan["seed"]))
        view = LlavaTextView(adapter, replace(obs, public_images=images))
        public_basis, _ = row_basis(view.public_layer0.double(), options.rank_rtol, options.rank_atol)
        # Projection and ambiguity depend on the declared input context, never
        # on a defense or alpha. Reuse them to keep all four arms identical.
        projected_features = torch.empty_like(features)
        for start in range(0, len(features), options.vocab_chunk_size):
            x = features[start:start + options.vocab_chunk_size]
            projected_features[start:start + len(x)] = (x @ public_basis.T) @ public_basis
        flags = (features - projected_features).norm(dim=-1) <= (
            options.zero_tolerance * features.norm(dim=-1))
        for condition, observation in observations.items():
            directory = output / "scores" / image / f"{condition}-{index:02d}"
            if directory.exists():
                raise FileExistsError("No implicit scoring retry")
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            gradients = [observation.tensors[n] for n in gradient_groups(adapter, observation, "qkv")[0]]
            scores, diagnostics = {}, {}
            for alpha in ALPHAS:
                key = alpha_key(alpha)
                values, diagnostics[key] = weighted_scores(
                    gradients, features, public_basis, alpha, options,
                    projected_features=projected_features, ambiguous=flags)
                scores[key] = values.cpu()
            baseline = read_tensors(roots[condition] / "attacks" / image / "token_candidates.safetensors")
            if not torch.equal(flags.cpu(), baseline["ambiguous"]):
                raise ValueError("Hard endpoint ambiguity differs from original")
            delta = float((scores[alpha_key(1.)] - baseline["scores"]).abs().max())
            if delta > 1e-6:
                raise ValueError(f"Hard endpoint scores differ by {delta}")
            if selected_ids(scores[alpha_key(1.)], flags.cpu(), forbidden) != set(
                    baseline["selected_ids"].tolist()):
                raise ValueError("Hard endpoint candidate set differs from original")
            write_tensors(directory / "scores.safetensors", {**scores, "ambiguous": flags.cpu()})
            write_json(directory / "complete.json", {
                "status": "completed", "condition": condition, "image": image, "index": index,
                "alphas": list(ALPHAS), "forbidden_ids": sorted(forbidden),
                "baseline_max_score_delta": delta, "baseline_candidates_exact": True,
                "diagnostics": diagnostics, "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "score_sha256": file_hash(directory / "scores.safetensors"),
                "source_sha256": plan["source_sha256"], "model_fingerprint": plan["model_fingerprint"],
                "observation_id": plan["sources"][f"{condition}-{index:02d}:{image}"]["observation_id"]})
            completed.append(f"{condition}-{index:02d}")
            write_json(output / f"progress-{image}.json", {
                "status": "running", "completed": len(completed), "total": 90,
                "last_group": completed[-1], "score_conditions": len(completed) * len(ALPHAS)})
            print("committed", image, completed[-1], "alpha1 delta", delta, flush=True)
        del observations, obs, observation, view, projected_features, images
        torch.cuda.empty_cache()
    if any(p._version != versions[name] for name, p in adapter.named_parameters()):
        raise ValueError("Read-only scoring mutated model")
    write_json(output / f"scores-complete-{image}.json", {
        "status": "completed", "groups": 90, "score_conditions": 360,
        "source_sha256": plan["source_sha256"], "initial_model_immutable": True})
    write_json(output / f"progress-{image}.json", {"status": "completed", "completed": 90, "total": 90})


def evaluate_all(output):
    plan = verify_plan(output)
    source = Path(plan["source"])
    groups = []
    for image in ("gt", "random"):
        read_json(output / f"scores-complete-{image}.json")
        for condition in CONDITIONS:
            for index in range(30):
                folder = output / "scores" / image / f"{condition}-{index:02d}"
                meta = read_json(folder / "complete.json")
                if file_hash(folder / "scores.safetensors") != meta["score_sha256"]:
                    raise ValueError("Committed scores changed")
                groups.append((image, condition, index, folder, meta))
    rows = []
    for image, condition, index, folder, meta in groups:
        # Only this evaluation phase opens the previously captured references.
        refs = read_tensors(source / "runs" / f"{condition}-{index:02d}" /
                            "captures" / image / "private/text_tokens.safetensors")
        truth = set(refs["questions"].flatten().tolist() + refs["targets"].flatten().tolist())
        forbidden = set(meta["forbidden_ids"])
        truth -= forbidden
        values = read_tensors(folder / "scores.safetensors")
        flags = values["ambiguous"].numpy()
        valid = np.ones(len(flags), dtype=bool)
        valid[list(forbidden)] = False
        allowed = np.flatnonzero(valid & ~flags)
        informative_truth = truth - set(np.flatnonzero(valid & flags).tolist())
        for alpha in ALPHAS:
            scores = values[alpha_key(alpha)]
            selected = selected_ids(scores, values["ambiguous"], forbidden)
            order = allowed[np.lexsort((allowed, scores.numpy()[allowed]))].tolist()
            positions = {token: rank + 1 for rank, token in enumerate(order)}
            ranks = sorted(positions[t] for t in informative_truth)
            ap = statistics.fmean((i + 1) / rank for i, rank in enumerate(ranks)) if ranks else None
            rows.append({"condition": condition, "image": image, "index": index, "alpha": alpha,
                         "candidate_recall": len(selected & truth) / len(truth),
                         "full_coverage": truth <= selected, "informative_ap": ap,
                         "selected_count": len(selected)})
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["condition"], row["image"], row["alpha"])].append(row)
    aggregate = [{"condition": condition, "image": image, "alpha": alpha, "n": len(items),
                  "candidate_recall": statistics.fmean(r["candidate_recall"] for r in items),
                  "full_coverage_count": sum(r["full_coverage"] for r in items),
                  "informative_ap": statistics.fmean(r["informative_ap"] for r in items
                                                     if r["informative_ap"] is not None),
                  "selected_count_mean": statistics.fmean(r["selected_count"] for r in items)}
                 for (condition, image, alpha), items in buckets.items()]
    if len(rows) != 720 or any(r["n"] != 30 for r in aggregate):
        raise ValueError("Incomplete fixed matrix")
    write_json(output / "summary.json", {
        "status": "completed", "cohort_n": 30, "score_groups": 180, "score_conditions": 720,
        "source_sha256": plan["source_sha256"], "aggregate": aggregate, "rows": rows,
        "alphas_frozen_before_scoring": True, "all_scores_committed_before_truth": True,
        "uses_private_reference": False, "contains_private_tokens": False,
        "note": "Paired filter-only diagnostic; no alpha chosen by truth, no sequence reconstruction. "
                "Common ambiguous extras for all alpha. AP excludes common ambiguous IDs."})


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
        raise FileExistsError("Inspect prior state; no implicit retry")
    write_json(state, {"status": "running"})
    try:
        if args.phase == "prepare":
            if args.source is None:
                raise ValueError("Source study required")
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
