"""Fixed-upload ridge spectral scoring, with a score-before-truth barrier.

The production attack, captures, public-input suppression and candidate budget
are unchanged. Beta is fixed in advance; no best beta is selected by references.
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
from attacks.analytic.dager_subspace import residual
from core.adapters.llava import LlavaTextView
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.config import DAGEROptions
from core.fl import load_observation, restore_model
from evaluation.soft_residual_filter_study import (
    CONDITIONS, prepare as prepare_uploads, selected_ids, verify_plan,
)


BETAS = (0., .01, .1, 1.)


def beta_key(beta):
    return "beta_" + str(beta).replace(".", "p")


@torch.no_grad()
def spectral_scores(gradients, features, public_basis, options, *, projected_features=None,
                    ambiguous=None):
    blocks = [g.to(features).detach() / g.to(features).norm()
              for g in gradients if g.norm() > 0]
    if not blocks:
        raise ValueError("Zero observed gradient")
    matrix = residual(torch.cat(blocks), public_basis)
    _, singular, vectors = torch.linalg.svd(matrix, full_matrices=False)
    retained = singular > max(options.rank_atol, options.rank_rtol * float(singular[0]))
    basis, singular = vectors[retained], singular[retained]
    if not len(singular):
        raise ValueError("Zero numerical rank")
    energy = singular.square()
    scale = torch.quantile(energy, .5)
    weights = {b: energy / (energy + b * scale) for b in BETAS}
    scores = {beta_key(b): features.new_empty(len(features)) for b in BETAS}
    if projected_features is None:
        projected_features = (features @ public_basis.T) @ public_basis
    projected = features - projected_features
    if ambiguous is None:
        ambiguous = projected.norm(dim=-1) <= options.zero_tolerance * features.norm(dim=-1)
    for start in range(0, len(features), options.vocab_chunk_size):
        x = projected[start:start + options.vocab_chunk_size]
        coordinates = x @ basis.T
        norm = x.norm(dim=-1).clamp_min(torch.finfo(x.dtype).tiny)
        orthogonal = x - coordinates @ basis
        base = orthogonal.norm(dim=-1) / norm
        for beta in BETAS:
            # Preserve the original endpoint exactly and avoid cancellation in
            # 1 - captured_energy / total_energy near a true span member.
            values = base if beta == 0 else (
                orthogonal.square().sum(-1) +
                (coordinates.square() * (1 - weights[beta])).sum(-1)
            ).clamp_min(0).sqrt() / norm
            scores[beta_key(beta)][start:start + len(x)] = values.clamp(0, 1)
    scores = {k: torch.where(ambiguous, torch.zeros_like(v), v) for k, v in scores.items()}
    if any(not torch.isfinite(v).all() for v in scores.values()):
        raise ValueError("Nonfinite spectral score")
    diagnostic = {
        "rank": len(singular), "public_rank": len(public_basis),
        "singular_values": singular.cpu().tolist(), "median_energy": float(scale),
        "min_max_ratio": float(singular[-1] / singular[0]),
        "median_max_ratio": float(torch.quantile(singular, .5) / singular[0]),
        "max_adjacent_gap": float((singular[:-1] / singular[1:]).max()) if len(singular) > 1 else None,
        "stable_rank": float(energy.sum() / energy[0]),
        "effective_dimensions": {beta_key(b): float(w.sum()) for b, w in weights.items()},
        "lambdas": {beta_key(b): float(b * scale) for b in BETAS},
    }
    return scores, diagnostic


def prepare(output, source):
    prepare_uploads(output, source)
    plan = read_json(output / "plan.json")
    del plan["alphas"]
    plan.update(betas=list(BETAS), method="ridge_spectral_token_distance",
                lambda_rule="beta * median(retained singular values squared)",
                suppression_alpha=1., parameter_selection="none; fixed paired diagnostic")
    write_json(output / "plan.json", plan)


@torch.no_grad()
def score_all(output, image):
    from attacks.analytic.dager_subspace import row_basis

    plan = verify_plan(output)
    if plan["betas"] != list(BETAS):
        raise ValueError("Prespecified betas changed")
    source = Path(plan["source"])
    options = DAGEROptions(**plan["options"])
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Model differs from original capture")
    versions = {name: p._version for name, p in adapter.named_parameters()}
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
                raise ValueError("Image context differs across defense conditions")
        else:
            if any(o.public_images is not None for o in observations.values()):
                raise ValueError("Random worker received a private image")
            images = torch.rand(1, 3, obs.model.image_size, obs.model.image_size,
                                generator=torch.Generator(device="cpu").manual_seed(plan["seed"]))
        view = LlavaTextView(adapter, replace(obs, public_images=images))
        public_basis, _ = row_basis(view.public_layer0.double(), options.rank_rtol, options.rank_atol)
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
            scores, diagnostic = spectral_scores(gradients, features, public_basis, options,
                projected_features=projected_features, ambiguous=flags)
            baseline = read_tensors(source / "runs" / f"{condition}-{index:02d}" /
                                    "attacks" / image / "token_candidates.safetensors")
            delta = float((scores[beta_key(0.)].cpu() - baseline["scores"]).abs().max())
            if not torch.equal(flags.cpu(), baseline["ambiguous"]) or delta > 1e-6:
                raise ValueError("Baseline endpoint differs from original scores")
            if selected_ids(scores[beta_key(0.)], flags, forbidden) != set(baseline["selected_ids"].tolist()):
                raise ValueError("Baseline endpoint candidates differ")
            write_tensors(directory / "scores.safetensors", {
                **{k: v.cpu() for k, v in scores.items()}, "ambiguous": flags.cpu()})
            write_json(directory / "complete.json", {
                "status": "completed", "condition": condition, "image": image, "index": index,
                "betas": list(BETAS), "forbidden_ids": sorted(forbidden), "diagnostics": diagnostic,
                "baseline_candidates_exact": True, "baseline_max_score_delta": delta,
                "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "score_sha256": file_hash(directory / "scores.safetensors"),
                "source_sha256": plan["source_sha256"], "model_fingerprint": plan["model_fingerprint"],
                "observation_id": plan["sources"][f"{condition}-{index:02d}:{image}"]["observation_id"]})
            completed += 1
            write_json(output / f"progress-{image}.json", {
                "status": "running", "completed": completed, "total": 90,
                "last_group": f"{condition}-{index:02d}"})
            print("committed", image, condition, index, "baseline delta", delta, flush=True)
        del observations, observation, obs, view, projected_features, images
        torch.cuda.empty_cache()
    if any(p._version != versions[name] for name, p in adapter.named_parameters()):
        raise ValueError("Read-only scoring mutated model")
    write_json(output / f"scores-complete-{image}.json", {
        "status": "completed", "groups": 90, "score_conditions": 360,
        "source_sha256": plan["source_sha256"], "initial_model_immutable": True})
    write_json(output / f"progress-{image}.json", {"status": "completed", "completed": 90, "total": 90})


def evaluate_all(output):
    plan = verify_plan(output)
    source, groups = Path(plan["source"]), []
    for image in ("gt", "random"):
        completion = read_json(output / f"scores-complete-{image}.json")
        if completion["status"] != "completed" or completion["groups"] != 90:
            raise ValueError("Incomplete scoring")
        for condition in CONDITIONS:
            for index in range(30):
                folder = output / "scores" / image / f"{condition}-{index:02d}"
                meta = read_json(folder / "complete.json")
                if (file_hash(folder / "scores.safetensors") != meta["score_sha256"] or
                        not meta["baseline_candidates_exact"] or meta["betas"] != list(BETAS)):
                    raise ValueError("Committed scores changed")
                groups.append((image, condition, index, folder, meta))
    # Commit the barrier before any private reference is opened.
    write_json(output / "evaluation-barrier.json", {
        "groups_verified": len(groups), "score_conditions": 720, "time": time.time(),
        "private_reference_read": False})
    rows = []
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
        for beta in BETAS:
            scores = values[beta_key(beta)]
            selected = selected_ids(scores, values["ambiguous"], forbidden)
            order = allowed[np.lexsort((allowed, scores.numpy()[allowed]))].tolist()
            positions = {token: rank + 1 for rank, token in enumerate(order)}
            ranks = sorted(positions[t] for t in informative_truth)
            ap = statistics.fmean((i + 1) / rank for i, rank in enumerate(ranks)) if ranks else None
            rows.append({"condition": condition, "image": image, "index": index, "beta": beta,
                         "candidate_recall": len(selected & truth) / len(truth),
                         "full_coverage": truth <= selected, "informative_ap": ap,
                         "selected_count": len(selected)})
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["condition"], row["image"], row["beta"])].append(row)
    aggregate = [{"condition": c, "image": image, "beta": b, "n": len(items),
                  "candidate_recall": statistics.fmean(r["candidate_recall"] for r in items),
                  "full_coverage_count": sum(r["full_coverage"] for r in items),
                  "informative_ap": statistics.fmean(r["informative_ap"] for r in items
                                                     if r["informative_ap"] is not None),
                  "selected_count_mean": statistics.fmean(r["selected_count"] for r in items)}
                 for (c, image, b), items in buckets.items()]
    if len(rows) != 720 or len(aggregate) != 24 or any(r["n"] != 30 for r in aggregate):
        raise ValueError("Incomplete fixed matrix")
    write_json(output / "summary.json", {
        "status": "completed", "cohort_n": 30, "score_groups": 180, "score_conditions": 720,
        "source_sha256": plan["source_sha256"], "aggregate": aggregate, "rows": rows,
        "betas_frozen_before_scoring": True, "all_scores_committed_before_truth": True,
        "uses_private_reference": False, "contains_private_tokens": False,
        "note": "Paired filter-only diagnostic; no beta chosen by truth; no sequence reconstruction. "
                "Common ambiguous extras; Informative AP excludes these IDs."})


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
