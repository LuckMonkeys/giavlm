"""First-stage DAGER sensitivity to surrogate visual subspaces.

Scoring reads only a committed public observation. Private token IDs are joined
later by ``evaluate_scores`` after every vocabulary-score artifact is committed.
Blends with the declared true public image are oracle sensitivity diagnostics,
not attacks under an image-private knowledge condition.
"""
import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path

import numpy as np
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids, gradient_groups
from attacks.analytic.dager_subspace import SpanFilter, residual, row_basis
from core.adapters.llava import LlavaTextView
from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json, write_tensors)
from core.config import DAGEROptions, TrainingSpec, digest
from core.fl import load_observation, restore_model
from evaluation.dager_ablation import _ranking_metrics
from evaluation.token_recovery import set_metrics


SCHEMA = 1


def _label_alpha(alpha):
    return str(alpha).replace(".", "p")


def condition_plan(alphas, seeds, random_seeds):
    plan = [{"label": "raw", "kind": "raw"},
            {"label": "template_only", "kind": "template_only"},
            {"label": "true_image", "kind": "true_image", "alpha": 0.0}]
    plan.extend({"label": f"random_subspace_s{seed}", "kind": "random_subspace",
                 "seed": seed} for seed in random_seeds)
    plan.extend({"label": f"blend_a{_label_alpha(alpha)}_s{seed}", "kind": "blend",
                 "alpha": alpha, "seed": seed}
                for alpha in alphas for seed in seeds)
    return plan


def _public_parts(view):
    template_mask = view.known_mask.clone()
    template_mask[view.visual_start:view.visual_stop] = False
    template = view.layer_input(view.embeddings[:, template_mask], 0)[0]
    image = view.layer_input(
        view.embeddings[:, view.visual_start:view.visual_stop], 0)[0]
    return template, image


def _basis(vectors, options):
    return row_basis(vectors.to(getattr(torch, options.analysis_dtype)),
                     options.rank_rtol, options.rank_atol)[0]


def _incremental_image_basis(image_inputs, template_basis, options):
    return _basis(residual(image_inputs.to(template_basis), template_basis), options)


def _subspace_overlap(left, right):
    denominator = min(len(left), len(right))
    if not denominator:
        return 0.0
    return float((left @ right.T).square().sum().div(denominator))


def _image_metrics(image, truth, image_basis, truth_basis, image_inputs, truth_inputs):
    mse = float((image.float() - truth.float()).square().mean())
    cosine = torch.nn.functional.cosine_similarity(
        image_inputs.float().flatten()[None], truth_inputs.float().flatten()[None]).item()
    return {"pixel_mse": mse, "pixel_psnr": None if mse == 0 else -10 * math.log10(mse),
            "aligned_visual_cosine": cosine,
            "visual_subspace_overlap": _subspace_overlap(image_basis, truth_basis),
            "visual_subspace_rank": len(image_basis)}


@torch.no_grad()
def _score(span, candidate_features, chunk_size):
    scores = torch.empty(len(candidate_features), dtype=span.basis.dtype, device="cpu")
    ambiguous = torch.empty(len(candidate_features), dtype=torch.bool, device="cpu")
    for start in range(0, len(candidate_features), chunk_size):
        stop = min(start + chunk_size, len(candidate_features))
        values, flags = span.score(candidate_features[start:stop])
        scores[start:stop] = values.cpu()
        ambiguous[start:stop] = flags.cpu()
    return scores, ambiguous


def _random_public_inputs(template, template_basis, image_rank, options, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(image_rank, template.shape[1], generator=generator,
                         dtype=getattr(torch, options.analysis_dtype), device="cpu")
    values = values.to(template_basis.device)
    values = residual(values, template_basis)
    random_basis = _basis(values, options)
    if len(random_basis) != image_rank:
        raise RuntimeError("Matched random visual subspace lost numerical rank")
    return torch.cat([template.to(random_basis), random_basis]), random_basis


def score_surrogates(observation_dir, model_dir, config_path, output, device="cuda:0",
                     alphas=(0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1.0),
                     seeds=(0, 1, 2), random_seeds=(0, 1, 2), resume=False):
    """Commit full-vocabulary scores without reading any private reference."""
    observation_dir, model_dir = Path(observation_dir), Path(model_dir)
    config_path, output = Path(config_path), Path(output)
    protocol = read_json(config_path)
    options = DAGEROptions(**protocol["attack"]["dager"])
    options = replace(options, mode="public_residual")
    plan = condition_plan(tuple(alphas), tuple(seeds), tuple(random_seeds))
    signature = digest({
        "schema": SCHEMA, "observation": file_hash(observation_dir / "update.safetensors"),
        "public_image": file_hash(observation_dir / "public_images.safetensors"),
        "model_json": file_hash(model_dir / "model.json"),
        "model_tensors": file_hash(model_dir / "model.safetensors"),
        "options": asdict(options), "plan": plan, "source": source_fingerprint()})
    state_path = output / "state.json"
    if state_path.exists():
        state = read_json(state_path)
        if not resume:
            raise FileExistsError("Diagnostic exists; pass --resume to continue")
        if state.get("schema_version") != SCHEMA or state.get("signature") != signature:
            raise ValueError("Diagnostic resume inputs or implementation changed")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Diagnostic output is nonempty")
    else:
        state = {"schema_version": SCHEMA, "signature": signature, "completed": []}
        write_json(state_path, state)

    obs = load_observation(observation_dir, device)
    if obs.model.family != "llava" or obs.sample_count != 1 or obs.public_images is None:
        raise ValueError("Surrogate diagnostic requires one declared public LLaVA image")
    adapter = restore_model(model_dir, device)
    if adapter.fingerprint() != obs.model_fingerprint:
        raise ValueError("Observation and restored model fingerprints differ")
    adapter.set_training_spec(obs.training)
    names = gradient_groups(adapter, obs, options.projections)[0]
    gradients = [obs.tensors[name] for name in names]
    true_image = obs.public_images

    true_view = LlavaTextView(adapter, obs)
    template_inputs, true_image_inputs = _public_parts(true_view)
    template_basis = _basis(template_inputs, options)
    truth_image_basis = _incremental_image_basis(
        true_image_inputs, template_basis, options)
    true_full_span = SpanFilter.build(gradients, true_view.public_layer0, options)
    incremental_rank = (true_full_span.diagnostics["public_rank"]
                        - len(template_basis))
    if incremental_rank <= 0:
        raise RuntimeError("True image added no public-input directions")
    if incremental_rank != len(truth_image_basis):
        raise RuntimeError("Template/image public ranks are not additive")
    candidate_features = true_view.layer_input(adapter.embedding().weight, 0).to(
        getattr(torch, options.analysis_dtype))
    forbidden = sorted(forbidden_token_ids(adapter))
    noises = {seed: torch.rand(true_image.shape,
                               generator=torch.Generator().manual_seed(20261002 + seed))
              for seed in seeds}

    try:
        for condition in plan:
            label, kind = condition["label"], condition["kind"]
            metadata_path = output / "conditions" / f"{label}.json"
            tensor_path = output / "conditions" / f"{label}.safetensors"
            if label in state["completed"]:
                metadata = read_json(metadata_path)
                if file_hash(tensor_path) != metadata["tensor_sha256"]:
                    raise ValueError(f"Condition artifact integrity failed: {label}")
                continue

            image_metrics = None
            if kind == "raw":
                span = SpanFilter.build(
                    gradients, true_view.public_layer0, replace(options, mode="raw"))
            elif kind == "template_only":
                span = SpanFilter.build(gradients, template_inputs, options)
            elif kind == "random_subspace":
                public, random_basis = _random_public_inputs(
                    template_inputs, template_basis, incremental_rank, options,
                    20262002 + condition["seed"])
                span = SpanFilter.build(gradients, public, options)
                condition = {**condition, "matched_visual_rank": len(random_basis)}
            else:
                if kind == "true_image":
                    image = true_image
                    view = true_view
                    image_inputs, image_basis = true_image_inputs, truth_image_basis
                elif kind == "blend":
                    alpha = condition["alpha"]
                    image = ((1 - alpha) * true_image + alpha * noises[condition["seed"]]).clamp(0, 1)
                    view = LlavaTextView(adapter, replace(obs, public_images=image))
                    _, image_inputs = _public_parts(view)
                    image_basis = _incremental_image_basis(
                        image_inputs, template_basis, options)
                else:
                    raise ValueError(f"Unknown surrogate condition: {kind}")
                span = SpanFilter.build(gradients, view.public_layer0, options)
                image_metrics = _image_metrics(
                    image, true_image, image_basis, truth_image_basis,
                    image_inputs, true_image_inputs)

            scores, ambiguous = _score(span, candidate_features, options.vocab_chunk_size)
            write_tensors(tensor_path, {"scores": scores, "ambiguous": ambiguous})
            write_json(metadata_path, {
                "schema_version": SCHEMA, "signature": signature,
                "condition": condition, "span": span.diagnostics,
                "image_similarity": image_metrics, "forbidden_ids": forbidden,
                "tensor_sha256": file_hash(tensor_path)})
            state["completed"].append(label)
            write_json(state_path, state)
            print(f"committed {label} ({len(state['completed'])}/{len(plan)})", flush=True)

        write_json(output / "scores_complete.json", {
            "schema_version": SCHEMA, "signature": signature,
            "condition_count": len(plan), "conditions": [row["label"] for row in plan],
            "reference_policy": "No private token IDs or private text read during scoring",
            "diagnostic_boundary": (
                "True-image blends are oracle public-image misspecification diagnostics")})
    except Exception as error:
        write_json(output / "failed.json", {
            "schema_version": SCHEMA, "signature": signature,
            "error_type": type(error).__name__, "reason": str(error),
            "completed": state["completed"]})
        raise
    return output / "scores_complete.json"


def _condition_metrics(tensors, metadata, truth, by_field, topks):
    forbidden = set(metadata["forbidden_ids"])
    ranking = sorted((token_id for token_id in range(len(tensors["scores"]))
                      if token_id not in forbidden),
                     key=lambda token_id: (float(tensors["scores"][token_id]), token_id))
    result = _ranking_metrics(ranking, truth, topks)
    selected = set(ranking[:50])
    informative = [token_id for token_id in ranking if not tensors["ambiguous"][token_id]]
    result.update({
        "recall_by_field_at_50": {
            field: set_metrics(selected, ids)["recall"] for field, ids in by_field.items()},
        "ambiguous_in_top50": sum(bool(tensors["ambiguous"][token_id])
                                  for token_id in selected),
        "ambiguous_true_tokens": sum(bool(tensors["ambiguous"][token_id])
                                     for token_id in truth),
        "informative_top50": set_metrics(set(informative[:50]), truth),
    })
    return result


def _curve_value(metrics, topk, field="recall"):
    return next(row[field] for row in metrics["curve"] if row["topk"] == topk)


def _correlation_result(result):
    return {"spearman": (float(result.statistic)
                          if np.isfinite(result.statistic) else None),
            "pvalue": float(result.pvalue) if np.isfinite(result.pvalue) else None}


def evaluate_scores(scores_dir, truth_dir, output=None,
                    topks=(10, 20, 50, 100, 200, 500, 1000)):
    """Join committed scores with private IDs and retain aggregates only."""
    scores_dir, truth_dir = Path(scores_dir), Path(truth_dir)
    complete = read_json(scores_dir / "scores_complete.json")
    truth_meta = read_json(truth_dir / "truth.json")
    training = TrainingSpec(**truth_meta["training"])
    references = read_tensors(truth_dir / "text_tokens.safetensors")
    reports = {}
    common_forbidden = None
    for label in complete["conditions"]:
        metadata_path = scores_dir / "conditions" / f"{label}.json"
        tensor_path = scores_dir / "conditions" / f"{label}.safetensors"
        metadata = read_json(metadata_path)
        if metadata["signature"] != complete["signature"]:
            raise ValueError(f"Condition signature differs: {label}")
        if file_hash(tensor_path) != metadata["tensor_sha256"]:
            raise ValueError(f"Condition tensor integrity failed: {label}")
        forbidden = set(metadata["forbidden_ids"])
        if common_forbidden is not None and forbidden != common_forbidden:
            raise ValueError("Conditions use different forbidden token sets")
        common_forbidden = forbidden
        private_fields = ["targets"] + (["questions"] if training.question_private else [])
        by_field = {field: set(references[field].flatten().tolist()) - forbidden
                    for field in private_fields}
        truth = set().union(*by_field.values())
        metrics = _condition_metrics(
            read_tensors(tensor_path), metadata, truth, by_field, topks)
        reports[label] = {"condition": metadata["condition"],
                          "span": metadata["span"],
                          "image_similarity": metadata["image_similarity"],
                          "metrics": metrics}

    raw = reports["raw"]["metrics"]
    for report in reports.values():
        report["delta_vs_raw"] = {
            f"recall_at_{topk}": _curve_value(report["metrics"], topk)
            - _curve_value(raw, topk) for topk in topks}
        report["delta_vs_raw"]["average_precision"] = (
            report["metrics"]["average_precision"] - raw["average_precision"])

    blends = [report for report in reports.values()
              if report["condition"]["kind"] == "blend"]
    alpha_summary = {}
    for alpha in sorted({row["condition"]["alpha"] for row in blends}):
        rows = [row for row in blends if row["condition"]["alpha"] == alpha]
        alpha_summary[str(alpha)] = {
            "count": len(rows),
            "mean_visual_subspace_overlap": float(np.mean([
                row["image_similarity"]["visual_subspace_overlap"] for row in rows])),
            "mean_recall_at_50": float(np.mean([
                _curve_value(row["metrics"], 50) for row in rows])),
            "minimum_recall_at_50": min(_curve_value(row["metrics"], 50) for row in rows),
            "maximum_recall_at_50": max(_curve_value(row["metrics"], 50) for row in rows),
            "mean_average_precision": float(np.mean([
                row["metrics"]["average_precision"] for row in rows])),
            "mean_ambiguous_in_top50": float(np.mean([
                row["metrics"]["ambiguous_in_top50"] for row in rows])),
        }

    overlap = np.asarray([row["image_similarity"]["visual_subspace_overlap"]
                          for row in blends])
    recall = np.asarray([_curve_value(row["metrics"], 50) for row in blends])
    average_precision = np.asarray([row["metrics"]["average_precision"] for row in blends])
    from scipy.stats import spearmanr
    recall_corr = spearmanr(overlap, recall)
    ap_corr = spearmanr(overlap, average_precision)
    report = {
        "schema_version": SCHEMA, "unit": "unique_token_id",
        "score_signature": complete["signature"], "topks": list(topks),
        "conditions": reports, "blend_alpha_summary": alpha_summary,
        "descriptive_correlations": {
            "visual_subspace_overlap_vs_recall_at_50": _correlation_result(recall_corr),
            "visual_subspace_overlap_vs_average_precision": _correlation_result(ap_corr)},
        "reference_policy": (
            "Private token IDs joined only after all score artifacts committed; "
            "no token IDs or text retained"),
        "limitations": [
            "Single captured gradient; surrogate conditions are repeated measures, not samples",
            "True-image blends are oracle sensitivity diagnostics, not an image-private attack",
            "Correlations are descriptive and do not establish cross-sample generalization"]}
    destination = Path(output) if output else scores_dir / "evaluation.json"
    write_json(destination, report)
    return report


def _float_tuple(value):
    return tuple(float(item) for item in value.split(",") if item)


def _int_tuple(value):
    return tuple(int(item) for item in value.split(",") if item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--observation", required=True)
    score.add_argument("--model", required=True)
    score.add_argument("--config", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--alphas", type=_float_tuple,
                       default=(0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1.0))
    score.add_argument("--seeds", type=_int_tuple, default=(0, 1, 2))
    score.add_argument("--random-seeds", type=_int_tuple, default=(0, 1, 2))
    score.add_argument("--resume", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--truth", required=True)
    evaluate.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.stage == "score":
            score_surrogates(args.observation, args.model, args.config, args.output,
                             args.device, args.alphas, args.seeds, args.random_seeds,
                             args.resume)
        else:
            evaluate_scores(args.scores, args.truth, args.output)
    except Exception as error:
        failure_dir = Path(args.output) if args.stage == "score" else Path(args.scores)
        if args.stage == "evaluate" and args.output:
            failure_dir = Path(args.output).parent
        write_json(failure_dir / f"failed_{args.stage}.json", {
            "status": "failed", "stage": args.stage,
            "error_type": type(error).__name__, "reason": str(error)})
        raise


if __name__ == "__main__":
    main()
