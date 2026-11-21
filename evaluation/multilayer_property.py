"""Public-only layer selection and frozen score fusion for visual properties."""

import argparse
from pathlib import Path

import numpy as np

from core.artifacts import file_hash, read_json, write_json
from evaluation.image_property_scoring import classification_stats, paired_bootstrap


LAYERS = (1, 8, 16, 24, 32)
OLD = Path("outputs/diagnostics/image_property_next_20261028")


def margins(scores):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite positive/negative prototype distances")
    return values[:, 1] - values[:, 0]


def freeze_rule(public_scores, public_metrics):
    """Only public development labels may influence the frozen rule."""
    selected = max(LAYERS, key=lambda layer: (
        np.mean([fold[f"gradient_{layer}"]["balanced_accuracy"] for fold in public_metrics.values()]),
        np.mean([fold[f"gradient_{layer}"]["class_auroc"]["head"] for fold in public_metrics.values()]),
        -layer))
    eligible = [layer for layer in LAYERS if np.mean([
        fold[f"gradient_{layer}"]["balanced_accuracy"] for fold in public_metrics.values()]) >= .65]
    if selected not in eligible or len(eligible) < 2:
        raise ValueError("Public prerequisite needs at least two eligible layers")
    scales = {}
    for layer in eligible:
        values = np.concatenate([margins(fold["scores"][f"gradient_{layer}"])
                                 for fold in public_scores.values()])
        scale = float(np.sqrt(np.mean(values ** 2)))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Public margin scale is degenerate")
        scales[str(layer)] = scale
    return {"layers": eligible, "single_layer": selected, "rms_scales": scales,
            "primary": "mean_standardized", "threshold": 0.,
            "selection": "Public mean BA >= .65; single best mean BA, then mean AUC, then shallower",
            "scale": "Public RMS of signed distance margin; no centering or target calibration",
            "vote_tie": "Fall back to public-selected single layer; exact distance tie predicts positive",
            "methods": ["single_selected", "single_32", "majority", "mean_raw", "mean_standardized"]}


def fuse(scores, rule):
    individual = {layer: margins(scores[f"gradient_{layer}"]) for layer in LAYERS}
    x = np.stack([individual[layer] for layer in rule["layers"]])
    standardized = np.stack([individual[layer] / rule["rms_scales"][str(layer)]
                             for layer in rule["layers"]])
    single = individual[rule["single_layer"]]
    votes = np.where(x >= 0, 1., -1.).sum(axis=0)
    votes = np.where(votes == 0, np.where(single >= 0, 1., -1.) * .5, votes)
    values = {"single_selected": single, "single_32": individual[32], "majority": votes,
              "mean_raw": x.mean(axis=0), "mean_standardized": standardized.mean(axis=0)}
    return {name: np.stack([-value, value], axis=1).tolist() for name, value in values.items()}


def complementarity(labels, scores, layers, classes):
    """Label-aware diagnostics; never used to select a layer or method."""
    y = np.asarray([classes.index(label) for label in labels])
    predictions = {layer: np.asarray(scores[f"gradient_{layer}"]).argmin(1) for layer in layers}
    correct = {layer: predictions[layer] == y for layer in layers}
    pairs = []
    for index, first in enumerate(layers):
        for second in layers[index + 1:]:
            pairs.append({"first": first, "second": second,
                          "prediction_disagreement": int((predictions[first] != predictions[second]).sum()),
                          "first_only_correct": int((correct[first] & ~correct[second]).sum()),
                          "second_only_correct": int((~correct[first] & correct[second]).sum()),
                          "both_wrong": int((~correct[first] & ~correct[second]).sum())})
    return {"pairs": pairs, "any_layer_correct": int(np.stack(list(correct.values())).any(0).sum()),
            "note": "Reference-aware upper-bound diagnostic only; no oracle selection"}


def historical(root):
    if (root / "historical/complete.json").exists():
        raise FileExistsError("Historical audit already committed")
    public = read_json(OLD / "styles/probe_predictions.json")
    metrics = read_json(OLD / "styles/development_report.json")["prototype_probes"]
    rule = freeze_rule(public, metrics)
    write_json(root / "head-rule.json", {**rule, "sources": {
        str(OLD / "styles/probe_predictions.json"): file_hash(OLD / "styles/probe_predictions.json"),
        str(OLD / "styles/development_report.json"): file_hash(OLD / "styles/development_report.json")},
        "source_sha256": file_hash(Path(__file__))})
    source = OLD / "deep-probe/predictions.json"
    rows = read_json(source)
    scores = {f"gradient_{layer}": [row["scores"][f"visual_{layer}"] for row in rows] for layer in LAYERS}
    predictions = fuse(scores, rule)
    path = root / "historical/predictions.json"
    write_json(path, {"scores": predictions, "source_sha256": file_hash(source),
                      "rule_sha256": file_hash(root / "head-rule.json"), "labels_read": False})
    write_json(root / "historical/predictions_complete.json", {"sha256": file_hash(path)})
    # Evaluation labels are loaded only after the frozen rule and scores exist.
    truth = read_json("outputs/diagnostics/synthetic_head_n20_gpu5_20261017/private/targets.json")
    labels = [row["label"] for row in truth]
    classes = ["head", "non-head"]
    stats = {name: {**classification_stats(labels, value, classes),
                    **paired_bootstrap(labels, value, classes)} for name, value in predictions.items()}
    gains = {name: paired_bootstrap(labels, value, classes, predictions["single_selected"])
             for name, value in predictions.items() if name != "single_selected"}
    results = {"scope": "Post-hoc audit on the same historical 20 medical images; no independent confirmation",
               "n": len(labels), "rule": rule, "metrics": stats, "paired_vs_single": gains,
               "complementarity": complementarity(labels, scores, rule["layers"], classes),
               "predictions_sha256": file_hash(path)}
    write_json(root / "historical/report.json", results)
    write_json(root / "historical/complete.json", {"sha256": file_hash(root / "historical/report.json")})
    print({name: value["correct"] for name, value in stats.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    historical(args.root)


if __name__ == "__main__":
    main()
