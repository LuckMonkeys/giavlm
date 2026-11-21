"""Public-only bank sampling, module selection and offset calibration."""

import numpy as np
import torch

from evaluation.image_property_scoring import binary_auc


METHODS = ("Q24", "K24", "V24", "Q32", "K32", "V32")
SIZES = (1, 2, 4, 8)


def build_banks(rows, labels, seed=20261105):
    """Sample entire public content families; keep their three styles together."""
    groups = {
        label: sorted({r["group"] for r in rows if labels[r["id"]] == label})
        for label in ("head", "non-head")
    }
    if any(len(v) != 8 for v in groups.values()) or len(rows) != 48:
        raise ValueError("Expected eight fit families per class and 48 images")
    if any(sum(r["group"] == g for r in rows) != 3 for v in groups.values() for g in v):
        raise ValueError("Every public family must contain all three styles")
    weights, metadata = [], []
    for count in SIZES:
        for repeat in range(1 if count == 8 else 20):
            rng = np.random.default_rng(seed + count * 1000 + repeat)
            chosen = [
                set(rng.choice(groups[c], count, replace=False).tolist())
                for c in ("head", "non-head")
            ]
            w = np.array(
                [[r["group"] in selected for r in rows] for selected in chosen], dtype=float
            )
            w /= w.sum(1, keepdims=True)
            weights.append(w)
            metadata.append(
                {
                    "id": f"real-{count:02d}-{repeat:02d}",
                    "kind": "real",
                    "families_per_class": count,
                    "images_per_class": count * 3,
                    "repeat": repeat,
                    "families": [sorted(v) for v in chosen],
                }
            )
    all_groups = sorted({r["group"] for r in rows})
    for repeat in range(20):
        rng = np.random.default_rng(seed + 100000 + repeat)
        positive = set(rng.choice(all_groups, 8, replace=False).tolist())
        chosen = [positive, set(all_groups) - positive]
        w = np.array([[r["group"] in selected for r in rows] for selected in chosen], dtype=float)
        w /= w.sum(1, keepdims=True)
        weights.append(w)
        metadata.append(
            {
                "id": f"null-{repeat:02d}",
                "kind": "null",
                "families_per_class": 8,
                "images_per_class": 24,
                "repeat": repeat,
                "families": [sorted(v) for v in chosen],
            }
        )
    return torch.tensor(np.stack(weights), dtype=torch.float64), metadata


def bank_geometry(features, weights):
    values = weights.flatten(0, 1) @ features.double()
    norm2 = values.square().sum(1)
    if not torch.isfinite(norm2).all() or (norm2 <= 1e-24).any():
        raise ValueError("Prototype mean is zero or nonfinite")
    return values / norm2.sqrt()[:, None], norm2


def projected_distances(projected_features, weights, norm2):
    """Exactly the original residual norm, using small cached projections."""
    projected_mean = weights.flatten(0, 1) @ projected_features.double()
    residual2 = 1 - projected_mean.square().sum(1) / norm2
    if not torch.isfinite(residual2).all() or float(residual2.min()) < -1e-10:
        raise ValueError("Invalid squared orthogonal residual")
    return residual2.clamp_min(0).sqrt().reshape(len(weights), 2)


def margin(scores):
    values = np.asarray(scores, dtype=float)
    if values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite positive/negative distances")
    return values[..., 1] - values[..., 0]


def balanced_accuracy(labels, predictions):
    y, prediction = np.asarray(labels, dtype=bool), np.asarray(predictions, dtype=bool)
    if not y.any() or y.all() or y.shape != prediction.shape:
        raise ValueError("Both public classes required")
    return float((prediction[y].mean() + (~prediction[~y]).mean()) / 2)


def public_quality(values, labels, styles, threshold=0.0):
    values, y, styles = np.asarray(values), np.asarray(labels, dtype=bool), np.asarray(styles)
    per_style = {
        style: balanced_accuracy(y[styles == style], values[styles == style] >= threshold)
        for style in sorted(set(styles))
    }
    return {
        "worst": min(per_style.values()),
        "mean": float(np.mean(list(per_style.values()))),
        "auroc": binary_auc(y, values),
        "per_style": per_style,
    }


def select_module(scores, labels, styles):
    """No medical or public-audit argument is accepted."""
    qualities = {m: public_quality(margin(scores[m]), labels, styles) for m in METHODS}
    selected = max(
        METHODS,
        key=lambda m: (
            qualities[m]["worst"],
            qualities[m]["mean"],
            qualities[m]["auroc"],
            -int(m[1:]),
            -("Q", "K", "V").index(m[0]),
        ),
    )
    return {
        "selected": selected,
        "qualities": qualities,
        "selection": "worst style BA, mean style BA, AUC, shallower, Q/K/V",
    }


def calibrate_offset(values, labels, styles):
    values = np.asarray(values, dtype=float)
    unique = np.unique(values)
    if not len(unique) or not np.isfinite(values).all():
        raise ValueError("Nonempty finite public calibration scores required")
    candidates = np.unique(
        np.r_[
            0.0,
            np.nextafter(unique[0], -np.inf),
            unique[:-1] + np.diff(unique) / 2,
            np.nextafter(unique[-1], np.inf),
        ]
    )
    candidates = [float(t) for t in candidates]
    qualities = {t: public_quality(values, labels, styles, t) for t in candidates}
    chosen = max(
        candidates, key=lambda t: (qualities[t]["worst"], qualities[t]["mean"], -abs(t), -t)
    )
    return {
        "threshold": chosen,
        "quality": qualities[chosen],
        "candidates": len(candidates),
        "selection": "worst style BA, mean style BA, nearest zero, smaller threshold",
        "score_direction": "Head when original distance margin >= threshold",
    }


def calibrated_scores(scores, threshold):
    centered = margin(scores) - threshold
    return np.stack([-centered, centered], axis=-1).tolist()


def stability_bounds(unit_prototypes, full_index):
    values = unit_prototypes.reshape(-1, 2, unit_prototypes.shape[-1])
    return (values - values[full_index]).norm(dim=-1).sum(1)
