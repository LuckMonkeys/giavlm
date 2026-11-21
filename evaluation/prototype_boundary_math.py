"""Frozen score decomposition for the prototype boundary diagnosis."""

import numpy as np

from evaluation.prototype_method_math import calibrate_offset


def decompose(distances):
    d = np.asarray(distances, dtype=float)
    if d.shape[-1] != 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Expected finite positive/negative residual distances")
    energy = 1 - d**2
    if energy.min() < -1e-12:
        raise ValueError("Unit prototype has invalid projection energy")
    energy = np.maximum(energy, 0)
    delta = energy[..., 0] - energy[..., 1]
    total = energy.sum(-1)
    valid = total > 1e-12
    relative = np.divide(delta, total, out=np.zeros_like(delta), where=valid)
    margin = d[..., 1] - d[..., 0]
    if not np.array_equal(relative[valid] >= 0, margin[valid] >= 0):
        raise ValueError("Relative energy changed the zero-threshold decision")
    return dict(margin=margin, coverage=total / 2, delta=delta, relative=relative, valid=valid)


def fit_rule(values, labels, styles, valid):
    if not np.asarray(valid).all():
        raise ValueError("Public calibration contains unscorable projections")
    return calibrate_offset(values, labels, styles)


def geometry(prototypes, basis):
    mu = prototypes.double()
    p, v = (mu[0] + mu[1]) / 2, (mu[0] - mu[1]) / 2
    pp, pv = p @ basis.T, v @ basis.T
    energies = (mu @ basis.T).square().sum(1)
    cross = 4 * float(pp @ pv)
    if abs(float(energies[0] - energies[1]) - cross) > 1e-12:
        raise ValueError("Common/contrast identity failed")
    return {
        "common_energy": float(pp.square().sum()),
        "contrast_energy": float(pv.square().sum()),
        "four_cross": cross,
        "prototype_cosine": float(mu[0] @ mu[1]),
    }


def scalar_stats(values):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError("Nonempty finite diagnostic values required")
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "q25": float(np.quantile(a, 0.25)),
        "q75": float(np.quantile(a, 0.75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired_interval(values, seed=20261105):
    a = np.asarray(values, dtype=float)
    if not len(a):
        raise ValueError("Empty paired difference")
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (2000, len(a)))].mean(1)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "n": len(a),
    }
