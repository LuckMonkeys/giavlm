"""Observable-only scores and fixed-budget candidates for question diagnostics."""
import numpy as np
import torch

from attacks.analytic.dager_subspace import residual


def energy(gradient, features, *, rowwise=False, chunk=512):
    if gradient.ndim != 2 or features.ndim != 2 or gradient.shape[1] != features.shape[1]:
        raise ValueError("Input feature coordinates differ from gradient")
    if not torch.isfinite(gradient).all() or gradient.norm() <= 0:
        raise ValueError("Zero or nonfinite gradient")
    g = gradient.double()
    if rowwise:
        norms = g.square().sum(1)
        valid = norms > 0
        g = g[valid] / norms[valid].sqrt()[:, None]
    else:
        g = g / g.norm()
    scores = features.new_empty(len(features), dtype=torch.float64)
    for start in range(0, len(features), chunk):
        x = features[start:start + chunk].double()
        norm = x.square().sum(1)
        value = (x @ g.T).square().sum(1) / norm.clamp_min(1e-30)
        scores[start:start + len(x)] = -value / (len(g) if rowwise else 1)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite energy score")
    return scores


def ridge_operator(public_columns, rho):
    """Return H on observed coordinates using the exact dual ridge identity."""
    d = public_columns.double()
    count = len(d)
    if count == 0 or rho <= 0:
        raise ValueError("Ridge needs observed coordinates and positive rho")
    scale = d.square().sum() / count
    if scale <= 0:
        return torch.eye(count, device=d.device, dtype=d.dtype), {
            "support": count, "rank": 0, "condition_bound": 1., "lambda": 0.}
    lam = rho * scale
    covariance = d @ d.T
    h = lam * torch.cholesky_inverse(torch.linalg.cholesky(
        covariance + lam * torch.eye(count, device=d.device, dtype=d.dtype)))
    rank = int(torch.linalg.matrix_rank(d, rtol=1e-7, atol=1e-10))
    return h, {"support": count, "rank": rank, "condition_bound": float(1 + count / rho),
               "lambda": float(lam)}


def support_energy(gradient, features, public_basis, *, rho=None, projected=None, chunk=1024):
    """Per-row matching feature/update transforms; no zero-filled span claim."""
    g = gradient.double()
    result = features.new_zeros(len(features), dtype=torch.float64)
    diagnostics = []
    projected = residual(features, public_basis) if projected is None and rho is None else projected
    gr = residual(g, public_basis) if rho is None else None
    valid = 0
    for index, row in enumerate(g):
        support = torch.nonzero(row != 0).flatten()
        if not len(support):
            diagnostics.append({"support": 0, "valid": False})
            continue
        if rho is None:
            r = gr[index, support]
            h, info = None, {"support": len(support), "rank": None,
                             "condition_bound": None, "lambda": None}
        else:
            h, info = ridge_operator(public_basis[:, support].T, rho)
            r = h @ row[support]
        rnorm = r.square().sum()
        if rnorm <= 1e-30:
            diagnostics.append(dict(info, valid=False))
            continue
        valid += 1
        for start in range(0, len(features), chunk):
            x = (projected if rho is None else features)[start:start + chunk, support]
            z = x if h is None else x @ h.T
            result[start:start + len(x)] -= ((z @ r).square() /
                (rnorm * z.square().sum(1)).clamp_min(1e-30))
        diagnostics.append(dict(info, valid=True, residual_norm=float(rnorm.sqrt())))
    if not valid or not torch.isfinite(result).all():
        raise ValueError("No valid support rows or nonfinite score")
    return result / valid, diagnostics


def rank(scores, flags, forbidden):
    values = scores.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ranking")
    valid = ~flags.cpu().numpy().copy()
    valid[list(forbidden)] = False
    ids = np.flatnonzero(valid)
    return ids[np.lexsort((ids, values[ids]))].tolist()


def blend(orders, quotas, ambiguous, total=100):
    if sum(quotas) != total:
        raise ValueError("Quotas must preserve informative budget")
    chosen = set()
    for order, quota in zip(orders, quotas, strict=True):
        before = len(chosen)
        for token in order:
            if token not in chosen:
                chosen.add(int(token))
            if len(chosen) == before + quota:
                break
        if len(chosen) != before + quota:
            raise ValueError("Insufficient nonduplicate candidates")
    if chosen & ambiguous:
        raise ValueError("Informative candidates contain ambiguous IDs")
    return chosen | ambiguous


def rank_consensus(first, second, flags, forbidden):
    ranks = []
    for scores in (first, second):
        order = rank(scores, flags, forbidden)
        value = torch.full_like(scores, float(len(scores)))
        value[order] = torch.arange(len(order), dtype=value.dtype)
        ranks.append(value / len(order))
    return sum(ranks) / 2


def metrics(chosen, questions, answers):
    q, a = set(questions), set(answers)
    truths = {"question": q, "answer": a, "joint": q | a, "exclusive": q - a}
    result = {"selected_count": len(chosen)}
    for name, truth in truths.items():
        result[name + "_recall"] = len(chosen & truth) / len(truth) if truth else None
        result[name + "_full_coverage"] = truth <= chosen if truth else None
    return result
