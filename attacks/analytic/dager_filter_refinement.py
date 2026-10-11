"""Experimental first-layer scores using public observations only.

These functions neither load private references nor change the registered attack.
Lower scores are better, including the negative sparse coefficient norms.
"""
import torch

from attacks.analytic.dager_subspace import residual, row_basis


def projected_system(gradients, public_inputs, features, options):
    basis, _ = row_basis(public_inputs.double(), options.rank_rtol, options.rank_atol)
    projected = residual(features.double(), basis)
    converted = [g.to(device=basis.device, dtype=torch.float64) for g in gradients]
    blocks = [residual(g / g.norm().clamp_min(1e-30), basis) for g in converted]
    matrix = torch.cat(blocks)
    _, singular, vectors = torch.linalg.svd(matrix, full_matrices=False)
    keep = singular > max(options.rank_atol, options.rank_rtol * float(singular[0]))
    norms = projected.norm(dim=1)
    ratio = norms / features.double().norm(dim=1).clamp_min(1e-30)
    ambiguous = ratio <= options.zero_tolerance
    return projected, blocks, matrix, singular[keep], vectors[keep], ratio, ambiguous


def spectral_scores(projected, vectors, weights=None):
    """Distance to a hard span, or residual after a soft spectral projection."""
    norm2 = projected.square().sum(1).clamp_min(1e-30)
    coefficients2 = (projected @ vectors.T).square()
    if weights is not None:
        coefficients2 = coefficients2 * (2 * weights - weights.square())
    return (1 - coefficients2.sum(1) / norm2).clamp(0, 1).sqrt()


def rank_vector(scores, allowed):
    indices = torch.tensor(allowed, device=scores.device, dtype=torch.long)
    order = torch.argsort(scores[indices], stable=True)
    result = torch.full_like(scores, float(len(scores) + 1), dtype=torch.float64)
    result[indices[order]] = torch.arange(1, len(indices) + 1, device=scores.device,
                                        dtype=torch.float64)
    return result


def fuse_rankings(score_vectors, allowed):
    ranks = torch.stack([rank_vector(s, allowed) for s in score_vectors])
    return {"mean_rank": ranks.mean(0), "median_rank": ranks.median(0).values,
            "best_rank": ranks.min(0).values,
            "rrf": -(1 / (60 + ranks)).mean(0)}


def sparse_support(projected, matrix, pool, lambda_ratios=(0.005, 0.02, 0.1),
                   iterations=400, tolerance=1e-6, analysis_dtype=torch.float32):
    """Signed multi-measurement group lasso on a public, preselected dictionary.

    The coefficient matrix is (gradient rows, candidate tokens). Columns share
    support. FISTA step sizes use the exact dictionary Gram spectral norm.
    No truth-based early stopping or candidate supplementation is performed.
    """
    dictionary = projected[pool].to(analysis_dtype)
    dictionary = dictionary / dictionary.norm(dim=1, keepdim=True).clamp_min(1e-20)
    target = matrix.to(analysis_dtype) / matrix.to(analysis_dtype).norm().clamp_min(1e-20)
    gram = dictionary @ dictionary.T
    cross = target @ dictionary.T
    # Symmetric eigvalsh avoids an underestimated power-iteration Lipschitz bound.
    lipschitz = torch.linalg.eigvalsh(gram.double())[-1].to(analysis_dtype).clamp_min(1e-12)
    lambda_max = cross.norm(dim=0).max()
    results = {}
    for ratio in lambda_ratios:
        penalty = lambda_max * ratio
        current = torch.zeros_like(cross)
        extrapolated = current.clone()
        momentum = 1.0
        change = float("inf")
        for step in range(iterations):
            tentative = extrapolated - (extrapolated @ gram - cross) / lipschitz
            norms = tentative.norm(dim=0).clamp_min(1e-20)
            updated = tentative * (1 - penalty / (lipschitz * norms)).clamp_min(0)
            change = float((updated - current).norm() / updated.norm().clamp_min(1e-20))
            next_momentum = (1 + (1 + 4 * momentum * momentum) ** 0.5) / 2
            extrapolated = updated + ((momentum - 1) / next_momentum) * (updated - current)
            current = updated
            momentum = next_momentum
            if change < tolerance:
                break
        reconstruction = current @ dictionary
        error = float((reconstruction - target).square().sum())
        norms = current.norm(dim=0)
        gradient = current @ gram - cross
        active = norms > 1e-10
        kkt = (gradient.norm(dim=0) - penalty).clamp_min(0)
        kkt[active] = (gradient[:, active] + penalty * current[:, active]
                       / norms[active]).norm(dim=0)
        kkt_relative = float(kkt.max() / lambda_max.clamp_min(1e-20))
        results[ratio] = (current.norm(dim=0).double(), {
            "steps": step + 1, "relative_change": change,
            "converged": change < tolerance and kkt_relative < 10 * tolerance,
            "kkt_relative": kkt_relative,
            "relative_squared_residual": error,
            "objective": error / 2 + float(penalty * current.norm(dim=0).sum()),
            "active_candidates": int((current.norm(dim=0) > 1e-7).sum()),
            "lambda_ratio": ratio})
    return results


def adaptive_candidates(scores, ambiguous, forbidden, minimum=4, maximum=100,
                        minimum_gap=0.15, minimum_ratio=2.0):
    """Predeclared gap rule; otherwise retain the full conservative maximum."""
    informative = [i for i in range(len(scores)) if i not in forbidden and not ambiguous[i]]
    ranked = sorted(informative, key=lambda i: (float(scores[i]), i))
    limit = min(maximum, len(ranked))
    gaps = []
    for k in range(minimum, limit):
        left, right = float(scores[ranked[k - 1]]), float(scores[ranked[k]])
        if right - left >= minimum_gap and right / max(left, 1e-12) >= minimum_ratio:
            gaps.append((right - left, k))
    cutoff = max(gaps)[1] if gaps else limit
    uncertain = [i for i in range(len(scores)) if i not in forbidden and ambiguous[i]]
    return sorted(ranked[:cutoff] + uncertain), cutoff
