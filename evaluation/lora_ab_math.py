"""Observable A/B consistency and residual-left projections; no references."""
import torch

from attacks.analytic.dager_subspace import residual, row_basis


METHODS = ("baseline", "ab_consistency", "residual_left", "residual_left_gated")
NULL_SAMPLES = 64
NULL_QUANTILE = .95


def relative_difference(left, right):
    denominator = .5 * (left.norm() + right.norm())
    return float((left - right).norm() / denominator.clamp_min(torch.finfo(left.dtype).tiny))


def consistency_projection(ga, gb, a, b):
    """Weighted orthogonal projection onto X A^T = B^T Z.

    Variances are observable max-absolute-value squared proxies, not clean
    gradient scales. A common noise ratio cancels. Solve the r-by-r Sylvester
    equation by two eigendecompositions; never treat observed G_B as exact.
    """
    va, vb = ga.abs().max().square(), gb.abs().max().square()
    if min(float(va), float(vb)) <= 0:
        raise ValueError("A/B consistency requires two nonzero observed blocks")
    common = torch.maximum(va, vb)
    va, vb = va / common, vb / common
    ea, ua = torch.linalg.eigh(a @ a.T)
    eb, ub = torch.linalg.eigh(b.T @ b)
    if float(ea.min()) <= 0 or float(eb.min()) < -1e-12 * float(eb.max()):
        raise ValueError("Invalid model Gram matrix")
    denominator = vb * eb.clamp_min(0)[:, None] + va * ea[None, :]
    defect = ga @ a.T - b.T @ gb
    lagrange = ub @ ((ub.T @ defect @ ua) / denominator) @ ua.T
    x, z = ga - va * lagrange @ a, gb + vb * b @ lagrange
    error = relative_difference(x @ a.T, b.T @ z)
    if not torch.isfinite(x).all() or not torch.isfinite(z).all() or error > 1e-8:
        raise ValueError(f"A/B projection failed to satisfy its constraint: {error}")
    return x, z, error


def psd_root(matrix):
    values, vectors = torch.linalg.eigh(matrix)
    if float(values.min()) < -1e-10 * float(values.abs().max()):
        raise ValueError("Noise covariance is not positive semidefinite")
    return (vectors * values.clamp_min(0).sqrt()) @ vectors.T


def null_threshold(ga, gb, a_public, b, noise_ratio, seed):
    if noise_ratio == 0:
        return 0.
    if noise_ratio < 0:
        raise ValueError("Negative public noise ratio")
    rank = len(ga)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    za = torch.randn(NULL_SAMPLES, rank, rank, generator=generator, dtype=torch.float64).to(ga)
    zb = torch.randn(NULL_SAMPLES, rank, rank, generator=generator, dtype=torch.float64).to(ga)
    sigma_a, sigma_b = noise_ratio * ga.abs().max(), noise_ratio * gb.abs().max()
    # B^T N_B has row covariance sigma_B^2 B^T B. N_A P A^T
    # has column covariance sigma_A^2 A P A^T, independently.
    samples = (sigma_b * psd_root(b.T @ b) @ zb -
               sigma_a * za @ psd_root(a_public @ a_public.T).T)
    maxima = torch.linalg.svdvals(samples)[:, 0]
    return float(torch.quantile(maxima, NULL_QUANTILE))


def left_project(value, vectors, rank):
    if not 0 <= rank <= len(value):
        raise ValueError("Invalid left rank")
    # Both fallbacks are declared, exact identity operations. In particular,
    # a zero signal estimate must not erase the observed gradient silently.
    if rank in (0, len(value)):
        return value, torch.eye(len(value), device=value.device, dtype=value.dtype)
    projection = vectors[:, :rank] @ vectors[:, :rank].T
    return projection @ value, projection


@torch.no_grad()
def module_blocks(ga, gb, a, b, public_basis, private_length, noise_ratio, seed):
    if ga.shape != a.shape or gb.shape != b.shape or ga.shape[0] != gb.shape[1]:
        raise ValueError("Uploaded A/B gradient shapes differ from public weights")
    norm = ga.norm()
    if float(norm) == 0:
        raise ValueError("Zero observed A gradient")
    raw = residual(ga / norm, public_basis)
    x, _, correction_error = consistency_projection(ga, gb, a, b)
    joint = residual(x / norm, public_basis)
    a_public, ga_public = a @ public_basis.T, ga @ public_basis.T
    sketch = b.T @ gb - ga_public @ a_public.T
    left, singular, _ = torch.linalg.svd(sketch, full_matrices=False)
    numeric_rank = int((singular > singular[0] * 1e-10).sum()) if float(singular[0]) > 0 else 0
    length_rank = min(private_length, len(ga), numeric_rank)
    threshold = null_threshold(ga, gb, a_public, b, noise_ratio, seed)
    gate_rank = min(length_rank, int((singular > threshold).sum())) if noise_ratio else length_rank
    low, low_projection = left_project(raw, left, length_rank)
    gated, gated_projection = left_project(raw, left, gate_rank)
    raw_product, other_product = ga @ a.T, b.T @ gb
    reference_sketch = residual(ga, public_basis) @ a.T
    raw_singular = torch.linalg.svdvals(ga)
    residual_singular = torch.linalg.svdvals(raw)
    diagnostic = {
        "raw_identity_relative_error": relative_difference(raw_product, other_product),
        "residual_identity_relative_error": relative_difference(sketch, reference_sketch),
        "corrected_identity_relative_error": correction_error,
        "relative_a_correction": float((x - ga).norm() / norm),
        "raw_singular_values": raw_singular.cpu().tolist(),
        "residual_singular_values": residual_singular.cpu().tolist(),
        "sketch_singular_values": singular.cpu().tolist(),
        "raw_numerical_rank": int((raw_singular > raw_singular[0] * 1e-10).sum()),
        "residual_numerical_rank": int((residual_singular > residual_singular[0] * 1e-10).sum()),
        "sketch_numerical_rank": numeric_rank,
        "length_rank": length_rank, "gate_rank": gate_rank, "null_threshold": threshold,
        "length_fallback": length_rank in (0, len(ga)), "gate_fallback": gate_rank in (0, len(ga)),
        "sigma_a_proxy": float(noise_ratio * ga.abs().max()),
        "sigma_b_proxy": float(noise_ratio * gb.abs().max()),
        "observed_a_max": float(ga.abs().max()), "observed_b_max": float(gb.abs().max()),
        "a_norm": float(norm), "private_length_cap": private_length,
    }
    artifacts = {"raw_residual": raw * norm, "joint_residual": joint * norm,
                 "left_projection": low_projection, "gate_projection": gated_projection,
                 "sketch": sketch}
    return dict(zip(METHODS, (raw, joint, low, gated), strict=True)), diagnostic, artifacts


@torch.no_grad()
def score_blocks(blocks, projected_features, ambiguous, options):
    basis, singular = row_basis(torch.cat(blocks), options.rank_rtol, options.rank_atol)
    scores = projected_features.new_empty(len(projected_features))
    for start in range(0, len(scores), options.vocab_chunk_size):
        x = projected_features[start:start + options.vocab_chunk_size]
        norms = x.norm(dim=-1).clamp_min(torch.finfo(x.dtype).tiny)
        scores[start:start + len(x)] = (residual(x, basis).norm(dim=-1) / norms).clamp(0, 1)
    scores = torch.where(ambiguous, torch.zeros_like(scores), scores)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite candidate score")
    return scores, {"rank": len(basis), "singular_values": singular.cpu().tolist()}
