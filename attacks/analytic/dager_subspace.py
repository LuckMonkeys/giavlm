"""Independent implementation of DAGER span checks (Petrov et al., NeurIPS 2024).

The public-input quotient and normalized Q/K/V stacking are VLM adaptations.
Rows follow PyTorch Linear weight orientation; no embedding gradients are used.
"""
from dataclasses import dataclass

import torch


def row_basis(matrix, rtol, atol):
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("Span decomposition requires a finite matrix")
    if matrix.numel() == 0:
        return matrix.new_empty((0, matrix.shape[1])), matrix.new_empty(0)
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    threshold = max(atol, rtol * singular[0].item())
    return vh[singular > threshold], singular


def residual(vectors, basis):
    return vectors - (vectors @ basis.T) @ basis


@dataclass
class SpanFilter:
    basis: torch.Tensor
    public_basis: torch.Tensor
    zero_tolerance: float
    diagnostics: dict

    @classmethod
    def build(cls, gradients, public_inputs, options):
        dtype = getattr(torch, options.analysis_dtype)
        device = public_inputs.device
        public = public_inputs.to(dtype)
        if options.mode == "raw":
            public = public[:0]
        public_basis, _ = row_basis(public, options.rank_rtol, options.rank_atol)
        blocks = []
        norms = []
        for gradient in gradients:
            value = gradient.detach().to(device=device, dtype=dtype)
            if value.ndim != 2 or value.shape[1] != public_inputs.shape[1]:
                raise ValueError("Gradient does not have the expected input dimension")
            norm = value.norm().item()
            norms.append(norm)
            # Normalize before removing public directions: do not amplify tiny
            # roundoff residuals of a wholly public gradient block.
            if norm > 0:
                blocks.append(residual(value / norm, public_basis))
        matrix = (torch.cat(blocks) if blocks else public.new_empty((0, public.shape[1])))
        basis, singular = row_basis(matrix, options.rank_rtol, options.rank_atol)
        diagnostics = {"rank": len(basis), "public_rank": len(public_basis),
                       "input_dimension": public.shape[1], "block_norms": norms,
                       "quotient_dimension": public.shape[1] - len(public_basis),
                       "saturated": len(basis) >= public.shape[1] - len(public_basis),
                       "singular_values": singular.cpu().tolist(),
                       "rank_rtol": options.rank_rtol, "rank_atol": options.rank_atol}
        return cls(basis, public_basis, options.zero_tolerance, diagnostics)

    def score(self, vectors):
        vectors = vectors.to(device=self.basis.device, dtype=self.basis.dtype)
        if not torch.isfinite(vectors).all():
            raise ValueError("Nonfinite candidate features in DAGER span check")
        projected = residual(vectors, self.public_basis)
        norm = projected.norm(dim=-1)
        ambiguous = norm <= self.zero_tolerance * vectors.norm(dim=-1).clamp_min(
            torch.finfo(vectors.dtype).tiny)
        distance = residual(projected, self.basis).norm(dim=-1) / norm.clamp_min(
            torch.finfo(vectors.dtype).tiny)
        # Zero quotient vectors provide no membership evidence. The caller keeps
        # their IDs explicitly, and records ambiguity rather than a positive hit.
        distance = torch.where(ambiguous, torch.zeros_like(distance), distance).clamp(0, 1)
        return distance, ambiguous


def select_tokens(scores, ambiguous, forbidden, options):
    allowed = [i for i in range(len(scores)) if i not in forbidden]
    uncertain = [i for i in allowed if bool(ambiguous[i])]
    informative = [i for i in allowed if not bool(ambiguous[i])]
    ranked = sorted(informative, key=lambda i: (float(scores[i]), i))
    if options.token_selection == "threshold":
        ranked = [i for i in ranked if float(scores[i]) <= options.token_threshold]
    selected = ranked[:options.max_candidates]
    # The cap applies to informative candidates. Public-overlap tokens are never
    # silently discarded by that cap; search still has an explicit beam budget.
    return sorted(selected + uncertain), {
        "informative_candidates": len(selected), "ambiguous_candidates": len(uncertain),
        "candidates_before_cap": len(ranked), "truncated": len(ranked) > len(selected)}
