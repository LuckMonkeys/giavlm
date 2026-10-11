import torch

from attacks.analytic.dager_filter_refinement import (
    adaptive_candidates, fuse_rankings, projected_system, sparse_support, spectral_scores,
)
from attacks.analytic.dager_subspace import SpanFilter
from core.config import DAGEROptions


def test_spectral_full_span_matches_original_with_public_projection():
    generator = torch.Generator().manual_seed(13)
    features = torch.randn(30, 20, generator=generator, dtype=torch.float64)
    public = features[:3]
    gradients = [torch.randn(4, 20, generator=generator, dtype=torch.float64) for _ in range(3)]
    options = DAGEROptions()
    projected, _, _, _, vectors, _, ambiguous = projected_system(
        gradients, public, features, options)
    original, expected_ambiguous = SpanFilter.build(gradients, public, options).score(features)
    actual = spectral_scores(projected, vectors)
    torch.testing.assert_close(ambiguous, expected_ambiguous)
    torch.testing.assert_close(actual[~ambiguous], original[~ambiguous], atol=1e-10, rtol=1e-9)


def test_signed_sparse_support_recovers_shared_tokens():
    features = torch.eye(12, dtype=torch.float64)
    matrix = torch.zeros(3, 12, dtype=torch.float64)
    matrix[:, 2] = torch.tensor([1., -2., 3.])
    matrix[:, 8] = torch.tensor([-3., 1., -1.])
    results = sparse_support(features, matrix, list(range(12)), lambda_ratios=(0.01,))
    norms, diagnostic = results[0.01]
    assert set(norms.topk(2).indices.tolist()) == {2, 8}
    assert diagnostic['converged']
    assert diagnostic['relative_squared_residual'] < 0.001


def test_gap_selection_retains_ambiguity_and_conservative_fallback():
    scores = torch.tensor([0., 0.01, 0.02, 0.03, 0.04, 0.8, 0.9, 1.])
    ambiguous = torch.tensor([True] + [False] * 7)
    selected, count = adaptive_candidates(scores, ambiguous, {7}, maximum=6)
    assert count == 4 and selected == [0, 1, 2, 3, 4]
    selected, count = adaptive_candidates(torch.arange(8) / 10, ambiguous, {7}, maximum=6)
    assert count == 6 and selected == list(range(7))


def test_rank_fusion_is_invariant_to_positive_score_rescaling():
    left = torch.tensor([0.1, 0.3, 0.2])
    right = torch.tensor([0.4, 0.1, 0.2])
    first = fuse_rankings([left, right], [0, 1, 2])
    second = fuse_rankings([left * 3, right * 10], [0, 1, 2])
    for key in first:
        torch.testing.assert_close(first[key], second[key])
