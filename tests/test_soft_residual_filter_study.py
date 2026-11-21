from dataclasses import replace

import pytest
import torch

from attacks.analytic.dager_subspace import SpanFilter, residual, row_basis
from core.config import DAGEROptions
from evaluation.soft_residual_filter_study import (
    ALPHAS, common_flags, evaluate_all, selected_ids, weighted_scores,
)


def matrices():
    generator = torch.Generator().manual_seed(725)
    public = torch.eye(16, dtype=torch.float64)[:4]
    features = torch.cat([public[:2], torch.randn(40, 16, generator=generator, dtype=torch.float64)])
    gradients = [torch.randn(3, 16, generator=generator, dtype=torch.float64) for _ in range(2)]
    options = DAGEROptions(rank_rtol=1e-10, rank_atol=1e-12, vocab_chunk_size=7)
    return public, features, gradients, options


def test_raw_and_hard_endpoints_match_original_span_scores():
    public, features, gradients, options = matrices()
    flags = common_flags(features, public, options.zero_tolerance)
    assert flags[:2].all() and not flags[2:].any()
    hard = SpanFilter.build(gradients, public, options)
    raw = SpanFilter.build(gradients, public, replace(options, mode="raw"))
    for alpha, span in [(0., raw), (1., hard)]:
        actual, _ = weighted_scores(gradients, features, public, alpha, options)
        expected, _ = span.score(features)
        mask = ~flags if alpha == 0 else torch.ones_like(flags)
        torch.testing.assert_close(actual[mask], expected[mask], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("alpha", [.5, .9])
def test_soft_scores_apply_same_transform_to_both_sides(alpha):
    public, features, gradients, options = matrices()
    q = torch.eye(16, dtype=torch.float64) - alpha * public.T @ public
    matrix = torch.cat([g / g.norm() for g in gradients]) @ q
    basis, _ = row_basis(matrix, options.rank_rtol, options.rank_atol)
    x = features @ q
    expected = residual(x, basis).norm(dim=-1) / x.norm(dim=-1)
    flags = common_flags(features, public, options.zero_tolerance)
    expected[flags] = 0
    actual, _ = weighted_scores(gradients, features, public, alpha, options)
    cached, _ = weighted_scores(gradients, features, public, alpha, options,
        projected_features=(features @ public.T) @ public, ambiguous=flags)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual, cached, rtol=0, atol=0)
    assert len(selected_ids(actual, flags, set(), k=10)) == 12


def test_ambiguity_budget_is_identical_for_every_alpha_and_ties_use_ids():
    public, features, gradients, options = matrices()
    flags = common_flags(features, public, options.zero_tolerance)
    for alpha in ALPHAS:
        scores, _ = weighted_scores(gradients, features, public, alpha, options)
        ids = selected_ids(scores, flags, {40}, k=10)
        assert {0, 1} <= ids and len(ids) == 12 and 40 not in ids
    assert selected_ids(torch.ones(5), torch.zeros(5, dtype=torch.bool), set(), 2) == {0, 1}


def test_evaluation_does_not_open_truth_before_all_scores_commit(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.soft_residual_filter_study.verify_plan", lambda *args: {"source": "/tmp/x"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Truth read before score completion barrier")
    monkeypatch.setattr("evaluation.soft_residual_filter_study.read_tensors", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
