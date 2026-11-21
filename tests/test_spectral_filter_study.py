import pytest
import torch

from attacks.analytic.dager_subspace import SpanFilter, residual
from evaluation.soft_residual_filter_study import selected_ids
from evaluation.spectral_filter_study import BETAS, beta_key, evaluate_all, spectral_scores
from tests.test_soft_residual_filter_study import matrices


def test_zero_beta_reproduces_original_scores_and_candidates():
    public, features, gradients, options = matrices()
    span = SpanFilter.build(gradients, public, options)
    expected, flags = span.score(features)
    scores, _ = spectral_scores(gradients, features, public, options)
    torch.testing.assert_close(scores[beta_key(0.)], expected, rtol=1e-12, atol=1e-12)
    assert selected_ids(scores[beta_key(0.)], flags, set(), 10) == selected_ids(expected, flags, set(), 10)


@pytest.mark.parametrize("beta", [.01, .1, 1.])
def test_weighted_distance_equals_direct_ridge_objective(beta):
    public, features, gradients, options = matrices()
    scores, diagnostic = spectral_scores(gradients, features, public, options)
    matrix = residual(torch.cat([g / g.norm() for g in gradients]), public)
    x = residual(features, public)
    lam = diagnostic["lambdas"][beta_key(beta)]
    gram = matrix @ matrix.T + lam * torch.eye(len(matrix), dtype=matrix.dtype)
    coefficients = torch.linalg.solve(gram, matrix @ x.T).T
    error = (x - coefficients @ matrix).square().sum(-1) + lam * coefficients.square().sum(-1)
    expected = error.sqrt() / x.norm(dim=-1).clamp_min(torch.finfo(x.dtype).tiny)
    expected[:2] = 0
    torch.testing.assert_close(scores[beta_key(beta)], expected, rtol=1e-10, atol=1e-12)


def test_weights_preserve_common_budget_scale_invariance_and_monotonicity():
    public, features, gradients, options = matrices()
    scores, _ = spectral_scores(gradients, features, public, options)
    cached, _ = spectral_scores([gradients[0] * 10, gradients[1] * .1], features, public, options,
                                projected_features=(features @ public.T) @ public)
    flags = features.new_zeros(len(features), dtype=torch.bool)
    flags[:2] = True
    for beta in BETAS:
        values = scores[beta_key(beta)]
        torch.testing.assert_close(values, cached[beta_key(beta)], rtol=1e-10, atol=1e-12)
        assert torch.all((values >= 0) & (values <= 1))
        assert len(selected_ids(values, flags, set(), 10)) == 12
    for previous, current in zip(BETAS, BETAS[1:]):
        assert torch.all(scores[beta_key(current)] >= scores[beta_key(previous)] - 1e-12)


def test_evaluation_requires_all_score_commits_before_truth(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.spectral_filter_study.verify_plan", lambda *args: {"source": "/tmp/x"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Private reference read before all-score barrier")
    monkeypatch.setattr("evaluation.spectral_filter_study.read_tensors", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
    assert not (tmp_path / "evaluation-barrier.json").exists()
