import pytest
import torch

from attacks.analytic.dager_subspace import row_basis
from evaluation.question_filter_evaluate import evaluate, first_metrics
from evaluation.question_filter_math import (blend, energy, metrics, rank, ridge_operator,
                                             support_energy)


def random_system():
    rng = torch.Generator().manual_seed(115)
    g = torch.randn(3, 9, generator=rng, dtype=torch.float64)
    x = torch.randn(13, 9, generator=rng, dtype=torch.float64)
    u, _ = row_basis(torch.randn(2, 9, generator=rng, dtype=torch.float64), 1e-12, 0)
    return g, x, u


def test_energy_matches_direct_cosines_and_preserves_scaling():
    g, x, _ = random_system()
    expected = -torch.stack([sum(torch.dot(row, w)**2 for row in g) /
                             (w.square().sum() * g.square().sum()) for w in x])
    torch.testing.assert_close(energy(g, x, chunk=4), expected)
    torch.testing.assert_close(energy(g * 3, x * 7), expected)
    row_expected = -torch.stack([torch.stack([torch.dot(row, w)**2 /
        (row.square().sum() * w.square().sum()) for row in g]).mean() for w in x])
    torch.testing.assert_close(energy(g, x, rowwise=True), row_expected)


def test_ridge_dual_matches_independent_primal_fit():
    g, x, u = random_system()
    g[:, [0, 2, 5, 8]] = 0
    rho = .1
    actual, diagnostics = support_energy(g, x, u, rho=rho, chunk=3)
    expected = torch.zeros(len(x), dtype=torch.float64)
    for row in g:
        ids = torch.nonzero(row).flatten()
        d = u[:, ids].T
        lam = rho * d.square().sum() / len(ids)
        normal = d.T @ d + lam * torch.eye(len(u), dtype=torch.float64)
        r = row[ids] - d @ torch.linalg.solve(normal, d.T @ row[ids])
        for j, value in enumerate(x):
            z = value[ids] - d @ torch.linalg.solve(normal, d.T @ value[ids])
            expected[j] -= torch.dot(r, z)**2 / (r.square().sum() * z.square().sum()) / len(g)
    torch.testing.assert_close(actual, expected)
    assert all(r['valid'] and r['lambda'] > 0 for r in diagnostics)


def test_postmask_applies_identical_transform_to_candidate_and_gradient():
    g, x, u = random_system()
    g[:, [0, 2, 5, 8]] = 0
    actual, _ = support_energy(g, x, u)
    q = torch.eye(9, dtype=torch.float64) - u.T @ u
    expected = torch.zeros(len(x), dtype=torch.float64)
    for row in g:
        ids = torch.nonzero(row).flatten()
        r = (q @ row)[ids]
        z = (x @ q)[:, ids]
        expected -= (z @ r).square() / (r.square().sum() * z.square().sum(1)) / len(g)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose((g @ q) * (g != 0), g @ q)


def test_unregularized_public_fit_can_absorb_all_observed_private_signal():
    d = torch.eye(4, dtype=torch.float64)
    observed = torch.tensor([1., 2., -1., 5.], dtype=torch.float64)
    assert torch.equal(observed - d @ torch.linalg.lstsq(d, observed).solution,
                       torch.zeros_like(observed))
    h, _ = ridge_operator(d, .1)
    assert (h @ observed).norm() > 0


def test_fixed_budgets_and_bridge_controls_share_the_same_initial_evidence():
    flags = torch.zeros(250, dtype=torch.bool)
    flags[:9] = True
    a = rank(torch.arange(250, dtype=torch.float64), flags, {249})
    b = rank(-torch.arange(250, dtype=torch.float64), flags, {249})
    chosen = blend([a, b, a], [60, 20, 20], set(range(9)))
    other = blend([a, b, b], [60, 20, 20], set(range(9)))
    assert len(chosen) == len(other) == 109
    assert set(a[:60]) | set(b[:20]) <= chosen & other
    with pytest.raises(ValueError):
        blend([a, b], [80, 19], set(range(9)))


def test_exclusive_metrics_do_not_count_shared_answer_tokens():
    row = metrics({1, 2, 3}, {1, 2, 4}, {1, 3})
    assert row['question_recall'] == 2/3
    assert row['exclusive_recall'] == .5
    assert row['answer_full_coverage'] and not row['joint_full_coverage']
    assert metrics({1}, {1}, {1})['exclusive_recall'] is None


def test_first_word_proposal_misses_are_not_conditioned_away():
    row = first_metrics([1, 2], [1, 2, 3], 8, set())
    assert not row['pool_hit'] and not row['hit_20']


def test_target_evaluation_refuses_missing_commits_before_reference_read(tmp_path, monkeypatch):
    monkeypatch.setattr('evaluation.question_filter_evaluate.verify_plan', lambda *a: {})
    def forbidden(*args, **kwargs):
        raise AssertionError('Private reference read before commitment barrier')
    monkeypatch.setattr('evaluation.question_filter_evaluate.read_tensors', forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path)
    assert not (tmp_path / 'evaluation-barrier.json').exists()


def test_positionwise_mlp_still_depends_on_preceding_attention():
    token = torch.tensor([.1, -.4], dtype=torch.float64)
    first = torch.stack([torch.tensor([1., 0.]), token]).double()
    second = torch.stack([torch.tensor([-1., 0.]), token]).double()
    def mlp_input(values):
        weights = torch.softmax(values[-1] @ values.T, dim=0)
        after = values[-1] + weights @ values
        return torch.nn.functional.silu(after) * after
    assert not torch.allclose(mlp_input(first), mlp_input(second))
