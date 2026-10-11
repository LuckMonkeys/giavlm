import torch

from attacks.analytic.dager_visual_filter import visual_weight


def test_visual_metric_equals_eliminated_ridge_nuisance():
    rng = torch.Generator().manual_seed(25)
    basis = torch.linalg.qr(torch.randn(9, 3, generator=rng, dtype=torch.float64))[0].T
    eigenvalues = torch.tensor([3., 1., .2], dtype=torch.float64)
    error = torch.randn(4, 9, generator=rng, dtype=torch.float64)
    ratio = .03
    ridge = ratio * eigenvalues[0]
    dictionary = eigenvalues.sqrt()[:, None] * basis
    coeff = error @ dictionary.T @ torch.linalg.inv(
        dictionary @ dictionary.T + ridge * torch.eye(3, dtype=torch.float64))
    minimized = (error - coeff @ dictionary).square().sum() + ridge * coeff.square().sum()
    weighted = visual_weight(error, basis, eigenvalues, ratio).square().sum()
    torch.testing.assert_close(weighted, minimized)


def test_visual_weight_preserves_orthogonal_text_and_suppresses_visual_signal():
    basis = torch.eye(5, dtype=torch.float64)[:2]
    eigenvalues = torch.tensor([2., 1.], dtype=torch.float64)
    features = torch.eye(5, dtype=torch.float64)
    result = visual_weight(features, basis, eigenvalues, .001)
    torch.testing.assert_close(result[2:], features[2:])
    assert result[:2].norm() < features[:2].norm() / 20
    hard = visual_weight(features, basis, eigenvalues, 0)
    torch.testing.assert_close(hard[:2], torch.zeros_like(hard[:2]))
