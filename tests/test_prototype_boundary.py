import numpy as np
import pytest
import torch

from evaluation.prototype_boundary_math import decompose, fit_rule, geometry


def test_projection_scale_changes_margin_but_not_relative_class_direction():
    energy = np.array([[0.8, 0.4], [0.4, 0.8], [0.08, 0.04], [0.04, 0.08]])
    z = decompose(np.sqrt(1 - energy))
    np.testing.assert_allclose(z["relative"][:2], z["relative"][2:])
    assert abs(z["margin"][0]) > abs(z["margin"][2])
    assert np.array_equal(z["margin"] >= 0, z["relative"] >= 0)


def test_zero_projection_is_unscorable_and_public_fit_fails_fast():
    z = decompose([[1.0, 1.0], [0.7, 0.9]])
    assert z["valid"].tolist() == [False, True]
    with pytest.raises(ValueError, match="unscorable"):
        fit_rule(z["relative"], [True, False], ["fixed", "fixed"], z["valid"])


def test_common_contrast_coupling_and_centering_degeneracy():
    g = torch.Generator().manual_seed(10)
    mu = torch.nn.functional.normalize(torch.randn(2, 8, generator=g, dtype=torch.float64), dim=1)
    q, _ = torch.linalg.qr(torch.randn(8, 3, generator=g, dtype=torch.float64))
    stats = geometry(mu, q.T)
    e = (mu @ q).square().sum(1)
    assert abs(stats["four_cross"] - float(e[0] - e[1])) < 1e-12
    centered = mu - mu.mean(0)
    d = (centered - (centered @ q) @ q.T).norm(dim=1)
    torch.testing.assert_close(d[0], d[1])


def test_public_relative_threshold_uses_only_supplied_public_examples():
    values = [0.6, 0.8, 0.4, 0.5]
    fit = fit_rule(values, [True, True, False, False], ["fixed"] * 4, [True] * 4)
    assert fit["threshold"] == 0.55
