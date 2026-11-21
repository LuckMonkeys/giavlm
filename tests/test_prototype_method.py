import numpy as np
import torch

from attacks.analytic.dager_subspace import row_basis
from evaluation.prototype_method_math import (
    METHODS,
    bank_geometry,
    build_banks,
    calibrate_offset,
    calibrated_scores,
    margin,
    projected_distances,
    select_module,
    stability_bounds,
)
from evaluation.image_property_scoring import binary_auc


def test_family_sampling_keeps_styles_and_counts_full_bank_only_once():
    rows = [
        {"id": f"{c}-{i}-{s}", "group": f"{c}-{i:02d}", "style": s}
        for c in ("head", "non-head")
        for i in range(8)
        for s in ("line", "photo", "render")
    ]
    labels = {r["id"]: r["group"].rsplit("-", 1)[0] for r in rows}
    weights, metadata = build_banks(rows, labels)
    assert weights.shape == (81, 2, 48)
    assert len([r for r in metadata if r["kind"] == "real" and r["families_per_class"] == 8]) == 1
    assert torch.allclose(weights.sum(-1), torch.ones(81, 2, dtype=torch.float64))
    for bank, w in zip(metadata, weights, strict=True):
        assert torch.all((w > 0).sum(1) == bank["images_per_class"])
        for group in sorted({r["group"] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r["group"] == group]
            assert torch.equal(w[:, ids[0]], w[:, ids[1]]) and torch.equal(
                w[:, ids[0]], w[:, ids[2]]
            )


def test_small_projection_cache_matches_direct_residual_and_stability_inequality():
    rng = torch.Generator().manual_seed(2)
    features = torch.randn(48, 9, generator=rng, dtype=torch.float64)
    weights = torch.zeros(2, 2, 48, dtype=torch.float64)
    weights[0, 0, :24] = 1 / 24
    weights[0, 1, 24:] = 1 / 24
    weights[1, 0, :3] = 1 / 3
    weights[1, 1, 24:27] = 1 / 3
    units, norm2 = bank_geometry(features, weights)
    basis, _ = row_basis(torch.randn(3, 9, generator=rng, dtype=torch.float64), 1e-6, 1e-10)
    cached = projected_distances(features @ basis.T, weights, norm2)
    direct = (units - (units @ basis.T) @ basis).norm(dim=1).reshape(2, 2)
    torch.testing.assert_close(cached, direct, atol=1e-14, rtol=1e-14)
    eps = stability_bounds(units, 0)
    actual = torch.tensor(np.abs(margin(cached) - margin(cached[0])))
    assert bool((actual <= eps + 1e-14).all())


def test_public_selection_prefers_worst_style_and_deterministic_ties():
    y = np.tile([True, True, False, False], 3)
    styles = np.repeat(["line", "photo", "render"], 4)
    ideal = np.where(y, 1.0, -1.0)
    partial = ideal.copy()
    for i in (0, 4, 8):
        partial[i] *= -1
    q = ideal.copy()
    q[8:10] *= -1
    scores = {m: np.stack([-ideal * 0, ideal * 0], axis=1).tolist() for m in METHODS}
    scores["K24"] = np.stack([-partial, partial], axis=1).tolist()
    scores["Q32"] = np.stack([-q, q], axis=1).tolist()
    rule = select_module(scores, y, styles)
    assert rule["selected"] == "K24"
    same = {m: np.stack([-ideal, ideal], axis=1).tolist() for m in METHODS}
    assert select_module(same, y, styles)["selected"] == "Q24"


def test_public_offset_repairs_public_bias_without_reversing_ranking():
    y = np.tile([True, True, False, False], 3)
    styles = np.repeat(["line", "photo", "render"], 4)
    values = np.tile([6.0, 8.0, 4.0, 5.0], 3)
    rule = calibrate_offset(values, y, styles)
    assert rule["threshold"] == 5.5
    before = binary_auc(y, values)
    original = np.stack([-values, values], axis=1)
    calibrated = calibrated_scores(original, 2 * rule["threshold"])
    assert np.array_equal(np.argmin(calibrated, axis=1) == 0, y)
    assert before == binary_auc(y, margin(calibrated)) == 1.0
    assert calibrate_offset(np.where(y, 1.0, -1.0), y, styles)["threshold"] == 0.0


def test_gradient_rowspace_containment_does_not_guarantee_attribute_preservation():
    h = torch.eye(3, dtype=torch.float64)
    c = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    g = c @ h
    basis, _ = row_basis(g, 1e-6, 1e-10)
    prototypes = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    distances = (prototypes - (prototypes @ basis.T) @ basis).norm(dim=1)
    assert distances.tolist() == [1.0, 0.0]
    assert torch.linalg.matrix_rank(g) <= torch.linalg.matrix_rank(h)
