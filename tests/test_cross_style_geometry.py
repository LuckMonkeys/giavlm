import numpy as np

from evaluation.cross_style_geometry import (consistency_weights, domain_coefficients,
                                              remove_directions, score_cross_style)
from evaluation.property_geometry import centered_blocks


def test_domain_removal_matches_dense_parameter_projector():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(12, 9))
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    styles = ["a"] * 4 + ["b"] * 4
    gram = (x @ x.T)[None]
    k, d, qn, *_ = centered_blocks(gram, list(range(8)), list(range(8, 12)))
    e = domain_coefficients(labels, styles)
    fit = x[:8] - x[:8].mean(0)
    query = x[8:] - x[:8].mean(0)
    _, _, u = np.linalg.svd(e.T @ fit, full_matrices=False)
    for alpha in [.5, 1.]:
        p = np.eye(9) - alpha * u[:2].T @ u[:2]
        kk, dd, nn = remove_directions(k, d, qn, e, "all", alpha)
        np.testing.assert_allclose(kk[0], fit @ p @ p @ fit.T, atol=1e-12)
        np.testing.assert_allclose(dd[0], query @ p @ p @ fit.T, atol=1e-12)
        np.testing.assert_allclose(nn[0], np.square(query @ p).sum(1), atol=1e-12)
    assert np.linalg.matrix_rank(e) == 2


def test_consistency_weights_distinguish_same_and_opposite_class_directions():
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    styles = ["a"] * 4 + ["b"] * 4
    first = np.array([[1.], [1.], [-1.], [-1.], [1.], [1.], [-1.], [-1.]])
    second = first.copy()
    second[4:] *= -1
    gram = np.stack([first @ first.T, second @ second.T])
    weights, score = consistency_weights(gram, labels, styles, 1.)
    np.testing.assert_allclose(score, [1., 0.])
    np.testing.assert_allclose(weights, [1., 0.])


def test_cross_style_scores_only_depend_on_fit_labels():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(2, 12, 17))
    grams = x @ x.transpose(0, 2, 1)
    scores, _ = score_cross_style(grams, list(range(8)), [8, 9, 10, 11],
                                 [0, 0, 1, 1, 0, 0, 1, 1], ["a"] * 4 + ["b"] * 4)
    assert all(s.shape == (4, 2) and np.isfinite(s).all() for s in scores.values())
    np.testing.assert_allclose(scores["equal"], scores["consistency_b0"])
