import numpy as np

from evaluation.property_geometry import centered_blocks, score_geometry, weights


def example():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(3, 17, 25))
    return x, x @ x.transpose(0, 2, 1), np.arange(12), np.arange(12, 17), [0] * 6 + [1] * 6


def test_gram_centering_and_soft_projection_match_explicit_vectors():
    x, g, fit, query, labels = example()
    out, diag = score_geometry(g, fit, query, labels)
    mean = x[:, fit].mean(1, keepdims=True)
    f, q = x[:, fit] - mean, x[:, query] - mean
    k, d, qn, _, _ = centered_blocks(g, fit, query)
    np.testing.assert_allclose(k, f @ f.transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(d, q @ f.transpose(0, 2, 1), atol=1e-12)
    np.testing.assert_allclose(qn, (q * q).sum(2), atol=1e-12)
    expected = []
    for tensor in range(3):
        means = np.stack([f[tensor, :6].mean(0), f[tensor, 6:].mean(0)])
        residual = f[tensor] - means[np.asarray(labels)]
        _, _, v = np.linalg.svd(residual, full_matrices=False)
        p = np.eye(25) - .5 * v[:4].T @ v[:4]
        mp, qp = means @ p, q[tensor] @ p
        expected.append(1 - qp @ mp.T / np.linalg.norm(qp, axis=1)[:, None] / np.linalg.norm(mp, axis=1))
    np.testing.assert_allclose(out['suppress_k4_a0.5'], np.mean(expected, axis=0), atol=1e-12)
    assert 0 <= diag['suppression']['suppress_k4_a0.5']['between_retained'] <= 1


def test_weighting_scale_invariance_and_equal_tensor_endpoint():
    _, g, fit, query, labels = example()
    out, _ = score_geometry(g, fit, query, labels)
    scaled, _ = score_geometry(g * np.array([.0001, 10, 1e8])[:, None, None], fit, query, labels)
    np.testing.assert_allclose(out['weighted_b0'], out['centered_equal_tensor'])
    for key in ['weighted_b0.5', 'weighted_b1', 'fisher_r0.1', 'suppress_k1_a1']:
        np.testing.assert_allclose(out[key], scaled[key], atol=1e-10)


def test_query_never_changes_fitted_weights():
    _, g, fit, query, labels = example()
    _, first = score_geometry(g, fit, query, labels)
    altered = g.copy()
    altered[:, query, :] *= 7
    altered[:, :, query] *= 7
    _, second = score_geometry(altered, fit, query, labels)
    assert first['weights_b0.5'] == second['weights_b0.5']
    assert np.isclose(sum(first['weights_b0.5']), 1)


def test_fisher_matches_direct_regularized_solve():
    x, g, fit, query, labels = example()
    out, _ = score_geometry(g, fit, query, labels)
    margins = []
    for tensor in range(3):
        f = x[tensor, fit] - x[tensor, fit].mean(0)
        means = np.stack([f[:6].mean(0), f[6:].mean(0)])
        e = f - means[np.asarray(labels)]
        delta = means[0] - means[1]
        ridge = np.square(e).sum() / len(f) * .1
        direction = np.linalg.solve(e.T @ e + ridge * np.eye(25), delta)
        margin = (x[tensor, query] - x[tensor, fit].mean(0)) @ direction / (delta @ direction)
        margins.append(margin)
    expected = np.mean(margins, axis=0)
    np.testing.assert_allclose(out['fisher_r0.1'], np.stack([-expected, expected], axis=1), atol=1e-12)


def test_no_discriminative_candidate_signal_falls_back_to_uniform():
    k = np.eye(4)
    # Balanced class means coincide in this rank-one within-class construction.
    v = np.array([1., -1., 1., -1.])
    k = np.outer(v, v)[None]
    w, r = weights(k, [0, 0, 1, 1], 1)
    assert r[0] == 0 and w[0] == 1
