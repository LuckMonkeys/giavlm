import pytest
import torch

from attacks.analytic.dager_subspace import SpanFilter, residual, row_basis
from core.config import DAGEROptions
from evaluation.lora_ab_filter_study import evaluate_all
from evaluation.lora_ab_math import (
    consistency_projection, left_project, module_blocks, null_threshold, psd_root, score_blocks,
)


def fixture():
    generator = torch.Generator().manual_seed(304)
    a = torch.randn(4, 16, generator=generator, dtype=torch.float64)
    b = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    h = torch.randn(12, 16, generator=generator, dtype=torch.float64)
    return a, b, 2 * b.T @ h, 2 * h @ a.T


def test_clean_chain_identity_and_projection_is_idempotent():
    a, b, ga, gb = fixture()
    torch.testing.assert_close(ga @ a.T, b.T @ gb)
    x, z, error = consistency_projection(ga, gb, a, b)
    torch.testing.assert_close(x, ga, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(z, gb, rtol=1e-12, atol=1e-12)
    assert error < 1e-12


def test_joint_projection_matches_dense_weighted_constraint_solution():
    a, b, ga, gb = fixture()
    gb = gb + torch.arange(gb.numel(), dtype=gb.dtype).reshape_as(gb) / 3
    x, z, error = consistency_projection(ga, gb, a, b)
    # An independent vectorized constrained least-squares solve, rather than
    # repeating the Sylvester eigendecomposition used by the implementation.
    n = ga.numel() + gb.numel()
    columns = []
    for unit in torch.eye(n, dtype=ga.dtype):
        xa = unit[:ga.numel()].reshape_as(ga)
        xb = unit[ga.numel():].reshape_as(gb)
        columns.append((xa @ a.T - b.T @ xb).flatten())
    constraint = torch.stack(columns, dim=1)
    variance = torch.cat([ga.abs().max().square().expand(ga.numel()),
                          gb.abs().max().square().expand(gb.numel())])
    observed = torch.cat([ga.flatten(), gb.flatten()])
    gram = (constraint * variance) @ constraint.T
    expected = observed - variance * (constraint.T @ torch.linalg.solve(gram, constraint @ observed))
    torch.testing.assert_close(torch.cat([x.flatten(), z.flatten()]), expected, rtol=1e-10, atol=1e-10)
    assert error < 1e-12
    basis, _ = row_basis(a, 1e-10, 0)
    torch.testing.assert_close(residual(x - ga, basis), torch.zeros_like(ga), atol=1e-12, rtol=0)


def test_public_elimination_reveals_low_rank_residual_sketch():
    a, b, _, _ = fixture()
    public = torch.eye(16, dtype=a.dtype)[:6]
    generator = torch.Generator().manual_seed(305)
    ga = (torch.randn(4, 6, generator=generator, dtype=a.dtype) @ public +
          torch.randn(4, 2, generator=generator, dtype=a.dtype) @ torch.eye(16, dtype=a.dtype)[6:8])
    h = b @ torch.linalg.solve(b.T @ b, ga)
    gb = h @ a.T
    sketch = b.T @ gb - (ga @ public.T) @ (a @ public.T).T
    r = residual(ga, public)
    torch.testing.assert_close(sketch, r @ a.T)
    assert torch.linalg.matrix_rank(b.T @ gb) == 4
    assert torch.linalg.matrix_rank(sketch) == 2
    blocks, diagnostic, _ = module_blocks(ga, gb, a, b, public, 2, 0., 1729)
    assert diagnostic['length_rank'] == diagnostic['gate_rank'] == 2
    torch.testing.assert_close(blocks['residual_left'], blocks['baseline'], atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize('rank', [0, 4])
def test_zero_or_full_left_rank_preserves_exact_input(rank):
    _, _, ga, _ = fixture()
    result, projection = left_project(ga, torch.eye(4, dtype=ga.dtype), rank)
    assert torch.equal(result, ga)
    assert torch.equal(projection, torch.eye(4, dtype=ga.dtype))


def test_baseline_score_equals_original_span_filter():
    a, _, ga, _ = fixture()
    public = torch.eye(16, dtype=a.dtype)[:3]
    features = torch.cat([public[:1], a, ga])
    options = DAGEROptions(vocab_chunk_size=3)
    span = SpanFilter.build([ga], public, options)
    expected, flags = span.score(features)
    actual, _ = score_blocks([residual(ga / ga.norm(), public)], residual(features, public), flags, options)
    # The chunked GEMM can differ from the full-vocabulary call by one ulp.
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_colored_noise_root_and_fixed_null_draws():
    a, b, ga, gb = fixture()
    covariance = b.T @ b
    root = psd_root(covariance)
    torch.testing.assert_close(root @ root.T, covariance)
    public_a = a[:, :6]
    first = null_threshold(ga, gb, public_a, b, .01, 1729)
    second = null_threshold(ga, gb, public_a, b, .01, 1729)
    assert first == second and first > 0
    assert null_threshold(ga, gb, public_a, b, 0., 1729) == 0


def test_evaluation_preflights_all_scores_before_any_reference(tmp_path, monkeypatch):
    monkeypatch.setattr('evaluation.lora_ab_filter_study.verify_plan', lambda *args: {'source': '/tmp/x'})
    def forbidden(*args, **kwargs):
        raise AssertionError('Reference opened before the score completion barrier')
    monkeypatch.setattr('evaluation.lora_ab_filter_study.read_tensors', forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
    assert not (tmp_path / 'evaluation-barrier.json').exists()
