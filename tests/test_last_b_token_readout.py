import pytest
import torch
from torch.nn import functional as F

from evaluation.last_b_token_readout import (
    fusion_gate, mixed_ids, preflight, readout_scores, token_metrics,
)


def test_final_down_b_has_only_supervised_position_gradients_and_chain_rule():
    g = torch.Generator().manual_seed(404)
    x = torch.randn(6, 5, generator=g, dtype=torch.float64)
    a = torch.randn(3, 5, generator=g, dtype=torch.float64)
    b = torch.randn(4, 3, generator=g, dtype=torch.float64, requires_grad=True)
    head = torch.randn(7, 4, generator=g, dtype=torch.float64)
    gamma = torch.randn(4, generator=g, dtype=torch.float64)
    residual = torch.randn(6, 4, generator=g, dtype=torch.float64)
    h = residual + 2 * (x @ a.T) @ b.T
    h.retain_grad()
    epsilon = 1e-6
    scale = (h.square().mean(-1, keepdim=True) + epsilon).sqrt()
    logits = (gamma * h / scale) @ head.T
    positions = torch.tensor([2, 4, 5])
    targets = torch.tensor([1, 6, 3])
    F.cross_entropy(logits[positions], targets).backward()
    assert torch.equal(h.grad[[0, 1, 3]], torch.zeros(3, 4, dtype=h.dtype))
    torch.testing.assert_close(b.grad, 2 * h.grad.T @ (x @ a.T))
    # Exact RMSNorm Jacobian includes the radial correction omitted by the
    # readout proxy. The target word direction is therefore only approximate.
    p = logits.detach()[positions].softmax(-1)
    v = (p @ head - head[targets]) * gamma
    hs, scales = h.detach()[positions], scale.detach()[positions]
    delta = v / scales - hs * (hs * v).sum(-1, keepdim=True) / (4 * scales.pow(3))
    torch.testing.assert_close(h.grad[positions], delta / 3)


def test_energy_matches_individual_token_response_and_scale_invariance():
    g = torch.Generator().manual_seed(405)
    gradient = torch.randn(5, 3, generator=g, dtype=torch.float64)
    head = torch.randn(13, 5, generator=g, dtype=torch.float64)
    gamma = torch.randn(5, generator=g, dtype=torch.float64)
    actual = readout_scores(gradient, head, gamma, chunk=4)
    expected = []
    for word in head:
        u = word * gamma
        response = sum(torch.dot(u, column).square() for column in gradient.T)
        expected.append(-response / (u.square().sum() * gradient.square().sum()))
    torch.testing.assert_close(actual, torch.stack(expected))
    torch.testing.assert_close(actual, readout_scores(-7 * gradient, head, 3 * gamma, chunk=13))


@pytest.mark.parametrize('bad', ['zero', 'dtype', 'shape'])
def test_invalid_readout_fails_fast(bad):
    gradient = torch.ones(4, 2, dtype=torch.float64)
    head, gamma = torch.ones(9, 4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)
    if bad == 'zero':
        gradient.zero_()
    elif bad == 'dtype':
        gradient = gradient.float()
    else:
        gamma = gamma[:2]
    with pytest.raises(ValueError):
        readout_scores(gradient, head, gamma)


def test_fixed_mixture_deduplicates_and_retains_common_ambiguity():
    base = torch.arange(200, dtype=torch.float64)
    supplement = -base
    flags = torch.zeros(200, dtype=torch.bool)
    flags[[0, 1, 2]] = True
    forbidden = {0, 199}
    selected = mixed_ids(base, supplement, flags, forbidden)
    assert len(selected) == 102
    assert selected == set(range(1, 83)) | set(range(179, 199))


def test_question_and_answer_metrics_do_not_confuse_full_coverage():
    scores = torch.arange(6, dtype=torch.float64)
    flags = torch.tensor([True, False, False, False, False, False])
    chosen = {0, 1, 2}
    q = token_metrics(scores, chosen, {0, 1, 3}, flags, set())
    a = token_metrics(scores, chosen, {2}, flags, set())
    assert q['recall'] == 2/3 and not q['full_coverage']
    assert a['recall'] == 1 and a['full_coverage']
    assert q['ap'] == pytest.approx((1 + 2/3)/2)
    assert a['ap'] == .5


def test_fusion_gate_requires_both_metrics_and_wrong_upload_control():
    rows = []
    for condition in ('fedsgd-b1', 'noise', 'sparsify'):
        for image in ('gt', 'random'):
            for method in ('baseline', 'last_b', 'lm_prior', 'wrong_upload'):
                rows.append(dict(condition=condition, image=image, method=method,
                                 answer_recall=.7 if method == 'last_b' else .5,
                                 answer_ap=.7 if method == 'last_b' else .5))
    assert fusion_gate(rows)['passed']
    for r in rows:
        if r['method'] == 'wrong_upload':
            r['answer_ap'] = .8
    assert not fusion_gate(rows)['passed']


def test_preflight_never_opens_references_before_all_score_commits(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Premature reference read')
    monkeypatch.setattr('evaluation.last_b_token_readout.read_tensors', forbidden)
    with pytest.raises(FileNotFoundError):
        preflight(tmp_path, {})
    assert not (tmp_path / 'evaluation-barrier.json').exists()
