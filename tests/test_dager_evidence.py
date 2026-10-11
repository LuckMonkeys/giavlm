import torch
from types import SimpleNamespace

from attacks.analytic.dager_evidence import b_energy, fit_logistic, logistic_score, negative_moments


def test_null_calibration_excludes_present_token_values():
    values = torch.arange(24, dtype=torch.float64).reshape(6, 2, 2)
    present = torch.zeros(6, 2, dtype=torch.bool)
    present[0, 0] = True
    before = negative_moments(values, present)
    values[0, 0] = 1e12
    after = negative_moments(values, present)
    for key in before:
        torch.testing.assert_close(before[key], after[key])


def test_b_innovation_removes_component_already_in_a_sketch():
    rng = torch.Generator().manual_seed(91)
    features = torch.randn(12, 6, generator=rng, dtype=torch.float64)
    a = torch.randn(2, 6, generator=rng, dtype=torch.float64)
    b = torch.randn(7, 2, generator=rng, dtype=torch.float64)
    gb = torch.randn(7, 2, generator=rng, dtype=torch.float64)
    value, independent, _ = b_energy(features, a, b, gb)
    changed, _, _ = b_energy(features, a, b, gb + b @ torch.eye(2, dtype=torch.float64) * 5)
    torch.testing.assert_close(value, changed)
    torch.testing.assert_close(b.T @ independent, torch.zeros(2, 2, dtype=torch.float64), atol=1e-12, rtol=0)


def test_public_logistic_evidence_learns_additional_signal():
    rng = torch.Generator().manual_seed(12)
    x = torch.randn(300, 2, generator=rng, dtype=torch.float64)
    y = (x[:, 0] + 2*x[:, 1] > 0).double()
    fit = fit_logistic(x[:200], y[:200], [0, 1], .01)
    correct = (logistic_score(x[200:], fit) < 0) == y[200:].bool()
    assert correct.double().mean() > .95


def test_surface_groups_handle_padded_embedding_rows():
    from evaluation.dager_evidence_study import group_ids
    tokens = ['▁Cat', 'cat', None, None, '?']
    adapter = SimpleNamespace(vocab_size=len(tokens), tokenizer=SimpleNamespace(
        convert_ids_to_tokens=lambda i: tokens[i]))
    groups = group_ids(adapter)
    assert groups[0] == groups[1]
    assert len(set(groups[2:].tolist())) == 3
