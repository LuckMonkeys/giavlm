import pytest
import torch

from evaluation import last_b_token_readout as base
from evaluation.last_b_positions import POSITIONS, candidates_for, evaluate_all, metrics


def test_later_attention_propagates_answer_loss_to_earlier_positions():
    """The penultimate output lacks the final module's supervised-only support."""
    generator = torch.Generator().manual_seed(504)
    early = torch.randn(4, 6, generator=generator, dtype=torch.float64, requires_grad=True)
    # Fixed causal attention is enough to exhibit the cross-position Jacobian;
    # use a dense answer query to avoid assuming an identity residual path.
    logits = early @ early.T / 6**.5
    causal = torch.ones(4, 4, dtype=torch.bool).tril()
    attention = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    after_attention = early + attention @ early
    final_output = after_attention + .1 * after_attention.sin()
    final_output.retain_grad()
    final_output[-1].square().sum().backward()
    assert torch.equal(final_output.grad[:-1], torch.zeros_like(final_output.grad[:-1]))
    assert bool((early.grad[:-1].norm(dim=-1) > 0).all())


@pytest.mark.parametrize('fusion', [False, True])
def test_one_sort_metrics_exactly_match_previous_fieldwise_definition(fusion):
    scores = torch.tensor([-.3, -.9, -.2, -.8, -.6, -.5], dtype=torch.float64)
    flags = torch.tensor([True, False, False, False, False, False])
    forbidden = {5}
    truths = {'question': {0, 1, 3}, 'answer': {2, 4}, 'joint': {0, 1, 2, 3, 4}}
    chosen = {0, 1, 3, 4}
    score = None if fusion else scores
    actual = metrics(score, chosen, truths, flags, forbidden)
    for part, truth in truths.items():
        expected = base.token_metrics(score, chosen, truth, flags, forbidden)
        assert actual == actual | {f'{part}_{key}': value for key, value in expected.items()}


def test_every_position_and_mixture_keeps_same_budget_without_truth():
    scores = {'baseline': torch.arange(240, dtype=torch.float64),
              'lm_prior': -torch.arange(240, dtype=torch.float64)}
    for i, position in enumerate(POSITIONS):
        scores[position] = torch.arange(240, dtype=torch.float64).roll(25 * i)
        scores['wrong_' + position] = scores[position].roll(1)
    flags = torch.zeros(240, dtype=torch.bool)
    flags[:10] = True
    chosen = candidates_for(scores, flags, {0, 239})
    assert len(chosen) == 12
    assert {len(v) for v in chosen.values()} == {109}
    for position in POSITIONS:
        assert chosen[position] == base.selected_ids(scores[position], flags, {0, 239})
        assert chosen['mix_' + position] == base.mixed_ids(scores['baseline'], scores[position], flags, {0, 239})


def test_position_evaluation_requires_every_candidate_commit_before_truth(tmp_path, monkeypatch):
    monkeypatch.setattr('evaluation.last_b_positions.verify_plan', lambda *args: {})
    def forbidden(*args, **kwargs):
        raise AssertionError('Premature private reference read')
    monkeypatch.setattr('evaluation.last_b_positions.read_tensors', forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
    assert not (tmp_path / 'evaluation-barrier.json').exists()
