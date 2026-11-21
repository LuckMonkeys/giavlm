import numpy as np
import pytest
import torch

from evaluation import last_b_token_readout as base
from evaluation.last_b_layer_sweep import preflight
from evaluation.last_b_sweep_math import (
    METHODS, POSITIONS, cached_scores, layer_ranking, order_and_ranks, public_features,
    selection, supplement, vector_metrics,
)


def test_cached_features_keep_original_score_formula_and_candidates():
    g = torch.Generator().manual_seed(604)
    head = torch.randn(220, 24, generator=g, dtype=torch.float64)
    gamma = torch.randn(24, generator=g, dtype=torch.float64)
    gradient = torch.randn(24, 8, generator=g, dtype=torch.float64)
    features, norms = public_features(head, gamma, chunk=17)
    actual = cached_scores(gradient, features, norms, chunk=17)
    expected = base.readout_scores(gradient, head, gamma, chunk=17)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    flags = torch.zeros(220, dtype=torch.bool)
    flags[:9] = True
    order, _ = order_and_ranks(actual, flags, {219})
    assert selection(order, set(range(9))) == base.selected_ids(expected, flags, {219})


@pytest.mark.parametrize('fused', [False, True])
def test_vectorized_metrics_and_supplement_match_previous_definitions(fused):
    scores = torch.arange(200, dtype=torch.float64).roll(30)
    original = torch.arange(200, dtype=torch.float64)
    flags = torch.zeros(200, dtype=torch.bool)
    flags[:10] = True
    forbidden, ambiguous = {0, 199}, set(range(1, 10))
    order, ranks = order_and_ranks(scores, flags, forbidden)
    base_order, _ = order_and_ranks(original, flags, forbidden)
    selected = supplement(base_order, order, ambiguous) if fused else selection(order, ambiguous)
    expected = base.mixed_ids(original, scores, flags, forbidden) if fused else base.selected_ids(scores, flags, forbidden)
    assert selected == expected and len(selected) == 109
    truths = {'question': {1, 20, 140}, 'answer': {3, 32, 125}}
    truths['joint'] = truths['question'] | truths['answer']
    actual = vector_metrics(None if fused else ranks, selected, truths, ambiguous)
    for part, truth in truths.items():
        previous = base.token_metrics(None if fused else scores, selected, truth, flags, forbidden)
        for key, value in previous.items():
            assert actual[f'{part}_{key}'] == (pytest.approx(value) if value is not None else None)


def test_rank_order_ties_keep_token_id_and_public_exclusions():
    scores = torch.zeros(8, dtype=torch.float64)
    flags = torch.tensor([True, False, False, False, True, False, False, False])
    order, ranks = order_and_ranks(scores, flags, {1, 7})
    assert order.tolist() == [2, 3, 5, 6]
    np.testing.assert_array_equal(ranks, [0, 0, 1, 2, 0, 3, 4, 0])


def test_cross_defense_development_ranking_uses_fixed_tiebreaks():
    rows = []
    for name in POSITIONS:
        for condition in ('noise', 'sparsify'):
            value = .7
            if name == 'l10_o':
                value = .9 if condition == 'noise' else .5
            rows.append(dict(condition=condition, image='random', method='mix_' + name,
                             answer_recall=value, answer_full_coverage=.4))
    ranked = layer_ranking(rows)
    assert ranked[0]['position'] == 'l32_down'
    assert ranked[1]['position'] == 'l32_o'
    assert ranked[-1]['position'] == 'l10_o'
    assert len(POSITIONS) == 64 and len(METHODS) == 195


def test_sweep_refuses_reference_evaluation_before_all_score_commits(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Premature private reference read')
    monkeypatch.setattr('evaluation.last_b_layer_sweep.read_tensors', forbidden)
    with pytest.raises(FileNotFoundError):
        preflight(tmp_path, {})
    assert not (tmp_path / 'evaluation-barrier.json').exists()
