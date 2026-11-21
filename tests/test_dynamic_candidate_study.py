from dataclasses import replace

import pytest
import torch

from attacks.analytic.dager_adapted import DAGERSearch, forbidden_token_ids
from attacks.analytic.dager_subspace import select_tokens
from core.adapters.llava import LlavaTextView
from core.artifacts import read_tensors
from evaluation.dynamic_candidate_study import TextOnlyView, evaluate_all, search, supplement
from test_dager import fixture, specification


def test_supplement_rescues_missing_lm_token_and_matches_expanded_width():
    nll = torch.tensor([9., 8., 7., 6., 5., 4., 3., 2., 1., 0.])
    # Both top LM entries are outside the static vocabulary; no oracle labels
    # are provided to supplement. Ties and overlaps have deterministic behavior.
    assert supplement([0, 1, 2], [0, 1, 2, 3, 4], nll, set(), count=2) == [0, 1, 2, 8, 9]
    assert supplement([8, 9, 0], [8, 9, 0, 1, 2], nll, set(), count=2) == [0, 1, 2, 8, 9]
    assert 9 not in supplement([0, 1, 2], [0, 1, 2, 3, 4], nll, {9}, count=2)


@pytest.mark.parametrize('k,variant', [(100, 'gradient100'), (116, 'gradient116')])
def test_static_engine_reproduces_original_search(k, variant, tmp_path):
    torch.manual_seed(701)
    adapter, _, obs = fixture(knowledge='image_known', answer='red')
    spec = specification(lm_prior_weight=.01, max_candidates=k, token_selection='topk')
    old = DAGERSearch(adapter, obs, spec, tmp_path / 'original')
    expected = old.run(False)
    helper = DAGERSearch(adapter, obs, spec, None)
    helper.prepare()
    helper.filter_tokens()
    expanded, _ = select_tokens(helper.scores, helper.ambiguous, forbidden_token_ids(adapter),
                                 replace(spec.dager, max_candidates=116))
    result = search(helper.view, helper.spans[1], helper.state['candidates'], expanded,
                    forbidden_token_ids(adapter), spec, variant, tmp_path / 'new')
    assert (result.questions, result.targets) == (expected.questions, expected.targets)
    assert result.provenance['final_score'] == expected.provenance['final_score']
    left, right = read_tensors(tmp_path / 'original/text_tokens.safetensors'), read_tensors(tmp_path / 'new/text_tokens.safetensors')
    assert all(torch.equal(left[k], right[k]) for k in left)
    for cost in ('prefix_candidates_evaluated', 'prefix_forward_batches', 'lm_prior_prefixes',
                 'lm_prior_forward_batches', 'lm_prior_input_tokens'):
        assert result.costs[cost] == expected.costs[cost]


def test_lm_only_has_no_visual_or_gradient_calls_and_matches_text_context(tmp_path, monkeypatch):
    torch.manual_seed(702)
    adapter, _, obs = fixture(knowledge='image_known', answer='red')
    ordinary = LlavaTextView(adapter, obs)
    view = TextOnlyView(adapter, obs.public_question_lengths[0], obs.public_target_lengths[0])
    assert view.text_prior_ids([[]]).tolist() == ordinary.text_prior_ids([[]]).tolist()
    def forbidden(*args, **kwargs):
        raise AssertionError('LM-only accessed visual features or gradient span')
    monkeypatch.setattr(adapter, 'visual_embeddings', forbidden)
    monkeypatch.setattr(TextOnlyView, 'prefix_features', forbidden)
    result = search(view, None, [], [], forbidden_token_ids(adapter),
                    specification(lm_prior_weight=.01), 'lm_only', tmp_path)
    assert result.status == 'completed'
    assert result.costs['prefix_forward_batches'] == 0
    assert result.provenance['final_score_type'] == 'text_only_nll'


def test_dynamic_search_matches_static_expansion_counts_when_all_tokens_fit(tmp_path):
    torch.manual_seed(703)
    adapter, _, obs = fixture(knowledge='image_known', answer='red')
    spec = specification(lm_prior_weight=.01, token_selection='topk')
    helper = DAGERSearch(adapter, obs, spec, None)
    helper.prepare()
    helper.filter_tokens()
    candidates = helper.state['candidates']
    forbidden = forbidden_token_ids(adapter)
    static = search(helper.view, helper.spans[1], candidates, candidates, forbidden, spec,
                    'gradient116', tmp_path / 'static')
    dynamic = search(helper.view, helper.spans[1], candidates, candidates, forbidden, spec,
                     'dynamic16', tmp_path / 'dynamic')
    assert dynamic.provenance['final_beam'] == static.provenance['final_beam']
    assert dynamic.costs['prefix_candidates_evaluated'] == static.costs['prefix_candidates_evaluated']


def test_evaluation_requires_all_commits_before_any_reference(tmp_path, monkeypatch):
    monkeypatch.setattr('evaluation.dynamic_candidate_study.verify', lambda *args: {})
    def forbidden(*args, **kwargs):
        raise AssertionError('Truth read before complete reconstructions')
    monkeypatch.setattr('evaluation.dynamic_candidate_study.read_tensors', forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
