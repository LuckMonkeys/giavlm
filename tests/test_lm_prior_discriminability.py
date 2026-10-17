from copy import deepcopy

import pytest
import torch

from evaluation.lm_prior_discriminability import inputs_for, text_conditions, validate_result


def test_committed_reconstruction_must_match_observation_and_condition():
    result = {'status': 'completed', 'observation_id': 'a',
              'condition': {'lora_rank': 8, 'knowledge': 'private', 'token_lengths_known': True},
              'provenance': {'options': {'lm_prior_weight': .01, 'mode': 'public_residual',
                                         'image_source': 'random'}}}
    validate_result(result, {'observation_id': 'a'}, 8, .01)
    for field, value in [('status', 'failed'), ('observation_id', 'b')]:
        invalid = deepcopy(result)
        invalid[field] = value
        with pytest.raises(ValueError):
            validate_result(invalid, {'observation_id': 'a'}, 8, .01)
    with pytest.raises(ValueError):
        validate_result(result, {'observation_id': 'a'}, 32, .01)
    with pytest.raises(ValueError):
        validate_result(result, {'observation_id': 'a'}, 8, 0.)


def test_controls_use_saved_tokens_and_only_replace_declared_field():
    ref = {'questions': torch.tensor([[3, 4, 2]]), 'targets': torch.tensor([[5, 2]])}
    base = {'questions': torch.tensor([[6, 7, 2]]), 'targets': torch.tensor([[8, 2]])}
    prior = {'questions': torch.tensor([[3, 7, 2]]), 'targets': torch.tensor([[5, 2]])}
    cases = dict(text_conditions(ref, base, prior))
    assert cases['baseline'] is base and cases['lm_prior'] is prior
    assert torch.equal(cases['prior_question']['questions'], prior['questions'])
    assert torch.equal(cases['prior_question']['targets'], ref['targets'])
    assert torch.equal(cases['prior_answer']['questions'], ref['questions'])
    assert torch.equal(cases['prior_answer']['targets'], prior['targets'])
    assert len(cases) == 5


def test_validation_does_not_substitute_development_observations():
    source, model, prior = inputs_for(32, 'validation')
    assert source == prior
    assert 'round-0010' in str(model)
    source, _, prior = inputs_for(32, 'development')
    assert source != prior
    with pytest.raises(ValueError):
        inputs_for(8, 'validation')
