from types import SimpleNamespace

import pytest

from evaluation.image_property_followup import oracle_text
from evaluation.image_property_followup_data import literal_property, NONEXPLICIT, region


def test_literal_control_does_not_treat_negation_as_confirmation():
    assert literal_property('Is this an MRI image?', 'No') is None
    assert literal_property('Is this an MRI image?', 'Yes') == 'MRI'
    assert literal_property('Is this CT or MRI?', 'Yes') is None
    assert literal_property('Which organ?', 'brain') is None
    assert literal_property('Which modality?', 'X-Ray.') == 'X-Ray'


def test_nonexplicit_filter_rejects_indirect_modality_words():
    for text in ['T2 weighted image', 'a radiograph', 'What modality?', 'CT scan', 'Magnetic resonance']:
        assert NONEXPLICIT.search(text)
    assert not NONEXPLICIT.search('Is the liver visible? Yes')
    assert region(['Brain_Tissue']) == region(['Brain']) == 'head'
    assert region(['Abdomen', 'Lung']) is None


def test_oracle_text_requires_declared_public_text():
    obs = SimpleNamespace(training=SimpleNamespace(knowledge='private'),
                          public_question_ids=[[1, 2]], public_target_ids=[[3, 4]])
    with pytest.raises(ValueError, match='declared public text'):
        oracle_text(obs, 'cpu')
    obs.training.knowledge = 'text_known'
    result = oracle_text(obs, 'cpu')
    assert result['questions'].tolist() == [[1, 2]]
    assert result['targets'].tolist() == [[3, 4]]
