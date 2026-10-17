from copy import deepcopy

import numpy as np
import pytest
import torch

from evaluation.image_property_data import qa_signature
from evaluation.image_property_scoring import (aggregate_classes, binary_auc,
                                               classification_stats, paired_bootstrap,
                                               score_bank)


def test_bank_scores_match_direct_float64_cosines():
    rng = np.random.default_rng(123)
    bank = [{k: torch.tensor(rng.normal(size=n), dtype=torch.float64)
             for k, n in [('small', 3), ('large', 19)]} for _ in range(5)]
    target = {k: torch.tensor(rng.normal(size=v.shape), dtype=torch.float64)
              for k, v in bank[0].items()}
    result = score_bank(bank, [target])[0]['scores']
    mean = {k: torch.stack([g[k] for g in bank]).mean(0) for k in target}
    for centered, global_name, equal_name in [(False, 'global', 'equal_tensor'),
                                              (True, 'centered', 'centered_equal_tensor')]:
        expected_global, expected_equal = [], []
        for candidate in bank:
            a = [candidate[k]-(mean[k] if centered else 0) for k in sorted(target)]
            b = [target[k]-(mean[k] if centered else 0) for k in sorted(target)]
            expected_global.append(float(1-torch.nn.functional.cosine_similarity(torch.cat(a), torch.cat(b), dim=0)))
            expected_equal.append(np.mean([float(1-torch.nn.functional.cosine_similarity(x, y, dim=0)) for x, y in zip(a, b)]))
        np.testing.assert_allclose(result[global_name], expected_global, atol=2e-7)
        np.testing.assert_allclose(result[equal_name], expected_equal, atol=2e-7)


def test_centering_uses_bank_only_and_removes_common_offset():
    bank = [{'a': torch.tensor([1., 2., 4.])}, {'a': torch.tensor([-3., 4., 2.])}]
    target = {'a': torch.tensor([3., 1., 2.])}
    offset = torch.tensor([100., -35., 6.])
    first = score_bank(bank, [target])[0]['scores']
    shifted = score_bank([{'a': p['a']+offset} for p in bank], [{'a': target['a']+offset}])[0]['scores']
    np.testing.assert_allclose(first['centered'], shifted['centered'], atol=1e-7)


def test_equal_tensor_mask_is_fixed_across_candidates():
    bank = [{'a': torch.tensor([1., 1.]), 'b': torch.zeros(2)},
            {'a': torch.tensor([1., -1.]), 'b': torch.tensor([1., 2.])}]
    target = {'a': torch.tensor([2., 1.]), 'b': torch.tensor([1., 1.])}
    score = score_bank(bank, [target])[0]
    assert score['valid_tensors']['equal_tensor'] == 1
    expected = [1-torch.nn.functional.cosine_similarity(g['a'], target['a'], dim=0) for g in bank]
    np.testing.assert_allclose(score['scores']['equal_tensor'], expected, atol=1e-7)


def test_identical_bank_has_undefined_centered_scores():
    bank = [{'a': torch.tensor([1., 2.])} for _ in range(3)]
    score = score_bank(bank, [{'a': torch.tensor([2., 1.])}])[0]['scores']
    assert score['centered'] is None
    assert score['centered_equal_tensor'] is None
    assert score['global'] is not None


def test_bank_incompatible_tensors_fail():
    with pytest.raises(ValueError):
        score_bank([{'a': torch.ones(3)}], [{'a': torch.ones(2)}])
    with pytest.raises(ValueError):
        score_bank([{'a': torch.ones(3)}], [{'b': torch.ones(3)}])
    with pytest.raises(ValueError):
        score_bank([{'a': torch.tensor([float('nan')])}], [{'a': torch.ones(1)}])


def test_top_m_is_per_class_and_ignores_order():
    values = [9., 3., 1., 2., 6., 4.]
    labels = ['A', 'B', 'A', 'A', 'B', 'B']
    assert aggregate_classes(values, labels, ['A', 'B'], 2) == [1.5, 3.5]
    with pytest.raises(ValueError):
        aggregate_classes(values, labels, ['A', 'B'], 4)


def test_missing_class_never_becomes_three_class_balanced_accuracy():
    stats = classification_stats(['CT', 'X-Ray'], [[0., 1., 2.], [1., 2., 0.]], ['CT', 'MRI', 'X-Ray'])
    assert stats['accuracy'] == 1
    assert stats['balanced_accuracy'] is None
    assert stats['supported_class_balanced_accuracy'] == 1
    assert stats['class_auroc']['MRI'] is None
    assert stats['recall']['MRI'] is None


def test_invalid_score_is_not_silently_dropped():
    stats = classification_stats(['A', 'B'], [None, [1., 0.]], ['A', 'B'])
    assert stats['n'] == 2 and stats['invalid'] == 1
    assert stats['accuracy'] == .5
    assert stats['confusion'][0][-1] == 1


def test_auc_ties_and_missing_support():
    assert binary_auc([0, 1], [1., 1.]) == .5
    assert binary_auc([0, 1], [0., 1.]) == 1.
    assert binary_auc([1, 1], [0., 1.]) is None


def test_paired_bootstrap_preserves_pairing():
    labels = ['A', 'A', 'B']
    scores = [[0, 1], [1, 0], [1, 0]]
    result = paired_bootstrap(labels, scores, ['A', 'B'], deepcopy(scores), repeats=100)
    assert result['accuracy_ci95'] == [0., 0.]
    assert result['supported_class_ba_ci95'] == [0., 0.]
    assert result['bootstrap_method'] == 'stratified image bootstrap'
    assert {'method': 'centered', **result}['method'] == 'centered'


def test_annotation_join_signature_keeps_full_multiset():
    first = [{'question': 'q', 'target': 'a'}, {'question': 'q', 'target': 'a'}]
    second = [{'question': 'q', 'answer': 'a'}]
    assert qa_signature(first, 'target') != qa_signature(second, 'answer')
