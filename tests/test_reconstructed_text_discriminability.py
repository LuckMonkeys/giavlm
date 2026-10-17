from collections import Counter

import pytest
import torch

from evaluation.reconstructed_text_discriminability import bank_statistics, shuffled, text_audit


@pytest.mark.parametrize('content', [[3, 4, 5, 3], [8], [8, 8], []])
def test_permutation_preserves_multiset_and_special_suffix(content):
    tokens = torch.tensor([content + [2, 0, 0]])
    output = shuffled(tokens, len(content), 21)
    assert Counter(output[0, :len(content)].tolist()) == Counter(content)
    assert torch.equal(output[0, len(content):], tokens[0, len(content):])
    assert torch.equal(tokens, torch.tensor([content + [2, 0, 0]]))
    assert torch.equal(output, shuffled(tokens, len(content), 21))
    if len(set(content)) > 1:
        assert not torch.equal(output, tokens)


def test_audit_distinguishes_order_errors_from_missing_tokens():
    ref = {'questions': torch.tensor([[3, 4, 3, 2, 0]]), 'targets': torch.tensor([[5, 2, 0]])}
    text = {'questions': torch.tensor([[4, 3, 3, 2, 0]]), 'targets': torch.tensor([[6, 2, 0]])}
    audit = text_audit(ref, text, {'questions': 3, 'targets': 1})
    assert audit['questions']['pure_order_error']
    assert audit['questions']['bag_recall'] == 1
    assert audit['questions']['position_accuracy'] == 1/3
    assert not audit['targets']['same_bag']
    assert audit['targets']['bag_recall'] == 0


def test_primary_ranking_excludes_truth_and_handles_ties():
    images = [{'id': name, 'mse': mse, 'losses': {'all': {'cosine': loss}}}
              for name, mse, loss in [('truth', 0., 100.), ('near1', .0001, 1.),
                                      ('near2', .0005, 2.), ('far1', .02, 3.), ('far2', .1, 4.)]]
    stats, _ = bank_statistics(images, 'all', 'cosine')
    assert stats['rho'] == 1 and stats['near_win'] == 1
    assert stats['truth_percentile'] == 1
    for image in images:
        image['losses']['all']['cosine'] = 1.
    stats, _ = bank_statistics(images, 'all', 'cosine')
    assert stats['rho'] == 0 and stats['near_win'] == .5
    assert stats['truth_percentile'] == .5
