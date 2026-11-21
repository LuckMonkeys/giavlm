import torch

from evaluation.question_filter_math import blend
from evaluation.question_filter_token_audit import changes, contextual_words, source_map, word_category


def test_gains_and_losses_balance_deduplicated_reference_hits():
    reference, before, after = {1, 2, 3}, {1, 7}, {2, 3, 8}
    gain, loss = changes(reference, after, before)
    assert gain == {2, 3} and loss == {1}
    assert len(reference & after) - len(reference & before) == len(gain) - len(loss)


def test_bpe_letter_in_mri_is_not_misclassified_as_pronoun():
    words = contextual_words('MRI is shown.', [(0, 1), (1, 2), (2, 3), (4, 6), (7, 12), (12, 13)])
    assert words[:3] == [{'MRI'}] * 3
    assert word_category(next(iter(words[2]))) == 'Medical modality'
    assert word_category('I') == 'Function word'
    assert word_category(next(iter(words[3]))) == 'Function word'
    assert word_category(next(iter(words[-1]))) == 'Punctuation'


def test_hyphenated_and_alphanumeric_modalities_keep_word_context():
    assert contextual_words('T2 x-ray', [(0, 1), (1, 2), (3, 4), (4, 5), (5, 8)]) == [
        {'T2'}, {'T2'}, {'x-ray'}, {'x-ray'}, {'x-ray'}]


def test_new_candidates_against_b_anchor_can_only_come_from_lm_channel():
    rng = torch.Generator().manual_seed(61105)
    ambiguous = set(range(300, 309))
    for _ in range(10):
        base = torch.randperm(300, generator=rng).tolist()
        b = torch.randperm(300, generator=rng).tolist()
        lm = torch.randperm(300, generator=rng).tolist()
        origins = source_map(base, b, lm, ambiguous)
        anchor = blend([base, b], [80, 20], ambiguous)
        assert len(origins) == 109
        assert all(origins[t] == 'lm20' for t in set(origins) - anchor)
        assert sum(v == 'original60' for v in origins.values()) == 60
        assert sum(v == 'b20' for v in origins.values()) == 20
        assert sum(v == 'lm20' for v in origins.values()) == 20


def test_unrecognized_words_are_not_silently_called_generic():
    assert word_category('rarewordfragment') == 'Other lexical content'
    assert word_category('What') == 'Question word'
    assert word_category('image') == 'General image term'
