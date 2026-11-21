import numpy as np

from evaluation.property_next_report import paired_interval, paired_swap_interval


def test_paired_gain_uses_head_positive_direction_and_keeps_classes_balanced():
    labels = ["head"] * 3 + ["non-head"] * 5
    correct = [[0., 1.]] * 3 + [[1., 0.]] * 5
    constant_nonhead = [[1., 0.]] * 8
    result = paired_interval(labels, correct, constant_nonhead, repeats=100)
    assert result["gain"] == .5
    assert result["ci95"] == [.5, .5]
    reverse = paired_interval(labels, constant_nonhead, correct, repeats=100)
    assert reverse["gain"] == -.5
    order = [4, 0, 6, 2, 5, 1, 3, 7]
    reordered = paired_interval(np.array(labels)[order].tolist(), np.array(correct)[order].tolist(),
                                np.array(constant_nonhead)[order].tolist(), repeats=100)
    assert reordered == result


def test_swapped_uploads_resample_the_six_pairs_without_pseudo_replication():
    labels = ["head"] * 6 + ["non-head"] * 6
    # All heads correct; three non-heads correct, three predicted head.
    first = [[0., 1.]] * 6 + [[1., 0.]] * 3 + [[0., 1.]] * 3
    mapping = [(i + 6) % 12 for i in range(12)]
    second = [first[i] for i in mapping]
    result = paired_swap_interval(labels, first, second, mapping)
    assert result["gain"] == .5 and result["pairs"] == 6
    np.testing.assert_allclose(result["ci95"], [1 / 6, 5 / 6])
