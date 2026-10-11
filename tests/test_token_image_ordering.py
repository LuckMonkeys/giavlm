import torch

from evaluation.token_image_ordering import background, image_bank, ordering_metrics, token_comparison


def test_bank_includes_near_and_far_and_preserves_truth():
    truth = torch.full((1, 3, 16, 16), .4)
    bank = image_bank(truth)
    assert len(bank) == 9
    torch.testing.assert_close(bank[0][1], truth)
    distances = [float((x-truth).square().mean()) for _, x, _ in bank[1:]]
    assert min(distances) <= 1e-3 and max(distances) >= 1e-2
    assert all(0 <= x.min() and x.max() <= 1 for _, x, _ in bank)
    assert distances[:4] == sorted(distances[:4])
    assert distances[4:] == sorted(distances[4:])


def test_ordering_direction_and_ties():
    x = [.0001, .0005, .02, .1]
    correct = ordering_metrics(x, [1, 2, 3, 4])
    reverse = ordering_metrics(x, [4, 3, 2, 1])
    tied = ordering_metrics(x, [1, 1, 1, 1])
    assert correct['rho'] == 1 and correct['near_win'] == 1
    assert reverse['rho'] == -1 and reverse['near_win'] == 0
    assert tied['rho'] == 0 and tied['near_win'] == .5
    comparison = token_comparison([1, 1, 1, 1], 0, [1, 2, 3])
    assert comparison['top1_credit'] == .25 and comparison['pairwise_win'] == .5


def test_background_keeps_lengths_and_varies_only_declared_context():
    refs = {'questions': torch.tensor([[3, 4, 5, 2, 0]]), 'targets': torch.tensor([[6, 7, 2, 0]])}
    guess = {'questions': torch.tensor([[8, 8, 8, 2, 0]]), 'targets': torch.tensor([[9, 9, 2, 0]])}
    lengths = {'questions': 3, 'targets': 2}
    full, fraction = background(refs, guess, lengths, 'targets', 1, 'oracle_rest', 8)
    assert fraction == 1
    torch.testing.assert_close(full['questions'], refs['questions'])
    partial, fraction = background(refs, guess, lengths, 'targets', 1, 'half_rest', 8)
    assert fraction == .5
    unknown, fraction = background(refs, guess, lengths, 'targets', 1, 'unknown_words', 8)
    assert fraction == 0
    for field, length in lengths.items():
        torch.testing.assert_close(partial[field][0, length:], refs[field][0, length:])
        torch.testing.assert_close(unknown[field], guess[field])
