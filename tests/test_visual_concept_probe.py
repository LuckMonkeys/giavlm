import torch

from evaluation.visual_concept_probe import concept_distance, class_scores, visual_basis, native_hard_loss


def test_concept_distance_rejects_zero_text_residual():
    basis = torch.tensor([[0., 1., 0.]], dtype=torch.float64)
    template = torch.tensor([[1., 0., 0.]], dtype=torch.float64)
    words = torch.eye(3, dtype=torch.float64)
    d, valid = concept_distance(words, basis, template)
    assert valid.tolist() == [False, True, True]
    assert d[1].item() == 0 and d[2].item() == 1


def test_rank_truncated_visual_space_matches_svd():
    torch.manual_seed(11)
    x = torch.randn(8, 17, dtype=torch.float64)
    template = torch.linalg.qr(torch.randn(17, 2, dtype=torch.float64)).Q.T
    basis, _ = visual_basis(x, template, 4)
    xp = x - (x @ template.T) @ template
    _, _, direct = torch.linalg.svd(xp, full_matrices=False)
    torch.testing.assert_close(basis.T @ basis, direct[:4].T @ direct[:4], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(basis @ template.T, torch.zeros(4, 2, dtype=torch.float64), atol=1e-12, rtol=0)


def test_class_scores_have_explicit_word_coverage():
    d = torch.arange(8, dtype=torch.float64) / 8
    assert class_scores(d, torch.ones(8, dtype=torch.bool)) == [.1875, .6875]
    mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.bool)
    assert class_scores(d, mask) is None


def test_injected_loss_includes_eos_and_excludes_padding():
    logits = torch.tensor([[[3., 1., 0.], [0., 4., 0.], [9., 0., 0.]]])
    target = torch.tensor([[0, 1, 2]])
    loss = native_hard_loss(logits, target, pad=2)
    expected = (-logits[0, :2].log_softmax(-1)[torch.arange(2), target[0, :2]]).mean()
    torch.testing.assert_close(loss, expected)


def test_public_intervals_use_head_positive_score_direction():
    from evaluation.visual_concept_report import interval
    labels = ['head'] * 4 + ['non-head'] * 4
    scores = [[0., 1.]] * 4 + [[1., 0.]] * 4
    assert interval(labels, scores, repeats=100)['head_auc_ci95'] == [1., 1.]
    families = ['a'] * 2 + ['b'] * 2 + ['c'] * 2 + ['d'] * 2
    assert interval(labels, scores, families, repeats=100)['balanced_accuracy_ci95'] == [1., 1.]
    assert interval(labels, [None] + scores[1:])['head_auc_ci95'] is None
