import torch

from evaluation.causal_visual_feature_check import differences, gradient_comparison


def test_gradient_scale_or_left_basis_change_does_not_imply_changed_rowspace():
    reference = torch.tensor([[1., 0., 0.], [0., 2., 0.]], dtype=torch.float64)
    mixed = torch.tensor([[0., 2.], [3., 0.]], dtype=torch.float64) @ reference
    result = gradient_comparison(mixed, reference)
    assert result["relative_l2"] > 0
    assert result["rowspace_distance"] < 1e-7
    assert result["rank"] == result["reference_rank"] == 2


def test_feature_difference_is_not_hidden_by_patch_mean_cancellation():
    reference = torch.ones(2, 3)
    changed = reference.clone()
    changed[0] += 1
    changed[1] -= 1
    assert torch.equal(changed.mean(0), reference.mean(0))
    result = differences(changed, reference)
    assert result["max_absolute_error"] == 1 and not result["bit_equal"]
