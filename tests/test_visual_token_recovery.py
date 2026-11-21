import torch

from evaluation.visual_token_recovery import block_split, normalized_errors, toy_direction_check


def test_second_order_direction_matches_finite_difference():
    result = toy_direction_check()
    assert result["relative_error"] < result["tolerance"]


def test_matching_error_preserves_double_backward_and_scale():
    x = torch.tensor([.3, -.1], requires_grad=True)
    weight = torch.tensor([.2, .4], requires_grad=True)
    gradient, = torch.autograd.grad((x * weight).sum().square(), weight, create_graph=True)
    reference = torch.tensor([.1, -.3])
    error = normalized_errors([gradient], [reference])
    derivative, = torch.autograd.grad(error.sum(), x)
    assert torch.isfinite(derivative).all() and derivative.norm() > 0
    torch.testing.assert_close(error, normalized_errors([gradient * 10], [reference * 10]))


def test_parameter_holdout_spreads_across_each_module_and_factor():
    names = [f"backend.language_model.model.layers.{layer}.{module}.lora_{factor}.default.weight"
             for module in ["self_attn.q_proj", "mlp.down_proj"]
             for factor in ["A", "B"] for layer in range(32)]
    held = block_split(names)
    for start in range(0, len(names), 32):
        assert sum(held[start:start + 32]) in [6, 7]
    assert held == block_split(names)
