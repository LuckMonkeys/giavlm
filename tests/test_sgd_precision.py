"""Functional SGD must match the actual optimizer, including low precision."""
import pytest
import torch

from core.config import TrainingSpec
from core.fl import _simulate_local_update
from core.types import Batch


class QuadraticClient(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-.3, .3, 128, dtype=dtype))

    def trainable(self):
        return {"weight": self.weight}

    def decay_parameter_names(self):
        return set()

    def forward(self, images, questions, targets):
        return (images * self.weight).float().square().mean()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 3])
def test_functional_sgd_matches_torch_optimizer_exactly(dtype, steps):
    model = QuadraticClient(dtype)
    spec = TrainingSpec(algorithm="fedavg", local_optimizer="sgd", local_steps=steps,
                        lr=.013, weight_decay=0., batch_size=1)
    images = torch.linspace(-2, 2, steps * 128, dtype=dtype).reshape(steps, 128)
    batch = Batch(images, torch.zeros(steps, 1, dtype=torch.long),
                  torch.zeros(steps, 1, dtype=torch.long))
    initial = model.weight.detach().clone()
    actual = _simulate_local_update(model, batch, spec, "client_delta")["weight"]
    assert torch.equal(model.weight, initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=spec.lr, foreach=False, fused=False)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        chunk = batch.slice(step, step + 1)
        model(chunk.images, chunk.questions, chunk.targets).backward()
        optimizer.step()
    torch.testing.assert_close(actual, model.weight.detach() - initial, rtol=0, atol=0)


def test_functional_sgd_preserves_differentiable_replay():
    model = QuadraticClient(torch.float64)
    spec = TrainingSpec(algorithm="fedavg", local_steps=2, lr=.013)
    images = torch.linspace(-2, 2, 256, dtype=torch.float64).reshape(2, 128).requires_grad_()
    batch = Batch(images, torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long))
    delta = _simulate_local_update(model, batch, spec, "client_delta", differentiable=True)["weight"]
    gradient, = torch.autograd.grad(delta.square().sum(), images)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
