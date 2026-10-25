from dataclasses import replace

import pytest
import torch

from core.artifacts import write_json
from evaluation.sgd_precision_study import evaluate, fixed_gradient_update, numeric_audit


def test_fixed_gradient_controls_distinguish_parameter_rounding():
    weights = {"w": torch.tensor([1., -.02, .003], dtype=torch.bfloat16)}
    gradients = {"w": torch.tensor([.1, .001, -.001], dtype=torch.bfloat16)}
    original = weights["w"].clone()
    ideal = fixed_gradient_update(weights, gradients, "ideal-fp64", .01)
    bf16 = fixed_gradient_update(weights, gradients, "sgd-bf16", .01)
    fp32 = fixed_gradient_update(weights, gradients, "sgd-fp32", .01)
    torch.testing.assert_close(ideal["w"], -.01 * gradients["w"].double(), rtol=0, atol=0)
    assert bf16["w"][0] == 0 and fp32["w"][0] != 0
    assert fp32["w"].dtype == torch.float32
    assert torch.equal(weights["w"], original)
    assert numeric_audit(gradients, fp32, ["w"], .01)["selected_A_relative_l2_vs_ideal"] < .001
    with pytest.raises(ValueError):
        fixed_gradient_update(weights, gradients, "unknown", .01)


def test_real_capture_agrees_with_actual_optimizer_and_keeps_differentiability():
    from tests.test_sgd_precision import QuadraticClient
    from core.config import TrainingSpec
    from core.fl import _simulate_local_update
    from core.types import Batch

    model = QuadraticClient(torch.bfloat16)
    spec = TrainingSpec(algorithm="fedsgd", batch_size=1, local_steps=1, lr=.013)
    batch = Batch(torch.linspace(-2, 2, 128, dtype=torch.bfloat16).reshape(1, 128),
                  torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long))
    gradient = _simulate_local_update(model, batch, spec, "gradient")
    delta = _simulate_local_update(model, batch, replace(spec, algorithm="fedavg"), "client_delta")
    control = fixed_gradient_update(model.trainable(), gradient, "sgd-bf16", spec.lr)
    torch.testing.assert_close(delta["weight"], control["weight"], rtol=0, atol=0)


def test_precision_eval_requires_every_result_before_references(tmp_path, monkeypatch):
    write_json(tmp_path / "plan.json", {"jobs": [{"id": "x"}]})
    def forbidden(*args, **kwargs):
        raise AssertionError("Private references accessed early")
    monkeypatch.setattr("evaluation.sgd_precision_study.shutil.copyfile", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate(tmp_path)
