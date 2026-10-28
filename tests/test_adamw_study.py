from dataclasses import replace

import pytest
import torch

from attacks.analytic.dager_adapted import dager_support
from core.config import TrainingSpec, validate_attack_config
from core.fl import _simulate_local_update, capture
from core.types import Batch
from evaluation.adamw_text_study import attack_spec, evaluate_all, native_smoke, set_precision
from core.artifacts import write_json
from tests.test_dager import run, specification
from tests.test_dager_pooled import pooled_fixture
from tests.test_sgd_precision import QuadraticClient
from tests.test_fp32_multistep_study import fp32_fixture


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [1, 2])
def test_adamw_replay_matches_native_optimizer(dtype, steps):
    model = QuadraticClient(dtype)
    spec = TrainingSpec(algorithm="fedavg", local_optimizer="adamw", local_steps=steps,
                        lr=1e-4, weight_decay=0., batch_size=1)
    images = torch.linspace(-2, 2, steps * 128, dtype=dtype).reshape(steps, 128)
    batch = Batch(images, torch.zeros(steps, 1, dtype=torch.long), torch.zeros(steps, 1, dtype=torch.long))
    before = model.weight.detach().clone()
    update = _simulate_local_update(model, batch, spec, "client_delta")["weight"]
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=0.,
                                 betas=(spec.adam_beta1, spec.adam_beta2), eps=spec.adam_epsilon,
                                 foreach=False, fused=False)
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        part = batch.slice(index, index + 1)
        model(part.images, part.questions, part.targets).backward()
        optimizer.step()
    torch.testing.assert_close(update, model.weight.detach() - before, rtol=0, atol=0)


def test_adamw_replay_preserves_higher_order_gradients():
    model = QuadraticClient(torch.float64)
    spec = TrainingSpec(algorithm="fedavg", local_optimizer="adamw", local_steps=2, lr=1e-4)
    images = torch.linspace(-2, 2, 256, dtype=torch.float64).reshape(2, 128).requires_grad_()
    batch = Batch(images, torch.zeros(2, 1, dtype=torch.long), torch.zeros(2, 1, dtype=torch.long))
    update = _simulate_local_update(model, batch, spec, "client_delta", differentiable=True)["weight"]
    derivative, = torch.autograd.grad(update.square().sum(), images)
    assert torch.isfinite(derivative).all() and derivative.norm() > 0


def test_adamw_normalization_does_not_preserve_gradient_row_space():
    gradient = torch.tensor([[1., 2., 3.], [2., 4., 6.]], dtype=torch.float64)
    update = -1e-4 * gradient / (gradient.abs() + 1e-8)
    basis = gradient[0] / gradient[0].norm()
    residual = update - (update @ basis)[:, None] * basis
    assert residual.norm() / update.norm() > .3


def test_adamw_proposals_are_opt_in_and_require_real_optimizer_replays():
    adapter, batch, _ = pooled_fixture("fedavg", 2)
    adapter.set_training_spec(replace(adapter.training_spec, local_optimizer="adamw", lr=1e-4))
    obs = capture(adapter, batch, [], [])
    default = specification(pooled_updates=True)
    assert dager_support(adapter, obs, default.dager).status == "not_applicable"
    invalid = specification(pooled_updates=True, adamw_delta_proposals=True)
    with pytest.raises(ValueError, match="replay reranking"):
        validate_attack_config(invalid)
    spec = specification(pooled_updates=True, adamw_delta_proposals=True, rerank_candidates=2,
                         token_selection="topk", max_candidates=25, beam_width=2)
    assert dager_support(adapter, obs, spec.dager).status == "supported"
    result = run(adapter, obs, spec)
    assert result.status == "completed" and result.costs["full_gradient_replays"] == 2
    assert not result.provenance["gradient_subspace_premise"]
    assert result.provenance["optimizer_replay"] == "adamw"


@pytest.mark.parametrize("mode", ["mixed", "bf16"])
def test_native_public_smoke_is_exact_and_restores_original_values(mode):
    adapter, _, training, names = fp32_fixture()
    adapter.bfloat16()
    adapter.spec = replace(adapter.spec, dtype="bfloat16")
    set_precision(adapter, mode)
    initial = {k: p.detach().clone() for k, p in adapter.named_parameters()}
    training = replace(training, local_optimizer="adamw", lr=1e-4)
    result = native_smoke(adapter, training, names)
    assert result["status"] == "passed" and result["native_optimizer_exact_match"]
    assert result["trace"][1]["selected_A_distance_from_initial"] > 0
    assert result["moment_dtypes"] == ["torch.float32" if mode == "mixed" else "torch.bfloat16"]
    for name, p in adapter.named_parameters():
        torch.testing.assert_close(p, initial[name], rtol=0, atol=0)


def test_adamw_study_never_opens_references_before_all_reconstructions(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.adamw_text_study.verify_plan", lambda *args: {})
    write_json(tmp_path / "plan.json", {"jobs": [{"id": "missing"}]})
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
    for adamw in (True, False):
        spec = attack_spec("random", adamw)
        assert spec.dager.rerank_candidates == 16 and spec.dager.adamw_delta_proposals == adamw
