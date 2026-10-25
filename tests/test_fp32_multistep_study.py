from dataclasses import replace

import pytest
import torch

from core.artifacts import write_json
from evaluation.fp32_multistep_study import (
    evaluate_all, precision_context, precision_settings, require_fp32, require_mixed,
    restore_fp32, restore_mixed, trace_local_update,
)
from tests.test_dager import fixture


def fp32_fixture():
    adapter, _, _ = fixture(knowledge="image_known")
    adapter.float()
    adapter.spec = replace(adapter.spec, dtype="float32")
    training = replace(adapter.training_spec, algorithm="fedavg", local_steps=2, batch_size=1)
    adapter.set_training_spec(training)
    batch = adapter.batch(torch.rand(2, 3, 8, 8), ["what color", "what"], ["red", "blue"])
    selected = [name for name in adapter.trainable() if ".lora_A." in name][:6]
    return adapter, batch, training, selected


def test_trace_proves_second_gradient_is_at_updated_fp32_weights():
    torch.manual_seed(174)
    adapter, batch, training, names = fp32_fixture()
    before = {name: p.detach().clone() for name, p in adapter.trainable().items()}
    update, trace = trace_local_update(adapter, batch, training, names)
    assert len(trace) == 2 and trace[0]["selected_A_distance_from_initial"] == 0
    assert trace[1]["selected_A_distance_from_initial"] > 0
    for name, p in adapter.trainable().items():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    optimizer = torch.optim.SGD(adapter.trainable().values(), lr=training.lr, foreach=False, fused=False)
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        part = batch.slice(index, index + 1)
        adapter(part.images, part.questions, part.targets).backward()
        optimizer.step()
    for name, p in adapter.trainable().items():
        torch.testing.assert_close(p.detach() - before[name], update[name], rtol=0, atol=0)


def test_fp32_restore_preserves_original_bf16_rounded_values(monkeypatch):
    adapter, _, _, _ = fp32_fixture()
    adapter.bfloat16()
    adapter.spec = replace(adapter.spec, dtype="bfloat16")
    before = {name: p.detach().float().clone() for name, p in adapter.named_parameters()}
    monkeypatch.setattr("evaluation.fp32_multistep_study.restore_model", lambda *args: adapter)
    result = restore_fp32(None, "cpu")
    require_fp32(result)
    for name, p in result.named_parameters():
        torch.testing.assert_close(p.detach(), before[name], rtol=0, atol=0)
    result.half()
    with pytest.raises(ValueError, match="FP32"):
        require_fp32(result)


def test_fp32_mode_disables_tf32_and_evaluation_is_after_commits(tmp_path, monkeypatch):
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        settings = precision_settings()
        assert settings["float32_matmul_precision"] == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
    monkeypatch.setattr("evaluation.fp32_multistep_study.verify_plan", lambda *args: {})
    write_json(tmp_path / "plan.json", {"jobs": [{"id": "x"}]})
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)


def test_mixed_two_steps_match_native_sgd_without_rounding_master_weights(monkeypatch):
    torch.manual_seed(174)
    adapter, _, training, names = fp32_fixture()
    adapter.bfloat16()
    adapter.spec = replace(adapter.spec, dtype="bfloat16")
    initial = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    monkeypatch.setattr("evaluation.fp32_multistep_study.restore_model", lambda *args: adapter)
    adapter = restore_mixed(None, "cpu")
    require_mixed(adapter)
    for name, p in adapter.named_parameters():
        torch.testing.assert_close(p, initial[name].to(p.dtype), rtol=0, atol=0)
    batch = adapter.batch(torch.rand(2, 3, 8, 8), ["what color", "what"], ["red", "blue"])
    before = {name: p.detach().clone() for name, p in adapter.trainable().items()}
    with precision_context("mixed", "cpu"):
        assert torch.is_autocast_enabled("cpu") and not torch.is_autocast_cache_enabled()
        update, trace = trace_local_update(adapter, batch, training, names)
    assert len(trace) == 2 and trace[1]["selected_A_distance_from_initial"] > 0
    assert all(t["selected_projection_output_dtype"] == "torch.bfloat16" for t in trace)
    assert all(value.dtype == torch.float32 for value in update.values())
    assert any((before[name] + value).bfloat16().float().ne(before[name] + value).any()
               for name, value in update.items())
    for name, p in adapter.trainable().items():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    optimizer = torch.optim.SGD(adapter.trainable().values(), lr=training.lr, foreach=False, fused=False)
    for index in range(2):
        optimizer.zero_grad(set_to_none=True)
        part = batch.slice(index, index + 1)
        with precision_context("mixed", "cpu"):
            loss = adapter(part.images, part.questions, part.targets)
        loss.backward()
        assert all(p.grad is None or p.grad.dtype == torch.float32 for p in adapter.trainable().values())
        optimizer.step()
    for name, p in adapter.trainable().items():
        torch.testing.assert_close(p.detach() - before[name], update[name], rtol=0, atol=0)
    for name, p in adapter.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p, initial[name], rtol=0, atol=0)
