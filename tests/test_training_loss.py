from dataclasses import replace

import pytest
import torch

from core.commands import selected_client_loss
from core.config import ModelSpec, TrainingSpec
from core.vlm_wrapper import build_model


def test_logged_loss_matches_full_token_mean_without_changing_model():
    adapter = build_model(ModelSpec(dtype='float64'), TrainingSpec(batch_size=2))
    batch = adapter.batch(torch.rand(5, 3, 8, 8, dtype=torch.float64),
                          ['what color'] * 5,
                          ['red', 'red circle', 'red circle left', '', 'blue'])
    with torch.no_grad():
        expected = adapter(batch.images, batch.questions, batch.targets).item()
    fingerprint = adapter.fingerprint()
    calls = []
    handle = adapter.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    try:
        actual = selected_client_loss(adapter, batch)
    finally:
        handle.remove()
    assert calls == [2, 2, 1]
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-7)
    assert adapter.fingerprint() == fingerprint
    assert all(p.grad is None for p in adapter.parameters())
    adapter.set_training_spec(replace(adapter.training_spec, batch_size=1))
    assert selected_client_loss(adapter, batch) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_logged_loss_rejects_empty_batch():
    adapter = build_model(ModelSpec(), TrainingSpec())
    batch = adapter.batch(torch.rand(1, 3, 8, 8), ['what color'], ['red'])
    with pytest.raises(ValueError, match='empty'):
        selected_client_loss(adapter, batch.slice(0, 0))
