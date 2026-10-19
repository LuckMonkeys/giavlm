"""Offline Qwen3.5 gradient and prefix replay checks in the new environment."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")

from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration  # noqa: E402

from attacks.analytic.dager_adapted import dager_support, gradient_groups  # noqa: E402
from attacks.analytic.dager_subspace import row_basis  # noqa: E402
from core.adapters.qwen35 import Qwen35Adapter, Qwen35TextView, qwen35_input_pieces  # noqa: E402
from core.adapters.tiny_llava import TinyTokenizer  # noqa: E402
from core.config import DAGEROptions, ModelSpec, TrainingSpec  # noqa: E402
from core.fl import capture  # noqa: E402


def fixture(knowledge="image_known"):
    config = Qwen3_5Config(
        text_config={"vocab_size": 27, "hidden_size": 48, "intermediate_size": 64,
                     "num_hidden_layers": 4, "num_attention_heads": 6, "num_key_value_heads": 2,
                     "head_dim": 8, "pad_token_id": 0, "eos_token_id": 2,
                     "linear_key_head_dim": 4, "linear_value_head_dim": 4,
                     "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.,
                                         "partial_rotary_factor": 1., "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "patch_size": 4, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "out_hidden_size": 48, "num_position_embeddings": 16},
        image_token_id=26, vision_start_token_id=24, vision_end_token_id=25)
    config._attn_implementation = "eager"
    backend = Qwen3_5ForConditionalGeneration(config)
    training = TrainingSpec(knowledge=knowledge, token_lengths_known=True,
                            lora_rank=8, lora_alpha=16)
    processor = SimpleNamespace(tokenizer=TinyTokenizer(), image_processor=SimpleNamespace(
        image_mean=[.5] * 3, image_std=[.5] * 3))
    adapter = Qwen35Adapter(ModelSpec(family="qwen3_5", revision="offline-test"), training,
                            backend=backend, processor=processor)
    adapter.configure_training()
    with torch.no_grad():
        for name, parameter in adapter.named_parameters():
            if ".lora_B." in name:
                parameter.normal_(std=.03)
    batch = adapter.batch(torch.rand(1, 3, 8, 8), ["what color"], ["red circle"])
    observation = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
    return adapter, batch, observation


@pytest.mark.parametrize("knowledge", ["image_known", "image_question_known"])
def test_qwen35_partial_inputs_match_full_forward_and_gradient_identity(knowledge):
    adapter, batch, obs = fixture(knowledge)
    view = Qwen35TextView(adapter, obs)
    captured = {}
    modules = [adapter.backend.model.language_model.layers[i].linear_attn.in_proj_qkv for i in (0, 1)]
    handles = []
    for i, module in enumerate(modules):
        def record(module, args, i=i):
            captured[i] = args[0].detach().clone()
            captured[i].requires_grad_(False)
        handles.append(module.register_forward_pre_hook(record))
    try:
        loss = adapter(batch.images, batch.questions, batch.targets)
        assert torch.isfinite(loss)
    finally:
        for handle in handles:
            handle.remove()
    truth = {"questions": batch.questions[0], "targets": batch.targets[0]}
    prefix = [int(truth[field][index]) for field, index, _ in view.slots]
    for depth in range(1, len(prefix) + 1):
        torch.testing.assert_close(view.prefix_features([prefix[:depth]]),
                                   captured[1][:, view.slots[depth - 1][2]], atol=2e-5, rtol=2e-4)
    for field, ids in view.token_ids(prefix).items():
        torch.testing.assert_close(ids, getattr(batch, field))
    visual, grid = adapter.visual_embeddings(batch.images)
    pieces = qwen35_input_pieces(adapter, visual, adapter.embedding()(batch.questions),
                                adapter.embedding()(batch.targets[:, :-1]))
    positions = adapter.position_ids(pieces, grid)
    assert positions.shape == (3, 1, view.embeddings.shape[1])
    full = torch.cat(pieces, 1)
    for layer in (0, 1):
        torch.testing.assert_close(view.layer_input(full, layer), captured[layer], atol=2e-5, rtol=2e-4)
        # Every uploaded LoRA-A row is a linear combination of this module's inputs.
        gradient = obs.tensors[gradient_groups(adapter, obs, "linear_inputs")[layer][0]].double()
        basis, _ = row_basis(captured[layer][0].double(), 1e-5, 1e-8)
        torch.testing.assert_close(gradient - gradient @ basis.T @ basis,
                                   torch.zeros_like(gradient), atol=2e-5, rtol=0)
    options = DAGEROptions(projections="linear_inputs")
    assert dager_support(adapter, obs, options).status == "supported"
    assert dager_support(adapter, obs, replace(options, projections="qkv")).status == "not_applicable"
    assert all(adapter.is_language(name) for name in adapter.trainable())
    assert not any("visual" in name for name in adapter.trainable())

    # Check the actual chain rule, rather than only containment in a broad span.
    module = modules[0]
    module.base_layer.weight.requires_grad_(True)
    try:
        loss = adapter(batch.images, batch.questions, batch.targets)
        ga, gw = torch.autograd.grad(loss, (module.lora_A.default.weight, module.base_layer.weight))
        expected = module.scaling["default"] * module.lora_B.default.weight.T @ gw
        torch.testing.assert_close(ga, expected, atol=2e-6, rtol=2e-4)
    finally:
        module.base_layer.weight.requires_grad_(False)


def test_qwen35_prior_uses_only_parent_and_aligns_full_vocab_nll():
    adapter, _, obs = fixture()
    view = Qwen35TextView(adapter, obs)
    candidates = torch.tensor([3, 4, 5])
    prefix = [3] * view.question_content_length
    contexts = view.text_prior_ids([prefix])
    encoded = adapter.tokenizer.encode
    boundary = encoded("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    assert contexts[0, -len(boundary):].tolist() == boundary
    assert view.text_prior_ids([prefix + [4]])[0, -1].item() == 4
    nll, cost = view.text_prior_nll([prefix], candidates)
    with torch.no_grad():
        hidden = view.decoder(input_ids=contexts, use_cache=False).last_hidden_state[:, -1]
        expected = -adapter.backend.lm_head(hidden).float().log_softmax(-1)[:, candidates]
    torch.testing.assert_close(nll, expected)
    assert cost == contexts.numel()
