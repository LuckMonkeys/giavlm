from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from attacks import Candidate, matching_loss
from core.config import ModelSpec, TrainingSpec
from core.fl import capture, simulate_update
from core.adapters.hf import HFAdapter
from core.adapters.tiny_llava import TinyTokenizer


def hf_fixture(family, strategy):
    from transformers import (Blip2Config, Blip2ForConditionalGeneration, Blip2QFormerConfig,
                              Blip2VisionConfig, CLIPVisionConfig, LlamaConfig, LlavaConfig,
                              LlavaForConditionalGeneration, OPTConfig, Qwen2_5_VLConfig,
                              Qwen2_5_VLForConditionalGeneration)
    if family == "llava":
        vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, image_size=8, patch_size=4)
        text = LlamaConfig(vocab_size=27, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=2, pad_token_id=0,
                           bos_token_id=1, eos_token_id=2)
        config = LlavaConfig(vision_config=vision.to_dict(), text_config=text.to_dict(),
                             image_token_index=26, vision_feature_layer=-1)
        config._attn_implementation = "eager"
        model = LlavaForConditionalGeneration(config)
    elif family == "blip2":
        vision = Blip2VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                   num_attention_heads=2, image_size=8, patch_size=4,
                                   initializer_range=0.02)
        qformer = Blip2QFormerConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                     num_attention_heads=2, encoder_hidden_size=16)
        text = OPTConfig(vocab_size=27, hidden_size=16, ffn_dim=32, num_hidden_layers=1,
                          num_attention_heads=2, word_embed_proj_dim=16, pad_token_id=0,
                          bos_token_id=1, eos_token_id=2)
        config = Blip2Config(vision_config=vision.to_dict(), qformer_config=qformer.to_dict(),
                             text_config=text.to_dict(), num_query_tokens=2)
        config._attn_implementation = "eager"
        model = Blip2ForConditionalGeneration(config)
    else:
        config = Qwen2_5_VLConfig(vocab_size=27, hidden_size=16, intermediate_size=32,
                                  num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                                  vision_start_token_id=24, vision_end_token_id=25, image_token_id=26,
                                  video_token_id=23,
                                  pad_token_id=0, bos_token_id=1, eos_token_id=2,
                                  rope_scaling={"type": "default", "mrope_section": [1, 1, 2]},
                                  vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                                                 "num_heads": 2, "patch_size": 2, "spatial_merge_size": 2,
                                                 "temporal_patch_size": 2, "window_size": 8,
                                                 "out_hidden_size": 16, "fullatt_block_indexes": [0]})
        config._attn_implementation = "eager"
        model = Qwen2_5_VLForConditionalGeneration(config)
    processor = SimpleNamespace(tokenizer=TinyTokenizer(),
                                 image_processor=SimpleNamespace(image_mean=[0.5] * 3, image_std=[0.5] * 3))
    training = TrainingSpec(fine_tuning_strategy=strategy)
    adapter = HFAdapter(ModelSpec(family=family, revision="test-fixture"), training,
                         backend=model, processor=processor)
    adapter.configure_training()
    return adapter


@pytest.mark.parametrize("family", ["llava", "blip2", "qwen2_5_vl"])
@pytest.mark.parametrize("strategy", ["f_c", "f_l", "f_cl", "f_2stage"])
def test_actual_hf_architectures_allow_second_order(family, strategy):
    adapter = hf_fixture(family, strategy)
    batch = adapter.batch(torch.rand(1, 3, 8, 8), ["what color is the object"], ["red"])
    obs = capture(adapter, batch, [], [])
    candidate = Candidate(adapter, obs, 33)
    predicted = simulate_update(adapter, candidate.batch(), adapter.training_spec, True)
    objective = matching_loss(predicted, obs.tensors, "l2")
    grads = torch.autograd.grad(objective, tuple(candidate.parameters()))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    names = set(obs.tensors)
    assert any(adapter.is_connector(name) for name in names) == (strategy != "f_l")
    assert any(".lora_" in name for name in names) == (strategy in {"f_l", "f_cl"})
    if strategy == "f_l":
        assert all("lora_" in name and adapter.is_language(name) for name in obs.tensors)
        targets = {name.split(".lora_", 1)[0] for name in obs.tensors}
        assert any(any(part in name for part in ("mlp", "fc1", "fc2")) for name in targets)
        assert not any(any(part in name.lower() for part in (
            "vision", "visual", "qformer", "projector", "lm_head")) for name in targets)


def test_qwen_patchification_matches_processor():
    from transformers import Qwen2VLImageProcessor
    adapter = hf_fixture("qwen2_5_vl", "f_c")
    processor = Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                       min_pixels=64, max_pixels=64, do_resize=False,
                                       do_rescale=False, do_normalize=False)
    images = torch.rand(2, 3, 8, 8)
    native = processor(images=list(images), return_tensors="pt")
    pixels, grid = adapter.qwen_pixels(images)
    torch.testing.assert_close(pixels, native.pixel_values)
    torch.testing.assert_close(grid, native.image_grid_thw)


@pytest.mark.parametrize("family", ["llava", "blip2"])
def test_fixed_rgb_view_matches_hf_image_processor(family):
    import numpy as np
    from PIL import Image
    from transformers import CLIPImageProcessor

    adapter = hf_fixture(family, "f_c")
    processor = CLIPImageProcessor(size={"height": 8, "width": 8},
                                   crop_size={"height": 8, "width": 8})
    adapter.processor.image_processor = processor
    image = Image.fromarray(np.arange(10 * 14 * 3, dtype=np.uint8).reshape(10, 14, 3))
    expected = processor.preprocess(images=image, return_tensors="pt",
                                    do_normalize=False)["pixel_values"][0]
    torch.testing.assert_close(adapter.prepare_image(image), expected)


def test_qwen_rmsnorm_is_excluded_from_weight_decay():
    adapter = hf_fixture("qwen2_5_vl", "f_c")
    decay = adapter.decay_parameter_names()
    norm_names = {name for name in adapter.trainable() if name.endswith("ln_q.weight")}
    assert norm_names
    assert not norm_names & decay


@pytest.mark.parametrize("family", ["llava", "blip2", "qwen2_5_vl"])
def test_hard_response_loss_matches_labels_cross_entropy(family):
    adapter = hf_fixture(family, "f_cl")
    batch = adapter.batch(torch.rand(1, 3, 8, 8), ["what color"], ["red"])
    questions = adapter.probabilities(batch.questions)
    targets = adapter.probabilities(batch.targets)
    logits = adapter.target_logits(batch.images, questions, targets)
    expected = F.cross_entropy(logits.float().flatten(0, 1), batch.targets.flatten(),
                               ignore_index=adapter.pad)
    torch.testing.assert_close(adapter(batch.images, batch.questions, batch.targets), expected)


@pytest.mark.parametrize("family", ["llava", "blip2", "qwen2_5_vl"])
@pytest.mark.parametrize("optimizer", ["sgd", "adamw"])
def test_hf_multistep_lora_replay(family, optimizer):
    adapter = hf_fixture(family, "f_l")
    adapter.training_spec = replace(adapter.training_spec, algorithm="fedavg", local_steps=2,
                                    local_optimizer=optimizer)
    batch = adapter.batch(torch.rand(2, 3, 8, 8), ["what color", "what shape"], ["red", "circle"])
    obs = capture(adapter, batch, [], [])
    candidate = Candidate(adapter, obs, 15)
    predicted = simulate_update(adapter, candidate.batch(), adapter.training_spec, True)
    loss = matching_loss(predicted, obs.tensors, "l2")
    grads = torch.autograd.grad(loss, tuple(candidate.parameters()))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
