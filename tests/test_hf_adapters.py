from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from giavlm.attacks import Candidate, matching_loss
from giavlm.config import ModelSpec, TrainingSpec
from giavlm.federated import capture, simulate_update
from giavlm.models import HFAdapter, TinyTokenizer


def hf_fixture(family, mode):
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
    training = TrainingSpec(mode=mode)
    adapter = HFAdapter(ModelSpec(family=family, revision="test-fixture"), training,
                         backend=model, processor=processor)
    adapter.configure_training()
    return adapter


@pytest.mark.parametrize("family", ["llava", "blip2", "qwen2_5_vl"])
@pytest.mark.parametrize("mode", ["full", "lora_llm"])
def test_actual_hf_architectures_allow_second_order(family, mode):
    adapter = hf_fixture(family, mode)
    batch = adapter.batch(torch.rand(1, 3, 8, 8), ["what color is the object"], ["red"])
    obs = capture(adapter, batch, [], [])
    candidate = Candidate(adapter, obs, 33)
    predicted = simulate_update(adapter, candidate.batch(), adapter.training_spec, True)
    objective = matching_loss(predicted, obs.tensors, "l2")
    grads = torch.autograd.grad(objective, tuple(candidate.parameters()))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
    if mode == "lora_llm":
        assert all("lora_" in name and adapter.is_language(name) for name in obs.tensors)


def test_qwen_patchification_matches_processor():
    from transformers import Qwen2VLImageProcessor
    adapter = hf_fixture("qwen2_5_vl", "full")
    processor = Qwen2VLImageProcessor(patch_size=2, temporal_patch_size=2, merge_size=2,
                                       min_pixels=64, max_pixels=64, do_resize=False,
                                       do_rescale=False, do_normalize=False)
    images = torch.rand(2, 3, 8, 8)
    native = processor(images=list(images), return_tensors="pt")
    pixels, grid = adapter.qwen_pixels(images)
    torch.testing.assert_close(pixels, native.pixel_values)
    torch.testing.assert_close(grid, native.image_grid_thw)


@pytest.mark.parametrize("family", ["llava", "blip2", "qwen2_5_vl"])
def test_hf_multistep_lora_replay(family):
    adapter = hf_fixture(family, "lora_llm")
    adapter.training_spec = replace(adapter.training_spec, observation="client_delta", local_steps=2)
    batch = adapter.batch(torch.rand(2, 3, 8, 8), ["what color", "what shape"], ["red", "circle"])
    obs = capture(adapter, batch, [], [])
    candidate = Candidate(adapter, obs, 15)
    predicted = simulate_update(adapter, candidate.batch(), adapter.training_spec, True)
    loss = matching_loss(predicted, obs.tensors, "l2")
    grads = torch.autograd.grad(loss, tuple(candidate.parameters()))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
