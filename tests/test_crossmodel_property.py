import torch
import pytest

from evaluation.crossmodel_property import METHODS, module_key, score_upload, check_lengths


def test_crossmodel_mapping_uses_real_last_full_attention_with_same_input_geometry():
    assert (
        module_key("qwen35", "Q32")
        == "backend.model.language_model.layers.31.self_attn.q_proj.lora_A.default.weight"
    )
    assert (
        module_key("llava", "K24")
        == "backend.language_model.model.layers.23.self_attn.k_proj.lora_A.default.weight"
    )
    keys = {method: module_key("qwen35", method) for method in METHODS}
    prototypes = {method: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) for method in METHODS}
    tensors = {
        key: torch.tensor([[0.0, 2.0, 0.0]])
        if method.startswith("K")
        else torch.tensor([[3.0, 0.0, 0.0]])
        for method, key in keys.items()
    }
    result = score_upload(tensors, prototypes, keys)
    assert result["Q32"] == [0.0, 1.0] and result["K32"] == [1.0, 0.0]
    assert result == score_upload({k: -3 * v for k, v in tensors.items()}, prototypes, keys)


def test_original_qa_capacity_guard_never_truncates():
    from types import SimpleNamespace

    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: text.split())
    adapter = SimpleNamespace(
        tokenizer=tokenizer, spec=SimpleNamespace(question_length=4, target_length=3)
    )
    assert check_lengths(adapter, "what is this", "an image") == [3, 2]
    with pytest.raises(ValueError, match="never truncate"):
        check_lengths(adapter, "what is this image", "yes")


def test_fixed_qa_control_only_swaps_opposite_attributes_within_modality():
    from evaluation.crossmodel_property_report import fixed_swap_control

    truth = [
        {"modality": "CT", "label": "head"},
        {"modality": "MRI", "label": "head"},
        {"modality": "CT", "label": "non-head"},
        {"modality": "X-Ray", "label": "non-head"},
    ]
    result = fixed_swap_control(truth, [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
    assert result["pairs_by_modality"] == {"CT": 1, "MRI": 0, "X-Ray": 0}
    assert result["paired_images"] == 2
    assert result["correct_gradient"]["correct"] == 2
    assert result["swapped_gradient"]["correct"] == 0


def test_cached_identity_capture_preserves_original_native_upload_and_private_surface():
    from dataclasses import asdict
    from core.config import ModelSpec, TrainingSpec
    from core.vlm_wrapper import build_model
    from core.fl import capture
    from evaluation.crossmodel_property import capture_fixed_model, validate_observation

    adapter = build_model(ModelSpec(), TrainingSpec(knowledge="private", token_lengths_known=False))
    batch = adapter.batch(
        torch.rand(1, 3, adapter.spec.image_size, adapter.spec.image_size), ["what"], ["yes"]
    )
    fingerprint = adapter.fingerprint()
    original = capture(adapter, batch, [], [])
    cached = capture_fixed_model(adapter, batch, fingerprint)
    assert cached.tensors.keys() == original.tensors.keys()
    assert all(torch.equal(v, original.tensors[k]) for k, v in cached.tensors.items())
    assert asdict(cached.training) == asdict(original.training)
    validate_observation(cached, fingerprint)
    assert adapter.fingerprint() == fingerprint
