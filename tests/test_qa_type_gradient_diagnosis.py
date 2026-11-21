import torch

from evaluation.qa_type_gradient_diagnosis import spectrum


def test_bfloat16_rounding_expands_formal_rank_without_many_energy_directions():
    generator = torch.Generator().manual_seed(16)
    value = torch.randn(32, 3, generator=generator) @ torch.randn(3, 128, generator=generator)
    quantized = value.to(torch.bfloat16)
    summary, bases = spectrum(quantized, 3)
    assert summary["original_rank"] > 3
    assert summary["energy_0p9999_rank"] <= 3
    assert summary["tail_energy_after_supervised"] < 1e-4
    scaled, _ = spectrum(quantized.double() * 17, 3)
    assert abs(summary["top1_energy"] - scaled["top1_energy"]) < 1e-12
    assert bases["public_supervised_rank"].shape == (3, 128)


def test_last_layer_query_gradient_is_supported_only_at_loss_positions():
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
                         attention_dropout=0., use_cache=False)
    config._attn_implementation = "eager"
    torch.manual_seed(32)
    model = LlamaForCausalLM(config).double().eval()
    captured, handles = {}, []
    for layer in (0, 1):
        def hook(module, inputs, output, key=layer):
            captured[key] = output
        handles.append(model.model.layers[layer].self_attn.q_proj.register_forward_hook(hook))
    for name in ("k_proj", "v_proj"):
        def hook(module, inputs, output, key=name):
            captured[key] = output
        handles.append(getattr(model.model.layers[1].self_attn, name).register_forward_hook(hook))
    try:
        logits = model(input_ids=torch.arange(9)[None], use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits[0, -3:], torch.tensor([9, 10, 11]))
        first, last, weight, key, value = torch.autograd.grad(loss, (
            captured[0], captured[1], model.model.layers[1].self_attn.q_proj.weight,
            captured["k_proj"], captured["v_proj"]))
    finally:
        for handle in handles:
            handle.remove()
    assert first[:, :-3].norm() > 0
    assert torch.count_nonzero(last[:, :-3]) == 0
    assert last[:, -3:].norm() > 0
    assert int(torch.linalg.matrix_rank(weight, atol=1e-12)) <= 3
    assert key[:, :-3].norm() > 0
    assert value[:, :-3].norm() > 0
