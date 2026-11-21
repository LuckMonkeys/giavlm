import torch

from evaluation.qa_type_controls import cases, collect_one, native_token_losses, token_subset


def test_factorial_cases_include_all_old_controls_and_predeclared_pairs():
    rows = cases()
    assert len(rows) == len({row["id"] for row in rows}) == 12
    by_id = {row["id"]: row for row in rows}
    assert by_id["present"]["q"] == by_id["present_long_yes"]["q"]
    assert by_id["description"]["a"] == by_id["present_description"]["a"] == by_id["examine_description"]["a"]
    assert {row["old_qa"] for row in rows if row["old_qa"]} == {"description", "visible", "present", "blank", "summary", "examine"}


def test_token_subset_is_family_disjoint_and_covers_all_public_anatomical_sources():
    bank = [{"group": group, "style": style} for group in ["head-09", "head-11", "non-head-09", "non-head-11",
            "abdomen-prompt-family-05", "abdomen-prompt-family-07"] for style in ["photo", "line", "render"]]
    rows = [bank[i] for i in token_subset(bank)]
    assert len({row["group"] for row in rows}) == 6
    assert {row["style"] for row in rows} == {"photo", "line", "render"}


def test_collect_actual_hf_qkv_and_token_gradients_with_native_loss(monkeypatch):
    import evaluation.qa_type_controls as controls
    from evaluation.visual_concept_probe import inject_logits, native_hard_loss
    from test_hf_adapters import hf_fixture
    monkeypatch.setattr(controls, "LAYERS", (1,))
    adapter = hf_fixture("llava", "f_l").eval()
    # A trained LoRA-B branch is required for nonzero LoRA-A gradients.
    with torch.no_grad():
        for name, value in adapter.trainable().items():
            if "lora_B" in name:
                value.normal_(std=.03)
        visual = adapter.visual_embeddings(torch.rand(1, 3, 8, 8))[0].detach()
    q, a = adapter.encode(["image"], adapter.spec.question_length), adapter.encode(["red"], adapter.spec.target_length)
    valid = a[0] != adapter.pad
    case = {"id": "test", "q": "image", "a": "red", "answer_token_ids": a[0, valid].tolist(),
            "token_text": ["red", "EOS"]}
    proto = {"head": {"visual_1": torch.nn.functional.normalize(torch.randn(2, 16), dim=1)}}
    row, tensors, tokens = collect_one(adapter, visual, case, proto, True)
    logits = inject_logits(adapter, visual, q, a)
    expected = native_hard_loss(logits, a, adapter.pad)
    assert row["loss"] == float(expected.detach())
    direct = torch.autograd.grad(expected, adapter.trainable()["backend.language_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"])[0]
    torch.testing.assert_close(tensors["L1_q"], direct, rtol=0, atol=0)
    assert all(all(value == 0 for value in match.values()) for match in row["interface"].values())
    assert row["loss_support"]["L1_q"]["nonzero_outside_supervised_positions"] == 0
    assert row["loss_support"]["L1_k"]["nonzero_visual_positions"] > 0
    assert row["loss_support"]["L1_v"]["nonzero_visual_positions"] > 0
    assert len(tokens) == 6
    assert all(torch.isfinite(value).all() for value in tokens.values())
    mask = a != adapter.pad
    assert torch.equal((native_token_losses(logits, a) * mask).sum() / mask.sum(), expected)
