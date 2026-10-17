"""Prior alignment, image isolation, score accounting, and restart semantics."""
from dataclasses import replace

import pytest
import torch

from core.adapters.llava import LlavaTextView
from core.artifacts import read_json, write_json
from core.config import validate_attack_config
from test_dager import fixture, run, specification


def test_text_prior_is_next_token_aligned_and_independent_of_image(monkeypatch):
    torch.manual_seed(84)
    adapter, batch, obs = fixture(knowledge="image_known")
    view = LlavaTextView(adapter, obs)
    q = batch.questions[0, :obs.public_question_lengths[0]].tolist()
    candidates = [4, 5, 6]
    parents = [q[:-1], q[:-1]]
    ids = view.text_prior_ids(parents)
    expected_ids = (adapter.tokenizer.encode("USER: ", add_special_tokens=False)
                    + adapter.tokenizer.encode("\n", add_special_tokens=False) + q[:-1])
    assert ids.tolist() == [expected_ids, expected_ids]
    with torch.no_grad():
        logits = adapter.backend.language_model(input_ids=ids, use_cache=False).logits[:, -1].float()
    expected = -logits.log_softmax(-1)[:, candidates]
    observed, count = view.text_prior_nll(parents, candidates)
    torch.testing.assert_close(observed, expected)
    assert count == ids.numel()
    other = LlavaTextView(adapter, replace(obs, public_images=1-obs.public_images))
    torch.testing.assert_close(observed, other.text_prior_nll(parents, candidates)[0], rtol=0, atol=0)
    def no_visual(*args):
        raise AssertionError("Prior read image features")
    monkeypatch.setattr(adapter, "visual_embeddings", no_visual)
    torch.testing.assert_close(observed, view.text_prior_nll(parents, candidates)[0])


def test_prior_inserts_answer_boundary_and_uses_only_declared_question():
    adapter, batch, obs = fixture(knowledge="image_known")
    view = LlavaTextView(adapter, obs)
    q = batch.questions[0, :obs.public_question_lengths[0]].tolist()
    y = batch.targets[0, :1].tolist()
    expected = (adapter.tokenizer.encode("USER: ", add_special_tokens=False)
                + adapter.tokenizer.encode("\n", add_special_tokens=False) + q
                + adapter.tokenizer.encode(" ASSISTANT: ", add_special_tokens=False) + y)
    assert view.text_prior_ids([q+y]).tolist() == [expected]
    public_obs = replace(obs, training=replace(obs.training, knowledge="image_question_known"),
                         public_question_ids=batch.questions.tolist())
    assert LlavaTextView(adapter, public_obs).text_prior_ids([y]).tolist() == [expected]
    with pytest.raises(ValueError, match="unfinished"):
        view.text_prior_ids([q, q+y])


def test_zero_weight_never_calls_prior_and_nonzero_scores_are_auditable(tmp_path, monkeypatch):
    torch.manual_seed(73)
    adapter, _, obs = fixture(answer="red")
    def forbidden(*args):
        raise AssertionError("Zero-weight search called the language prior")
    with monkeypatch.context() as patch:
        patch.setattr(LlavaTextView, "text_prior_nll", forbidden)
        baseline = run(adapter, obs, specification(), directory=tmp_path/"zero")
    assert baseline.costs["lm_prior_prefixes"] == 0
    weight = 0.01
    result = run(adapter, obs, specification(lm_prior_weight=weight), directory=tmp_path/"prior")
    pointer = read_json(tmp_path/"prior/checkpoint.json")
    state = read_json(tmp_path/"prior/checkpoints"/pointer["generation"]/"state.json")["state"]
    assert result.costs["lm_prior_prefixes"] > 0
    assert result.provenance["final_score_type"] == "span_distance_with_text_prior"
    for row in state["beam"]:
        assert row["score"] == pytest.approx(row["span_score"] + weight*row["lm_nll"])
    assert result.costs["prefix_candidates_evaluated"] == baseline.costs["prefix_candidates_evaluated"]


def test_prior_checkpoint_preserves_parent_scores_during_resume(tmp_path, monkeypatch):
    torch.manual_seed(65)
    adapter, _, obs = fixture()
    spec = specification(lm_prior_weight=0.01, prefix_batch_size=1)
    expected = run(adapter, obs, spec)
    original = LlavaTextView.prefix_features
    calls = 0
    def interrupt(self, prefixes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("prior interruption")
        return original(self, prefixes)
    with monkeypatch.context() as patch:
        patch.setattr(LlavaTextView, "prefix_features", interrupt)
        with pytest.raises(RuntimeError, match="prior interruption"):
            run(adapter, obs, spec, directory=tmp_path)
    resumed = run(adapter, obs, spec, directory=tmp_path, resume=True)
    assert (resumed.questions, resumed.targets) == (expected.questions, expected.targets)
    assert resumed.provenance["final_score"] == expected.provenance["final_score"]
    assert resumed.costs["lm_prior_prefixes"] == expected.costs["lm_prior_prefixes"]


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_invalid_prior_weight_rejected(weight):
    with pytest.raises(ValueError, match="lm_prior_weight"):
        validate_attack_config(specification(lm_prior_weight=weight))


def test_prior_parent_forwards_consume_budget(tmp_path):
    adapter, _, obs = fixture()
    spec = replace(specification(lm_prior_weight=0.01), max_evaluations=1)
    result = run(adapter, obs, spec, directory=tmp_path)
    assert result.status == "budget_exhausted"
    assert result.costs["lm_prior_prefixes"] == 1
    assert result.costs["prefix_candidates_evaluated"] == 0


def test_saved_beam_audit_distinguishes_pruning_from_ranking(tmp_path):
    from evaluation.dager_lm_prior_study import beam_audit
    torch.manual_seed(94)
    adapter, batch, obs = fixture(knowledge="image_known", answer="red")
    result = run(adapter, obs, specification(), directory=tmp_path)
    write_json(tmp_path/"result.json", {"status": result.status})
    q = batch.questions[0, :obs.public_question_lengths[0]].tolist()
    a = batch.targets[0, :obs.public_target_lengths[0]].tolist()
    audit = beam_audit(tmp_path, q, a)
    assert audit["full_coverage"]
    assert audit["final_joint_hit"] == (audit["first_drop_depth"] is None)
    assert audit["question_length"] == len(q)


def test_weight_selection_requires_em_gain_and_prefers_smaller_weight():
    from evaluation.dager_lm_prior_study import select_weight
    rows = [{"weight": w, "joint_em": 0.0, "question_em": 0.0, "answer_em": 0.3}
            for w in (0.0, 0.001, 0.01)]
    assert select_weight(rows) is None
    rows[1]["question_em"] = rows[2]["question_em"] = 0.2
    assert select_weight(rows) == 0.001
    rows[2]["joint_em"] = 0.1
    assert select_weight(rows) == 0.01
    assert select_weight([
        {"weight": 0., "joint_em": 0., "question_em": 0., "answer_em": .3},
        {"weight": .001, "joint_em": 0., "question_em": .1, "answer_em": .2},
    ]) is None


def test_rank8_study_freezes_weight_without_reading_development(tmp_path):
    from evaluation.dager_lm_prior_study import study_inputs, study_weights, SOURCE, MODEL, WEIGHTS
    source, model = study_inputs(8)
    assert str(source) == "outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100"
    assert str(model) == "outputs/federation/slake_llava_f_l_fedavg/round-0010"
    assert study_inputs(32) == (SOURCE, MODEL)
    assert study_weights("development", None, None) == WEIGHTS
    for phase in ("development", "validation"):
        assert study_weights(phase, .01, tmp_path/"absent") == (0.0, .01)
    for invalid in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            study_weights("development", invalid, None)


def test_f_l_capture_round_counter_does_not_change_uploaded_gradient():
    from core.fl import capture
    torch.manual_seed(125)
    adapter, batch, before = fixture(knowledge="private", strategy="f_l")
    original = adapter.training_spec
    adapter.set_training_spec(replace(original, server_round=10))
    after = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
    assert original.fine_tuning_stage == adapter.training_spec.fine_tuning_stage == "llm"
    assert before.training.server_round == 0 and after.training.server_round == 10
    assert before.model_fingerprint == after.model_fingerprint
    assert before.tensors.keys() == after.tensors.keys()
    for name in before.tensors:
        torch.testing.assert_close(before.tensors[name], after.tensors[name], rtol=0, atol=0)
