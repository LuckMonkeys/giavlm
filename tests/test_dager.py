"""Offline mathematical and interface tests; no pretrained weights or private data."""
from dataclasses import asdict, replace
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest
import torch

from attacks.analytic.dager_adapted import (DAGERSearch, dager_support, forbidden_token_ids,
                                            gradient_groups)
from attacks.analytic.dager_subspace import SpanFilter, residual, row_basis, select_tokens
from attacks.factory import create_attacker
from core.adapters.hf import HFAdapter
from core.adapters.llava import LlavaTextView, llava_input_pieces
from core.adapters.tiny_llava import TinyTokenizer
from core.artifacts import read_json, read_tensors, write_json, write_tensors
from core.config import AttackSpec, DAGEROptions, ModelSpec, TrainingSpec, load_config
from core.experiment import protocol_config
from core.fl import capture
from core.knowledge import AdversaryKnowledge


def fixture(knowledge="image_question_known", answer="red circle", question="what color", zero=False,
            strategy="f_l"):
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, image_size=8, patch_size=4)
    text = LlamaConfig(vocab_size=27, hidden_size=48, intermediate_size=64, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=4, pad_token_id=0,
                       bos_token_id=1, eos_token_id=2)
    config = LlavaConfig(vision_config=vision.to_dict(), text_config=text.to_dict(),
                         image_token_index=26, vision_feature_layer=-1)
    config._attn_implementation = "eager"
    model = LlavaForConditionalGeneration(config).double()
    training = TrainingSpec(fine_tuning_strategy=strategy, knowledge=knowledge,
                            token_lengths_known=True, lora_rank=16)
    processor = SimpleNamespace(tokenizer=TinyTokenizer(), image_processor=SimpleNamespace(
        image_mean=[0.5] * 3, image_std=[0.5] * 3))
    adapter = HFAdapter(ModelSpec(family="llava", revision="offline-test", dtype="float64"),
                         training, backend=model, processor=processor)
    adapter.configure_training()
    adapter.double()
    if not zero:
        with torch.no_grad():
            for name, parameter in adapter.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(std=0.03)
    batch = adapter.batch(torch.rand(1, 3, 8, 8, dtype=torch.float64), [question], [answer])
    observation = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
    return adapter, batch, observation


def specification(**options):
    return AttackSpec(method="dager_adapted", text_method="none", checkpoint_interval=1,
                      dager=DAGEROptions(rank_rtol=1e-9, rank_atol=1e-11,
                                          token_threshold=1e-6, **options))


def run(adapter, obs, spec, **kwargs):
    return create_attacker(adapter, spec).attack(
        obs.tensors, obs, AdversaryKnowledge(obs.training.knowledge, token_lengths_known=True), **kwargs)


def test_linear_and_lora_gradient_identity_and_initial_degeneracy():
    x = torch.randn(5, 12, dtype=torch.float64)
    d = torch.randn(5, 10, dtype=torch.float64)
    a = torch.randn(7, 12, dtype=torch.float64, requires_grad=True)
    b = torch.randn(10, 7, dtype=torch.float64, requires_grad=True)
    loss = (((x @ a.T) @ b.T) * d).sum()
    ga, gb = torch.autograd.grad(loss, (a, b))
    gw = d.T @ x
    torch.testing.assert_close(ga, b.T @ gw)
    torch.testing.assert_close(gb, gw @ a.T)
    basis, _ = row_basis(ga, 1e-10, 1e-12)
    torch.testing.assert_close(residual(x, basis), torch.zeros_like(x), atol=1e-10, rtol=0)
    with torch.no_grad():
        b.zero_()
    ga, gb = torch.autograd.grad((((x @ a.T) @ b.T) * d).sum(), (a, b))
    assert ga.count_nonzero() == 0 and gb.norm() > 0


def test_low_rank_alone_does_not_imply_true_token_membership():
    x = torch.eye(12, dtype=torch.float64)[:6]
    gradient = torch.randn(2, 6, dtype=torch.float64) @ x
    span = SpanFilter.build([gradient], x[:0], DAGEROptions(mode="raw"))
    scores, _ = span.score(x)
    assert span.diagnostics["rank"] == 2
    assert (scores > 0.1).all()  # True inputs need not lie in the smaller gradient span.


def test_public_quotient_cancels_unknown_backward_coefficients_and_joint_spans():
    public = torch.eye(12, dtype=torch.float64)[:5]
    private = torch.eye(12, dtype=torch.float64)[5:8]
    coefficients = torch.randn(3, 5, dtype=torch.float64)
    gradient = coefficients @ public + private
    options = DAGEROptions(rank_rtol=1e-10, rank_atol=1e-12)
    span = SpanFilter.build([row[None] for row in gradient], public, options)
    torch.testing.assert_close(residual(gradient, span.public_basis), private)
    scores, ambiguous = span.score(torch.eye(12, dtype=torch.float64))
    assert ambiguous[:5].all() and not ambiguous[5:].any()
    assert scores[5:8].max() < 1e-10 and scores[8:].min() > 0.99
    assert span.diagnostics["rank"] == 3


def test_public_overlap_is_retained_without_being_a_positive_detection():
    scores = torch.tensor([0., 0.001, 0.5, 0.002])
    ambiguous = torch.tensor([True, False, False, False])
    ids, metadata = select_tokens(scores, ambiguous, set(), DAGEROptions(max_candidates=1))
    assert ids == [0, 1]
    assert metadata["ambiguous_candidates"] == 1 and metadata["truncated"]


def test_dager_excludes_padded_embedding_rows_outside_tokenizer():
    adapter = SimpleNamespace(
        tokenizer=SimpleNamespace(__len__=lambda self: 5, all_special_ids=[0, 1]),
        vocab_size=8, eos=2, pad=0)
    # SpecialNamespace does not dispatch a per-instance __len__; use a tiny class.
    class Tokenizer:
        all_special_ids = [0, 1]
        def __len__(self):
            return 5
    adapter.tokenizer = Tokenizer()
    assert forbidden_token_ids(adapter) == {0, 1, 2, 5, 6, 7}
    adapter.tokenizer = Tokenizer()
    adapter.vocab_size = 4
    with pytest.raises(ValueError, match="exceeds"):
        forbidden_token_ids(adapter)


@pytest.mark.parametrize("knowledge", ["image_question_known", "image_known"])
def test_partial_forward_matches_native_attention_inputs_and_fixed_slots(knowledge):
    adapter, batch, obs = fixture(knowledge)
    view = LlavaTextView(adapter, obs)
    captured = {}
    handles = []
    for index in (0, 1):
        def record(module, args, index=index):
            captured[index] = args[0].detach().clone()
        handles.append(adapter.backend.language_model.model.layers[index].self_attn.q_proj.register_forward_pre_hook(record))
    try:
        adapter(batch.images, batch.questions, batch.targets)
    finally:
        for handle in handles:
            handle.remove()
    truth = {"questions": batch.questions[0], "targets": batch.targets[0]}
    prefix = [int(truth[field][index]) for field, index, _ in view.slots]
    for depth in range(1, len(prefix) + 1):
        actual = view.prefix_features([prefix[:depth]])
        # HF LLaMA RMSNorm/attention use float32 internally even for fp64 model
        # weights; a shorter causal prefix changes reduction rounding slightly.
        torch.testing.assert_close(actual, captured[1][:, view.slots[depth - 1][2]], atol=1e-6, rtol=1e-5)
    ids = view.token_ids(prefix)
    torch.testing.assert_close(ids["questions"], batch.questions)
    torch.testing.assert_close(ids["targets"], batch.targets)
    if knowledge == "image_question_known":
        assert view.slots[0][2] > adapter.spec.question_length
    assert view.public_layer1.shape[0] == view.slots[0][2]
    visual, _ = adapter.visual_embeddings(batch.images)
    embeds = torch.cat(llava_input_pieces(adapter, visual, adapter.embedding()(batch.questions),
                                         adapter.embedding()(batch.targets[:, :-1])), 1)
    torch.testing.assert_close(view.layer_input(embeds, 0), captured[0])
    torch.testing.assert_close(view.layer_input(embeds, 1), captured[1])


@pytest.mark.parametrize("knowledge,answer", [("image_question_known", "red circle"),
                                            ("image_question_known", "color"),
                                            ("image_known", "red")])
def test_synthetic_text_search_round_trip(knowledge, answer, tmp_path):
    adapter, batch, obs = fixture(knowledge, answer)
    result = run(adapter, obs, specification(), directory=tmp_path)
    assert result.status == "completed", result.reason
    assert result.images is None
    assert result.questions == adapter.decode(batch.questions)
    assert result.targets == adapter.decode(batch.targets)
    assert result.costs["full_gradient_replays"] == 0
    assert result.costs["vocab_tokens_scored"] == adapter.vocab_size
    assert not (tmp_path / "images.safetensors").exists()
    tokens = read_tensors(tmp_path / "text_tokens.safetensors")
    torch.testing.assert_close(tokens["targets"], batch.targets)
    assert "private_text_union" == read_json(tmp_path / "token_candidates.json")["scope"]


def test_raw_mode_and_optional_observable_reranking(tmp_path):
    adapter, batch, obs = fixture(answer="red")
    result = run(adapter, obs, specification(mode="raw", rerank_candidates=2), directory=tmp_path)
    assert result.status == "completed" and result.targets == adapter.decode(batch.targets)
    assert result.costs["full_gradient_replays"] == 2
    assert result.provenance["final_score_type"] == "relative_l2_update"


def test_support_checks_actual_gradients_and_uploaded_subset():
    adapter, _, obs = fixture(zero=True)
    assert "zero" in dager_support(adapter, obs).reason
    adapter, _, obs = fixture()
    assert obs.training.server_round == 0  # Nonzero fixture weights, not a round-number heuristic.
    assert dager_support(adapter, obs).status == "supported"
    names = gradient_groups(adapter, obs, "q")
    subset = replace(obs, tensors={name: obs.tensors[name] for group in names for name in group})
    assert dager_support(adapter, subset).status == "not_applicable"
    assert dager_support(adapter, subset, DAGEROptions(projections="q")).status == "supported"
    from attacks.registry import supports
    assert supports("dager_adapted", adapter, subset, specification(projections="q")).status == "supported"
    assert dager_support(adapter, replace(obs, training=replace(obs.training, algorithm="fedavg"))).status == "not_applicable"
    assert dager_support(adapter, replace(obs, training=replace(obs.training, token_lengths_known=False))).status == "not_applicable"
    assert run(adapter, obs, specification(), upload_metadata={"defense": {"name": "sign_sgd"}}).status == "not_applicable"


def test_budget_counts_candidates_not_batches_and_does_not_fake_completion(tmp_path):
    adapter, _, obs = fixture()
    spec = replace(specification(prefix_batch_size=32), max_evaluations=2)
    result = run(adapter, obs, spec, directory=tmp_path)
    assert result.status == "budget_exhausted" and not result.targets
    assert result.costs["prefix_candidates_evaluated"] == 2
    resumed = run(adapter, obs, spec, directory=tmp_path, resume=True)
    assert resumed.status == "budget_exhausted"
    assert resumed.costs["prefix_candidates_evaluated"] == 2
    pointer = read_json(tmp_path / "checkpoint.json")
    with pytest.raises(ValueError, match="differs"):
        run(adapter, obs, replace(spec, max_evaluations=3), directory=tmp_path, resume=True)
    assert read_json(tmp_path / "checkpoint.json") == pointer


def test_exception_checkpoint_resumes_same_search_without_losing_prefixes(tmp_path, monkeypatch):
    adapter, _, obs = fixture()
    spec = specification(prefix_batch_size=2)
    expected = run(adapter, obs, spec)
    original = LlavaTextView.prefix_features
    calls = 0

    def interrupt(self, prefixes):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic interruption")
        return original(self, prefixes)

    with monkeypatch.context() as patch:
        patch.setattr(LlavaTextView, "prefix_features", interrupt)
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            run(adapter, obs, spec, directory=tmp_path)
    assert read_json(tmp_path / "failed.json")["error_type"] == "RuntimeError"
    resumed = run(adapter, obs, spec, directory=tmp_path, resume=True)
    assert resumed.status == "completed"
    assert (resumed.questions, resumed.targets) == (expected.questions, expected.targets)
    assert resumed.costs["prefix_candidates_evaluated"] == expected.costs["prefix_candidates_evaluated"] + 2


def test_hydra_and_strict_configuration_expose_self_contained_method():
    from pathlib import Path
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        cfg = compose(config_name="config", overrides=["attack=dager_adapted", "model=llava",
                                                       "knowledge=image_question_known",
                                                       "knowledge.token_lengths_known=true"])
    parsed = protocol_config(cfg)
    assert parsed.attack.method == "dager_adapted" and parsed.attack.text_method == "none"
    assert isinstance(parsed.attack.dager, DAGEROptions)
    with pytest.raises(ValueError, match="text_method=none"):
        load_config(overrides=["attack.method=dager_adapted"])
    for override in ["attack.dager.rank_rtol=-1", "attack.dager.prefix_batch_size=0",
                     "attack.dager.mode=invalid", "attack.init_text_source=private_reference"]:
        with pytest.raises(ValueError):
            load_config(overrides=["attack.method=dager_adapted", "attack.text_method=none", override])


def test_dager_original_name_remains_unimplemented():
    adapter, _, obs = fixture()
    original = replace(specification(), method="dager", text_method="tag_adapted")
    assert run(adapter, obs, original).status == "not_implemented"


def test_rank_zero_no_candidates_and_time_budget_are_explicit(tmp_path, monkeypatch):
    adapter, _, obs = fixture()
    spec = specification()
    no_rank = replace(spec, dager=replace(spec.dager, rank_atol=100.))
    assert run(adapter, obs, no_rank).status == "no_signal"
    with monkeypatch.context() as patch:
        patch.setattr("attacks.analytic.dager_adapted.select_tokens", lambda *args: ([], {}))
        result = run(adapter, obs, spec)
        assert result.status == "no_candidates" and result.targets == []
    with monkeypatch.context() as patch:
        patch.setattr(DAGERSearch, "elapsed", lambda self: spec.seconds + 1)
        result = run(adapter, obs, spec, directory=tmp_path)
        assert result.status == "budget_exhausted"
        assert result.costs["vocab_tokens_scored"] == 0


def test_token_metrics_are_post_commit_and_distinguish_ambiguity(tmp_path):
    from evaluation.token_recovery import evaluate_token_candidates
    adapter, batch, obs = fixture(answer="color")
    output, truth = tmp_path / "attack", tmp_path / "truth"
    run(adapter, obs, specification(), directory=output)
    with pytest.raises(ValueError, match="committed"):
        evaluate_token_candidates(output, truth, obs.training)
    write_json(output / "result.json", {"status": "completed"})
    assert evaluate_token_candidates(output, truth, obs.training)["status"] == "unavailable"
    write_tensors(truth / "text_tokens.safetensors", {"questions": batch.questions, "targets": batch.targets})
    metrics = evaluate_token_candidates(output, truth, obs.training)
    assert metrics["candidates"]["recall"] == 1
    assert metrics["informative_detections"]["recall"] == 0
    assert metrics["ambiguous_reference_count"] == 1


def test_equal_topk_ablation_compares_scores_after_commit(tmp_path):
    from evaluation.dager_ablation import compare_token_filters, validate_token_filters
    truth = tmp_path / "truth"
    write_tensors(truth / "text_tokens.safetensors", {
        "questions": torch.tensor([[4, 5, 2, 0]]),
        "targets": torch.tensor([[6, 7, 2, 0]])})
    directories = {}
    for name, scores, ambiguous in [
            ("raw", [0.9, 0.8, 0.7, 0.1, 0.2, 0.6, 0.3, 0.4], [False] * 8),
            ("residual", [0.9, 0.8, 0.7, 0.6, 0.5, 0.1, 0.2, 0.3],
             [False, False, False, True, False, False, False, False])]:
        directory = tmp_path / name
        directories[name] = directory
        write_json(directory / "result.json", {"status": "budget_exhausted"})
        write_json(directory / "token_candidates.json", {
            "schema_version": 1, "scope": "private_text_union", "forbidden_ids": [0, 1, 2]})
        write_tensors(directory / "token_candidates.safetensors", {
            "token_ids": torch.arange(8), "scores": torch.tensor(scores),
            "ambiguous": torch.tensor(ambiguous)})
    report = compare_token_filters(
        directories, truth, TrainingSpec(knowledge="image_known"), topk=3)
    assert report["modes"]["raw"]["topk"]["recall"] == 0.5
    assert report["modes"]["residual"]["topk"]["recall"] == 0.75
    assert report["modes"]["residual"]["ambiguous_in_topk"] == 0
    assert report["pairs"]["raw__residual"]["intersection"] == 1
    validation = validate_token_filters(
        directories, truth, TrainingSpec(knowledge="image_known"),
        wrong_references=[{3, 4}, {3, 5}], exact_length_references=[{3, 4}],
        wrong_references_by_field={"questions": [{3, 4}], "targets": [{3, 5}]},
        exact_length_references_by_field={"questions": [{3, 4}], "targets": [{3, 5}]},
        distinct_references_by_field={"questions": [{3}], "targets": [{5}]},
        exact_length_distinct_references_by_field={"questions": [{3}], "targets": [{5}]},
        topk=3, topks=(1, 3, 5), random_draws=1000, seed=7)
    raw = validation["modes"]["raw"]
    residual = validation["modes"]["residual"]
    assert raw["ranking"]["curve"][1]["hits"] == 2
    assert residual["ranking"]["curve"][1]["hits"] == 3
    assert residual["ranking"]["average_precision"] > raw["ranking"]["average_precision"]
    assert raw["random_topk_control"]["expected_hits"] == 2.4
    assert raw["random_topk_control"]["exact_probability_at_least_observed"] == 1
    assert raw["wrong_text_control"]["reference_count"] == 2
    assert raw["exact_length_wrong_text_control"]["reference_count"] == 1
    assert raw["wrong_text_control_by_field"]["questions"]["reference_count"] == 1
    assert raw["exact_length_wrong_text_control_by_field"]["targets"]["reference_count"] == 1
    assert raw["distinct_wrong_text_control_by_field"]["questions"]["reference_count"] == 1
    assert raw["exact_length_distinct_wrong_text_control_by_field"]["targets"]["reference_count"] == 1
    assert "references" not in raw["wrong_text_control"]
    with pytest.raises(ValueError, match="committed"):
        compare_token_filters({"missing": tmp_path / "missing"}, truth,
                              TrainingSpec(knowledge="image_known"), topk=3)


def test_staged_command_commits_text_only_result_before_evaluation(tmp_path, monkeypatch):
    from core.commands import build_parser
    from core.fl import save_observation
    adapter, batch, obs = fixture(answer="red", strategy="f_cl")
    public, output, truth = tmp_path / "public", tmp_path / "attack", tmp_path / "truth"
    observation_id = save_observation(public, obs)
    monkeypatch.setattr("core.fl.restore_model", lambda *args: adapter)
    args = build_parser().parse_args([
        "attack", "--observation", str(public), "--output", str(output),
        "--set", "attack.method=dager_adapted", "--set", "attack.text_method=none"])
    args.func(args)
    result = read_json(output / "result.json")
    assert result["status"] == "completed" and result["condition"]["method"] == "dager_adapted"
    assert result["targets"] == adapter.decode(batch.targets)
    assert result["condition"]["fine_tuning_strategy"] == "f_cl"
    assert not truth.exists()  # No private reference was available during the attack.
    write_json(truth / "truth.json", {"observation_id": observation_id,
        "training": asdict(obs.training), "samples": [{"image_id": "synthetic", "sample_id": "synthetic",
        "model_question": adapter.decode(batch.questions)[0], "model_target": adapter.decode(batch.targets)[0]}]})
    write_tensors(truth / "images.safetensors", {"images": batch.images})
    write_tensors(truth / "text_tokens.safetensors", {"questions": batch.questions, "targets": batch.targets})
    args = build_parser().parse_args(["evaluate", "--reconstruction", str(output), "--truth", str(truth)])
    args.func(args)
    report = read_json(output / "evaluation.json")
    assert report["scored_fields"] == ["target"]
    assert report["token_detection"]["candidates"]["recall"] == 1
    assert report["samples"][0]["metrics"]["target_exact_match"] == 1


def test_changed_upload_and_checkpoint_schema_are_rejected(tmp_path):
    adapter, _, obs = fixture()
    spec = specification()
    run(adapter, obs, spec, directory=tmp_path)
    with pytest.raises(FileExistsError):
        run(adapter, obs, spec, directory=tmp_path)
    altered = replace(obs, tensors={name: tensor * 2 for name, tensor in obs.tensors.items()})
    with pytest.raises(ValueError, match="differs"):
        run(adapter, altered, spec, directory=tmp_path, resume=True)
    generation = read_json(tmp_path / "checkpoint.json")["generation"]
    path = tmp_path / "checkpoints" / generation / "state.json"
    meta = read_json(path)
    meta["schema"] = 0
    write_json(path, meta)
    with pytest.raises(ValueError, match="schema"):
        run(adapter, obs, spec, directory=tmp_path, resume=True)


def test_fresh_experiment_resume_flag_starts_without_a_checkpoint(tmp_path):
    adapter, batch, obs = fixture(answer="red")
    result = run(adapter, obs, specification(), directory=tmp_path, resume=True)
    assert result.status == "completed"
    assert result.targets == adapter.decode(batch.targets)
    assert (tmp_path / "checkpoint.json").exists()


def test_oom_is_saved_and_reraised_without_retry(tmp_path, monkeypatch):
    adapter, _, obs = fixture()
    calls = 0

    def oom(*args):
        nonlocal calls
        calls += 1
        raise torch.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(LlavaTextView, "prefix_features", oom)
    with pytest.raises(torch.OutOfMemoryError, match="synthetic OOM"):
        run(adapter, obs, specification(), directory=tmp_path)
    assert calls == 1
    assert read_json(tmp_path / "failed.json")["error_type"] == "OutOfMemoryError"
