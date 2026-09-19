from dataclasses import replace
import json

import pytest
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from attacks import AttackRunner, Candidate, matching_loss, supports
from core.aggregation import apply_server_update, create_federated_algorithm
from core.config import AttackSpec, ModelSpec, TrainingSpec, load_config
from core.fl import (capture, load_observation, mask_upload, resolve_upload,
                     restore_model, save_model, save_observation, simulate_update)
from core.vlm_wrapper import build_model, canonical_probabilities


def fixture(strategy="f_l", algorithm="fedsgd", steps=1, task="vqa", knowledge="private",
            optimizer="sgd", accumulation=1, weight_decay=0.0, dtype="float32",
            token_lengths_known=False, batch_size=1):
    training = TrainingSpec(fine_tuning_strategy=strategy, algorithm=algorithm, local_steps=steps,
                            local_optimizer=optimizer, gradient_accumulation_steps=accumulation,
                            weight_decay=weight_decay, task=task, knowledge=knowledge,
                            token_lengths_known=token_lengths_known, batch_size=batch_size)
    adapter = build_model(ModelSpec(dtype=dtype), training)
    batch = adapter.batch(torch.rand(training.sample_count, 3, 8, 8),
                          ["what color is the object"] * training.sample_count,
                          ["red blue"] * training.sample_count)
    return adapter, batch


@pytest.mark.parametrize("strategy", ["f_c", "f_l", "f_cl", "f_2stage"])
@pytest.mark.parametrize("task", ["vqa", "caption"])
def test_one_step_sgd_identity_and_input_gradients(strategy, task):
    adapter, batch = fixture(strategy=strategy, task=task)
    gradient = simulate_update(adapter, batch, adapter.training_spec)
    spec = replace(adapter.training_spec, algorithm="fedavg")
    delta = simulate_update(adapter, batch, spec)
    for name in gradient:
        torch.testing.assert_close(delta[name], -spec.lr * gradient[name], atol=2e-7, rtol=1e-4)
    obs = capture(adapter, batch, [], [])
    candidate = Candidate(adapter, obs, 19)
    fake = simulate_update(adapter, candidate.batch(), spec, True)
    loss = matching_loss(fake, delta, "l2")
    gradients = torch.autograd.grad(loss, tuple(candidate.parameters()))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    if strategy == "f_l":
        assert all("lora_" in name and "language" in name for name in gradient)
        assert all(g.count_nonzero() == 0 for name, g in gradient.items() if "lora_A" in name)
        assert any(g.count_nonzero() > 0 for name, g in gradient.items() if "lora_B" in name)
        assert supports("april_adapted", adapter, obs).status == "not_applicable"


@pytest.mark.parametrize("strategy", ["f_c", "f_l", "f_cl", "f_2stage"])
def test_multistep_matches_actual_optimizer(strategy):
    adapter, batch = fixture(strategy, "fedavg", steps=3)
    before = {k: v.detach().clone() for k, v in adapter.trainable().items()}
    observed = simulate_update(adapter, batch, adapter.training_spec)
    optimizer = torch.optim.SGD(adapter.trainable().values(), lr=adapter.training_spec.lr)
    for i in range(3):
        chunk = batch.slice(i, i + 1)
        optimizer.zero_grad()
        adapter(chunk.images, chunk.questions, chunk.targets).backward()
        optimizer.step()
    for name, value in adapter.trainable().items():
        torch.testing.assert_close(value - before[name], observed[name], atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("strategy", ["f_c", "f_l", "f_cl", "f_2stage"])
def test_adamw_accumulation_matches_actual_optimizer(strategy):
    adapter, batch = fixture(strategy, "fedavg", steps=2, optimizer="adamw", accumulation=2,
                             weight_decay=0.1, dtype="float64")
    spec = adapter.training_spec
    before = {name: value.detach().clone() for name, value in adapter.trainable().items()}
    observed = simulate_update(adapter, batch, spec)
    parameters, decay = adapter.trainable(), adapter.decay_parameter_names()
    optimizer = torch.optim.AdamW([
        {"params": [value for name, value in parameters.items() if name in decay],
         "weight_decay": spec.weight_decay},
        {"params": [value for name, value in parameters.items() if name not in decay],
         "weight_decay": 0.0},
    ], lr=spec.lr, betas=(spec.adam_beta1, spec.adam_beta2), eps=spec.adam_epsilon,
        foreach=False, fused=False)

    for step in range(spec.local_steps):
        optimizer.zero_grad(set_to_none=True)
        for accumulation_id in range(spec.gradient_accumulation_steps):
            index = (step * spec.gradient_accumulation_steps + accumulation_id) * spec.batch_size
            chunk = batch.slice(index, index + spec.batch_size)
            (adapter(chunk.images, chunk.questions, chunk.targets)
             / spec.gradient_accumulation_steps).backward()
        optimizer.step()

    for name, value in adapter.trainable().items():
        torch.testing.assert_close(value - before[name], observed[name], atol=1e-11, rtol=1e-7)
    assert all(not name.endswith(".bias") for name in decay)
    assert all("norm" not in name for name in decay)


def test_delta_aggregation_equals_parameter_averaging():
    adapter, batch = fixture("f_cl", "fedavg", steps=2)
    algorithm = create_federated_algorithm(adapter.training_spec)
    initial = {name: value.detach().clone() for name, value in adapter.trainable().items()}
    first = algorithm.client_update(adapter, batch)
    second = algorithm.client_update(adapter, replace(batch, images=1 - batch.images))
    delta = algorithm.aggregate(iter([first, second]), [2, 3])
    for name in initial:
        averaged_parameters = ((initial[name] + first[name]) * 2 / 5
                               + (initial[name] + second[name]) * 3 / 5)
        torch.testing.assert_close(initial[name] + delta[name], averaged_parameters)


@pytest.mark.parametrize("strategy,stage,connector,lora", [
    ("f_c", "connector", True, False),
    ("f_l", "llm", False, True),
    ("f_cl", "joint", True, True),
    ("f_2stage", "connector", True, False),
])
def test_fedvlm_fine_tuning_parameter_surfaces(strategy, stage, connector, lora):
    adapter, _ = fixture(strategy=strategy)
    names = set(adapter.trainable())
    assert adapter.training_spec.fine_tuning_stage == stage
    assert any(adapter.is_connector(name) for name in names) == connector
    assert any(".lora_" in name for name in names) == lora
    assert all(adapter.is_connector(name) or ".lora_" in name for name in names)


def test_two_stage_switches_from_connector_to_lora():
    adapter, _ = fixture(strategy="f_2stage")
    assert adapter.training_spec.fine_tuning_stage == "connector"
    assert all(adapter.is_connector(name) for name in adapter.trainable())
    adapter.set_training_spec(replace(adapter.training_spec, server_round=1))
    assert adapter.training_spec.fine_tuning_stage == "llm"
    assert adapter.trainable() and all(".lora_" in name for name in adapter.trainable())


@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
def test_federated_algorithms_use_same_initial_state_and_weights(method):
    adapter, batch = fixture("f_l", "fedavg", steps=2)
    training = replace(adapter.training_spec, algorithm=method,
                       local_steps=1 if method == "fedsgd" else 2)
    algorithm = create_federated_algorithm(training)
    assert algorithm.upload_type == ("gradient" if method == "fedsgd" else "client_delta")
    if method == "fedsgd":
        batch = batch.slice(0, 1)
    other = replace(batch, images=1 - batch.images)
    initial = {k: v.detach().clone() for k, v in adapter.trainable().items()}
    first = algorithm.client_update(adapter, batch)
    second = algorithm.client_update(adapter, other)
    server_update = algorithm.aggregate(iter([first, second]), [1, 3])
    algorithm.apply(adapter, server_update)
    scale = -training.lr if method == "fedsgd" else 1
    for name, value in adapter.trainable().items():
        expected = initial[name] + scale * (first[name] / 4 + second[name] * 3 / 4)
        torch.testing.assert_close(value, expected)


def test_federated_algorithm_applies_only_uploaded_parameters():
    adapter, batch = fixture()
    before = {name: value.detach().clone() for name, value in adapter.trainable().items()}
    uploaded_name = next(iter(before))
    training = replace(adapter.training_spec, upload_parameters=[uploaded_name])
    algorithm = create_federated_algorithm(training)
    upload = algorithm.client_update(adapter, batch)
    assert set(upload) == {uploaded_name}
    algorithm.apply(adapter, algorithm.aggregate(iter([upload]), [1]))
    for name, value in adapter.trainable().items():
        expected = before[name] - training.lr * upload[name] if name in upload else before[name]
        torch.testing.assert_close(value, expected)


def test_server_update_validation_precedes_model_mutation():
    adapter, _ = fixture()
    before = {name: value.detach().clone() for name, value in adapter.trainable().items()}
    update = {name: torch.zeros_like(value) for name, value in before.items()}
    first = next(iter(update))
    update[first] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_server_update(adapter, update)
    for name, value in adapter.trainable().items():
        torch.testing.assert_close(value, before[name])


def test_observation_allowlist_and_integrity(tmp_path):
    adapter, batch = fixture()
    obs = capture(adapter, batch, ["secret question"], ["secret answer"])
    save_observation(tmp_path, obs)
    assert "secret" not in (tmp_path / "observation.json").read_text()
    loaded = load_observation(tmp_path)
    assert not loaded.public_targets and not loaded.public_questions
    assert not loaded.public_target_lengths and not loaded.public_question_lengths
    meta = read_json(tmp_path / "observation.json")
    meta["ground_truth"] = "forbidden"
    (tmp_path / "observation.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unknown"):
        load_observation(tmp_path)


def test_legacy_artifact_schemas_are_rejected(tmp_path):
    adapter, batch = fixture()
    observation_dir = tmp_path / "observation"
    save_observation(observation_dir, capture(adapter, batch, [], []))
    meta = read_json(observation_dir / "observation.json")
    meta["schema_version"] = 3
    (observation_dir / "observation.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema v3"):
        load_observation(observation_dir)

    model_dir = tmp_path / "model"
    save_model(model_dir, adapter)
    model_meta = read_json(model_dir / "model.json")
    model_meta["schema_version"] = 3
    (model_dir / "model.json").write_text(json.dumps(model_meta))
    with pytest.raises(ValueError, match="schema v3"):
        restore_model(model_dir)


@pytest.mark.parametrize("strategy,connector,lora", [
    ("f_c", True, False),
    ("f_l", False, True),
    ("f_cl", True, True),
    ("f_2stage", True, True),
])
def test_model_snapshot_contains_only_strategy_mutable_state(
        tmp_path, strategy, connector, lora):
    adapter, batch = fixture(strategy=strategy)
    directory = tmp_path / strategy
    expected = adapter.federated_state()
    fingerprint = adapter.fingerprint()
    expected_update = simulate_update(adapter, batch, adapter.training_spec)

    save_model(directory, adapter)
    stored = read_tensors(directory / "model.safetensors")
    meta = read_json(directory / "model.json")

    assert meta["schema_version"] == 4
    assert meta["state_scope"] == "strategy_mutable"
    assert meta["parameter_names"] == list(expected)
    assert meta["state_sha256"] == file_hash(directory / "model.safetensors")
    assert set(stored) == set(expected)
    assert any(adapter.is_connector(name) for name in stored) == connector
    assert any(".lora_" in name for name in stored) == lora
    assert all(adapter.is_connector(name) or ".lora_" in name for name in stored)
    assert (directory / "model.safetensors").stat().st_size < sum(
        value.numel() * value.element_size() for value in adapter.state_dict().values())

    restored = restore_model(directory)
    assert restored.fingerprint() == fingerprint
    torch.testing.assert_close(
        restored(batch.images, batch.questions, batch.targets),
        adapter(batch.images, batch.questions, batch.targets))
    restored_update = simulate_update(restored, batch, restored.training_spec)
    assert restored_update.keys() == expected_update.keys()
    for name in expected_update:
        torch.testing.assert_close(restored_update[name], expected_update[name])


def test_two_stage_snapshot_retains_connector_after_switch_to_lora(tmp_path):
    adapter, batch = fixture(strategy="f_2stage")
    connector_before = {name: value.detach().clone() for name, value in adapter.federated_state().items()
                        if adapter.is_connector(name)}
    algorithm = create_federated_algorithm(adapter.training_spec)
    algorithm.apply(adapter, algorithm.client_update(adapter, batch))
    connector_after = {name: value.detach().clone() for name, value in adapter.federated_state().items()
                       if adapter.is_connector(name)}
    assert any(not torch.equal(connector_after[name], connector_before[name])
               for name in connector_before)

    adapter.set_training_spec(replace(adapter.training_spec, server_round=1))
    assert adapter.training_spec.fine_tuning_stage == "llm"
    directory = tmp_path / "two-stage"
    save_model(directory, adapter)
    restored = restore_model(directory)

    assert restored.training_spec.fine_tuning_stage == "llm"
    restored_state = restored.federated_state()
    for name, value in connector_after.items():
        torch.testing.assert_close(restored_state[name], value)
    assert any(".lora_" in name for name in restored_state)


def test_model_snapshot_rejects_tensor_hash_mismatch(tmp_path):
    adapter, _ = fixture(strategy="f_cl")
    save_model(tmp_path, adapter)
    tensors = read_tensors(tmp_path / "model.safetensors")
    name = next(iter(tensors))
    tensors[name] = tensors[name].clone()
    tensors[name].view(-1)[0].add_(1)
    write_tensors(tmp_path / "model.safetensors", tensors)
    with pytest.raises(ValueError, match="integrity"):
        restore_model(tmp_path)


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype"])
def test_model_snapshot_rejects_invalid_overlay(tmp_path, corruption):
    adapter, _ = fixture(strategy="f_cl")
    save_model(tmp_path, adapter)
    tensors = read_tensors(tmp_path / "model.safetensors")
    name = next(iter(tensors))
    if corruption == "missing":
        tensors.pop(name)
    elif corruption == "extra":
        tensors["unexpected.weight"] = torch.zeros(1)
    elif corruption == "shape":
        tensors[name] = tensors[name].reshape(-1)[:-1]
    else:
        tensors[name] = tensors[name].double()
    write_tensors(tmp_path / "model.safetensors", tensors)
    meta = read_json(tmp_path / "model.json")
    meta["state_sha256"] = file_hash(tmp_path / "model.safetensors")
    if corruption in {"missing", "extra"}:
        meta["parameter_names"] = sorted(tensors)
    write_json(tmp_path / "model.json", meta)
    with pytest.raises(ValueError, match="parameter names|shape differs|dtype differs"):
        restore_model(tmp_path)


def test_model_snapshot_rejects_fingerprint_mismatch(tmp_path):
    adapter, _ = fixture(strategy="f_cl")
    save_model(tmp_path, adapter)
    meta = read_json(tmp_path / "model.json")
    meta["fingerprint"] = "0" * 64
    write_json(tmp_path / "model.json", meta)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        restore_model(tmp_path)


@pytest.mark.parametrize("knowledge", ["private", "question_known", "text_known"])
def test_known_text_is_not_optimized(knowledge):
    adapter, batch = fixture(knowledge=knowledge)
    obs = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
    candidate = Candidate(adapter, obs, 3)
    names = dict(candidate.named_parameters())
    assert ("questions" in names) == (knowledge == "private")
    assert ("targets" in names) == (knowledge != "text_known")
    if knowledge == "text_known":
        assert candidate.decoded() == (obs.public_questions, obs.public_targets)


@pytest.mark.parametrize("task", ["vqa", "caption"])
@pytest.mark.parametrize("knowledge", ["private", "question_known", "text_known"])
def test_capture_exposes_only_authorized_token_lengths(tmp_path, task, knowledge):
    if task == "caption" and knowledge == "question_known":
        pytest.skip("question_known is not a caption condition")
    adapter, _ = fixture(task=task, knowledge=knowledge, token_lengths_known=True,
                         batch_size=2)
    images = torch.rand(2, 3, 8, 8)
    batch = adapter.batch(images, ["a", "what color"], ["", "red blue"])
    questions, targets = adapter.decode(batch.questions), adapter.decode(batch.targets)
    observation = capture(adapter, batch, questions, targets)

    expected_questions = adapter.content_lengths(batch.questions) if task == "vqa" else []
    assert observation.public_question_lengths == expected_questions
    assert observation.public_target_lengths == adapter.content_lengths(batch.targets)
    assert bool(observation.public_question_ids) == (task == "vqa" and knowledge != "private")
    assert bool(observation.public_target_ids) == (knowledge == "text_known")

    save_observation(tmp_path, observation)
    loaded = load_observation(tmp_path)
    assert loaded.public_question_lengths == observation.public_question_lengths
    assert loaded.public_target_lengths == observation.public_target_lengths


def test_known_lengths_fix_candidate_eos_pad_and_response_mask():
    adapter, _ = fixture(token_lengths_known=True, batch_size=2)
    batch = adapter.batch(torch.rand(2, 3, 8, 8), ["a", "what color"], ["", "red blue"])
    observation = capture(adapter, batch, [], [])
    candidate = Candidate(adapter, observation, 7)

    for field, lengths in [("questions", observation.public_question_lengths),
                           ("targets", observation.public_target_lengths)]:
        probabilities = candidate.distribution(field)
        discrete = candidate.distribution(field, True).argmax(-1)
        for row, length in enumerate(lengths):
            assert discrete[row, length] == adapter.eos
            assert torch.all(discrete[row, length + 1:] == adapter.pad)
            assert probabilities[row, :length, adapter.eos].count_nonzero() == 0
            assert probabilities[row, :length, adapter.pad].count_nonzero() == 0

    targets = candidate.distribution("targets")
    _, alive = canonical_probabilities(targets, adapter.eos, adapter.pad)
    expected = torch.arange(targets.shape[1])[None, :] <= torch.tensor(
        observation.public_target_lengths)[:, None]
    torch.testing.assert_close(alive.cpu(), expected.to(alive.dtype))

    before = candidate.batch(True).targets.clone()
    with torch.no_grad():
        for row, length in enumerate(observation.public_target_lengths):
            candidate.targets[row, length:].normal_()
    torch.testing.assert_close(candidate.batch(True).targets, before)

    predicted = simulate_update(adapter, candidate.batch(), adapter.training_spec, True)
    objective = matching_loss(predicted, observation.tensors, "l2")
    gradients = torch.autograd.grad(objective, tuple(candidate.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    by_name = dict(zip((name for name, _ in candidate.named_parameters()), gradients, strict=True))
    assert by_name["questions"].norm() > 0
    assert by_name["targets"].norm() > 0


def test_token_length_validation_rejects_unauthorized_or_malformed_values():
    adapter, batch = fixture()
    observation = capture(adapter, batch, [], [])
    observation.public_target_lengths = [1]
    with pytest.raises(ValueError, match="Unauthorized"):
        observation.validate()

    adapter, batch = fixture(token_lengths_known=True)
    observation = capture(adapter, batch, [], [])
    observation.public_target_lengths = []
    with pytest.raises(ValueError, match="Missing"):
        observation.validate()
    observation.public_target_lengths = [adapter.spec.target_length]
    with pytest.raises(ValueError, match="Malformed"):
        observation.validate()


def test_known_public_lengths_must_match_public_token_ids():
    adapter, batch = fixture(knowledge="text_known", token_lengths_known=True)
    observation = capture(adapter, batch, adapter.decode(batch.questions),
                          adapter.decode(batch.targets))
    observation.public_target_lengths[0] -= 1
    with pytest.raises(ValueError, match="differ"):
        Candidate(adapter, observation, 3)


def test_content_lengths_reject_noncanonical_slots():
    adapter, batch = fixture()
    no_eos = batch.targets.clone()
    no_eos[no_eos == adapter.eos] = 3
    with pytest.raises(ValueError, match="contain EOS"):
        adapter.content_lengths(no_eos)
    after_eos = batch.targets.clone()
    length = adapter.content_lengths(after_eos)[0]
    after_eos[0, length + 1] = 3
    with pytest.raises(ValueError, match="after EOS"):
        adapter.content_lengths(after_eos)


def test_known_lengths_dlg_checkpoint_resume(tmp_path):
    adapter, batch = fixture(token_lengths_known=True)
    observation = capture(adapter, batch, [], [])
    spec = AttackSpec(method="dlg_adapted", iterations=2, checkpoint_interval=1,
                      max_evaluations=20)
    first = AttackRunner(adapter, observation, spec).run(tmp_path)
    resumed = AttackRunner(adapter, observation, spec).run(tmp_path, resume=True)
    assert first.status == resumed.status == "completed"
    torch.testing.assert_close(first.images, resumed.images, rtol=0, atol=0)
    assert first.questions == resumed.questions and first.targets == resumed.targets


def test_replay_and_wrong_data():
    adapter, batch = fixture()
    obs = capture(adapter, batch, [], [])
    assert matching_loss(simulate_update(adapter, batch, adapter.training_spec), obs.tensors, "l2") == 0
    changed = replace(batch, images=1 - batch.images)
    assert matching_loss(simulate_update(adapter, changed, adapter.training_spec), obs.tensors, "l2") > 0


def test_budget_and_checkpoint_resume(tmp_path):
    adapter, batch = fixture()
    obs = capture(adapter, batch, [], [])
    spec = AttackSpec(iterations=4, checkpoint_interval=2, max_evaluations=20)
    first = AttackRunner(adapter, obs, spec).run(tmp_path)
    resumed = AttackRunner(adapter, obs, spec).run(tmp_path, resume=True)
    assert first.status == resumed.status == "completed"
    torch.testing.assert_close(first.images, resumed.images, rtol=0, atol=0)
    assert first.questions == resumed.questions and first.targets == resumed.targets
    assert first.costs["update_evaluations"] == resumed.costs["update_evaluations"]
    limited = AttackRunner(adapter, obs, replace(spec, max_evaluations=1)).run()
    assert limited.costs["update_evaluations"] == 1
    assert limited.reason == "budget_exhausted"
    unlimited = AttackRunner(
        adapter, obs, replace(spec, iterations=2, max_evaluations=None)).run()
    assert unlimited.reason == "iterations_completed"
    assert unlimited.costs["update_evaluations"] > 1
    with pytest.raises(ValueError, match="differs"):
        AttackRunner(adapter, obs, replace(spec, seed=50)).run(tmp_path, resume=True)


def test_attack_checkpoint_schema_v2_is_rejected(tmp_path):
    adapter, batch = fixture()
    observation = capture(adapter, batch, [], [])
    spec = AttackSpec(iterations=2, checkpoint_interval=1, max_evaluations=20)
    AttackRunner(adapter, observation, spec).run(tmp_path)
    pointer = read_json(tmp_path / "checkpoint.json")
    pointer["schema_version"] = 2
    (tmp_path / "checkpoint.json").write_text(json.dumps(pointer))
    with pytest.raises(ValueError, match="checkpoint schema"):
        AttackRunner(adapter, observation, spec).run(tmp_path, resume=True)


@pytest.mark.parametrize("method", ["dlg_adapted", "april_adapted", "gi_dqa_adapted", "random", "prior_only"])
def test_attack_variants_run(method):
    adapter, batch = fixture()
    obs = capture(adapter, batch, [], [])
    result = AttackRunner(adapter, obs, AttackSpec(method=method, iterations=4, checkpoint_interval=2)).run()
    if method == "april_adapted":
        assert result.status == "not_applicable"
        return
    assert result.status == "completed"
    assert torch.isfinite(result.images).all()
    if method in {"random", "prior_only"}:
        assert result.costs["update_evaluations"] == 0


def test_lamp_fixture_runs():
    adapter, batch = fixture()
    obs = capture(adapter, batch, [], [])
    spec = AttackSpec(text_method="lamp_adapted", prior_model="tiny-public-bigram", iterations=4,
                      checkpoint_interval=2, prior_interval=2)
    result = AttackRunner(adapter, obs, spec).run()
    assert result.status == "completed"
    assert result.costs["prior_evaluations"] > 0


def test_validation_rejects_ambiguous_protocol():
    with pytest.raises(ValueError, match="local_steps"):
        load_config(overrides=["training.local_steps=2"])
    with pytest.raises(ValueError, match="revision"):
        load_config(overrides=["model.family=llava"])
    with pytest.raises(ValueError, match="federated algorithm"):
        load_config(overrides=["training.algorithm=unknown"])


def test_implementation_specific_validation_is_owned_by_consumer():
    """Core config ignores inactive details; their owning runtime validates them."""
    connector = TrainingSpec(fine_tuning_strategy="f_c", lora_rank=0, lora_alpha=0)
    assert build_model(ModelSpec(), connector).training_spec == connector

    with pytest.raises(ValueError, match="LoRA rank"):
        build_model(ModelSpec(), TrainingSpec(fine_tuning_strategy="f_l", lora_rank=0))
    with pytest.raises(ValueError, match="dtype"):
        build_model(ModelSpec(dtype="float16"), TrainingSpec())
    with pytest.raises(ValueError, match="divisible"):
        build_model(ModelSpec(image_size=7, patch_size=4), TrainingSpec())

    adapter, batch = fixture()
    observation = capture(adapter, batch, [], [])
    with pytest.raises(ValueError, match="checkpoint_interval"):
        AttackRunner(adapter, observation, AttackSpec(checkpoint_interval=0))
    with pytest.raises(ValueError, match="prior_interval"):
        AttackRunner(adapter, observation,
                     AttackSpec(text_method="lamp_adapted", prior_interval=0))


def test_public_token_ids_do_not_depend_on_decode_roundtrip():
    adapter, batch = fixture(knowledge="text_known")
    batch.targets[0, 0] = 3  # An unknown token disappears from the decoded string.
    obs = capture(adapter, batch, adapter.decode(batch.questions), adapter.decode(batch.targets))
    candidate = Candidate(adapter, obs, 3)
    torch.testing.assert_close(candidate.batch(True).targets.argmax(-1), batch.targets)


def test_interrupted_resume_matches_continuous_run(tmp_path, monkeypatch):
    adapter, batch = fixture()
    obs = capture(adapter, batch, [], [])
    spec = AttackSpec(iterations=6, checkpoint_interval=2, max_evaluations=40,
                      text_method="lamp_adapted", prior_model="tiny-public-bigram", prior_interval=2)
    continuous = AttackRunner(adapter, obs, spec).run()
    real_save = AttackRunner.save_checkpoint

    def stop_after_commit(self, *args, **kwargs):
        real_save(self, *args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(AttackRunner, "save_checkpoint", stop_after_commit)
    with pytest.raises(KeyboardInterrupt):
        AttackRunner(adapter, obs, spec).run(tmp_path)
    monkeypatch.setattr(AttackRunner, "save_checkpoint", real_save)
    resumed = AttackRunner(adapter, obs, spec).run(tmp_path, resume=True)
    torch.testing.assert_close(continuous.images, resumed.images, atol=0, rtol=0)
    assert continuous.questions == resumed.questions and continuous.targets == resumed.targets
    assert continuous.history == resumed.history
    assert continuous.costs["update_evaluations"] == resumed.costs["update_evaluations"]


def test_lora_a_carries_no_signal_at_initialization():
    """PEFT starts B at zero, so the first federated round uploads only grad(B).

    With B = 0 the chain rule gives grad(A) = (alpha/r) * B^T G^T X = 0, while
    grad(B) = (alpha/r) * G^T X A^T is nonzero. Input activations X therefore
    reach the server only through the r-dimensional random projection A until B
    moves off zero. Locking this in because it bounds what a first-round attack
    on a LoRA upload can possibly recover.
    """
    training = TrainingSpec(fine_tuning_strategy="f_l")
    model, batch = fixture(strategy="f_l")
    update = simulate_update(model, batch, training)
    a_names = [n for n in update if "lora_A" in n]
    b_names = [n for n in update if "lora_B" in n]
    assert a_names and len(a_names) == len(b_names)

    lora_b = [p for n, p in model.named_parameters() if "lora_B" in n]
    assert all(float(p.abs().max()) == 0.0 for p in lora_b), "PEFT should initialize B at zero"
    assert all(float(update[n].abs().max()) == 0.0 for n in a_names), "grad(A) must vanish while B is zero"
    assert any(float(update[n].abs().max()) > 0.0 for n in b_names), "grad(B) must carry the signal"

    # Once B is off zero the A gradients become informative again.
    with torch.no_grad():
        for p in lora_b:
            p.add_(torch.full_like(p, 1e-3))
    moved = simulate_update(model, batch, training)
    assert any(float(moved[n].abs().max()) > 0.0 for n in a_names)


def test_upload_patterns_resolve_to_an_explicit_allowlist():
    names = ["a.lora_A.weight", "a.lora_B.weight", "b.lora_A.weight", "b.lora_B.weight"]
    assert resolve_upload(names, []) == sorted(names), "no mask uploads the whole trainable set"
    assert resolve_upload(names, ["*lora_B*"]) == ["a.lora_B.weight", "b.lora_B.weight"]
    assert resolve_upload(names, ["a.*", "*lora_B*"]) == [
        "a.lora_A.weight", "a.lora_B.weight", "b.lora_B.weight"], "patterns union, deduplicated"
    with pytest.raises(ValueError, match="matched no trainable parameter"):
        resolve_upload(names, ["vision.*"])
    with pytest.raises(ValueError, match="unique and nonempty"):
        mask_upload({"a": 1}, ["a", "a"])
    with pytest.raises(ValueError, match="unknown parameters"):
        mask_upload({"a": 1}, ["b"])


def test_masked_upload_shrinks_the_observation_and_still_inverts(tmp_path):
    """A client that uploads only grad(B) hands the attacker strictly less."""
    training = TrainingSpec(fine_tuning_strategy="f_l", upload_parameters=["*lora_B*"])
    model, batch = fixture(strategy="f_l")
    model.training_spec = training

    full = simulate_update(model, batch, replace(training, upload_parameters=[]))
    observation = capture(model, batch, model.decode(batch.questions), model.decode(batch.targets))

    assert set(observation.tensors) < set(full), "the mask must drop parameters"
    assert set(observation.tensors) == {n for n in full if "lora_B" in n}
    assert all(torch.equal(observation.tensors[n], full[n]) for n in observation.tensors), \
        "masking selects parameters; it must not alter their values"

    spec = AttackSpec(method="ig_adapted", iterations=2, checkpoint_interval=1, max_evaluations=8)
    result = AttackRunner(model, observation, spec).run()
    assert result.status == "completed"


def test_subset_observation_without_a_declared_mask_is_rejected():
    """Silently dropping parameters would misreport what the client uploaded."""
    model, batch = fixture(strategy="f_l")
    observation = capture(model, batch, model.decode(batch.questions), model.decode(batch.targets))
    dropped = sorted(observation.tensors)[0]
    observation.tensors = {k: v for k, v in observation.tensors.items() if k != dropped}
    spec = AttackSpec(method="ig_adapted", iterations=1, checkpoint_interval=1, max_evaluations=4)
    with pytest.raises(ValueError, match="do not match the configured trainable"):
        AttackRunner(model, observation, spec).run()
