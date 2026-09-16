from dataclasses import replace
import json

import pytest
import torch

from core.artifacts import read_json
from attacks import AttackRunner, Candidate, matching_loss, supports
from core.aggregation import apply_server_update, create_federated_algorithm
from core.config import AttackSpec, ModelSpec, TrainingSpec, load_config
from core.fl import (capture, load_observation, mask_upload, resolve_upload,
                     save_observation, simulate_update)
from core.vlm_wrapper import build_model


def fixture(mode="full", algorithm="fedsgd", steps=1, task="vqa", knowledge="private"):
    training = TrainingSpec(mode=mode, algorithm=algorithm, local_steps=steps,
                            task=task, knowledge=knowledge)
    adapter = build_model(ModelSpec(), training)
    batch = adapter.batch(torch.rand(steps, 3, 8, 8),
                          ["what color is the object"] * steps, ["red blue"] * steps)
    return adapter, batch


@pytest.mark.parametrize("mode", ["full", "llm_full", "lora_llm"])
@pytest.mark.parametrize("task", ["vqa", "caption"])
def test_one_step_sgd_identity_and_input_gradients(mode, task):
    adapter, batch = fixture(mode=mode, task=task)
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
    if mode == "lora_llm":
        assert all("lora_" in name and "language" in name for name in gradient)
        assert all(g.count_nonzero() == 0 for name, g in gradient.items() if "lora_A" in name)
        assert any(g.count_nonzero() > 0 for name, g in gradient.items() if "lora_B" in name)
        assert supports("april_adapted", adapter, obs).status == "not_applicable"


@pytest.mark.parametrize("mode", ["full", "lora_llm"])
def test_multistep_matches_actual_optimizer(mode):
    adapter, batch = fixture(mode, "fedavg", steps=3)
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


@pytest.mark.parametrize("method", ["fedsgd", "fedavg"])
def test_federated_algorithms_use_same_initial_state_and_weights(method):
    adapter, batch = fixture("lora_llm", "fedavg", steps=2)
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
    meta = read_json(tmp_path / "observation.json")
    meta["ground_truth"] = "forbidden"
    (tmp_path / "observation.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="unknown"):
        load_observation(tmp_path)


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
    with pytest.raises(ValueError, match="differs"):
        AttackRunner(adapter, obs, replace(spec, seed=50)).run(tmp_path, resume=True)


@pytest.mark.parametrize("method", ["dlg_adapted", "april_adapted", "gi_dqa_adapted", "random", "prior_only"])
def test_attack_variants_run(method):
    adapter, batch = fixture()
    obs = capture(adapter, batch, [], [])
    result = AttackRunner(adapter, obs, AttackSpec(method=method, iterations=4, checkpoint_interval=2)).run()
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
    full = TrainingSpec(mode="full", lora_rank=0, lora_alpha=0)
    assert build_model(ModelSpec(), full).training_spec == full

    with pytest.raises(ValueError, match="LoRA rank"):
        build_model(ModelSpec(), TrainingSpec(mode="lora_llm", lora_rank=0))
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
    training = TrainingSpec(mode="lora_llm")
    model, batch = fixture(mode="lora_llm")
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
    training = TrainingSpec(mode="lora_llm", upload_parameters=["*lora_B*"])
    model, batch = fixture(mode="lora_llm")
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
    model, batch = fixture(mode="lora_llm")
    observation = capture(model, batch, model.decode(batch.questions), model.decode(batch.targets))
    dropped = sorted(observation.tensors)[0]
    observation.tensors = {k: v for k, v in observation.tensors.items() if k != dropped}
    spec = AttackSpec(method="ig_adapted", iterations=1, checkpoint_interval=1, max_evaluations=4)
    with pytest.raises(ValueError, match="do not match the configured trainable"):
        AttackRunner(model, observation, spec).run()
