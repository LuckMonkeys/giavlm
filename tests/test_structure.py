from dataclasses import replace
from pathlib import Path
import re

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from core.artifacts import read_json
from core.config import AttackSpec, ModelSpec, TrainingSpec
from core.experiment import ExperimentRunner, protocol_config, run_experiment
from core.fl import NamedGradientAccumulator, capture, simulate_secure_aggregation
from core.knowledge import AdversaryKnowledge
from core.vlm_wrapper import build_model
from attacks.factory import create_attacker
from attacks.registry import IMPLEMENTED, METHODS, NOT_APPLICABLE, UNIMPLEMENTED
from defenses.factory import create_defense
from metrics.text import pii_exact_match_recall, token_set_f1
from utils.run_cmds import materialize_jobs


ROOT = Path(__file__).resolve().parents[1]


def configuration(tmp_path, *overrides, name="config"):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return compose(config_name=name, overrides=[f"output_dir={tmp_path}",
                        "attack.iterations=2", "attack.checkpoint_interval=1", *overrides])


@pytest.mark.parametrize("model,family", [("tiny_llava", "tiny"), ("llava", "llava"),
                                          ("blip2", "blip2"), ("qwen_vl", "qwen2_5_vl")])
def test_hydra_model_composition(tmp_path, model, family):
    config = configuration(tmp_path, f"model={model}", "fed=fedavg", "tuning=f_cl")
    protocol = protocol_config(config)
    assert protocol.model.family == family
    assert protocol.training.algorithm == "fedavg"
    assert protocol.training.local_optimizer == "adamw"
    assert protocol.training.lr == 2e-5
    assert protocol.training.local_steps == 2
    assert protocol.training.fine_tuning_strategy == "f_cl"


def test_hydra_fedavg_sgd_preset(tmp_path):
    config = configuration(tmp_path, "fed=fedavg_sgd")
    protocol = protocol_config(config)
    assert protocol.training.algorithm == "fedavg"
    assert protocol.training.local_optimizer == "sgd"
    assert protocol.training.local_steps == 2
    assert protocol.training.training_protocol == "native-sft-v2"
    assert protocol.training.fine_tuning_strategy == "f_l"
    assert "adam_beta1" not in config.fed


def test_optimizer_specific_hydra_fields(tmp_path):
    fedsgd = configuration(tmp_path, "fed=fedsgd").fed
    fedavg = configuration(tmp_path, "fed=fedavg").fed
    assert "adam_beta1" not in fedsgd
    assert fedavg.adam_beta1 == 0.9
    assert fedavg.adam_beta2 == 0.999
    assert fedavg.adam_epsilon == 1e-8


def test_preset_and_knowledge_validation(tmp_path):
    config = configuration(tmp_path, name="coco_captions_blip2_ig_private")
    assert protocol_config(config).training.task == "caption"
    config.knowledge.name = "question_known"
    with pytest.raises(ValueError, match="Caption"):
        protocol_config(config)

    known = protocol_config(configuration(tmp_path, "knowledge=private_lengths_known"))
    assert known.training.knowledge == "private"
    assert known.training.token_lengths_known is True


def test_slake_llava_dlg_preset_uses_bfloat16_fedsgd(tmp_path):
    config = configuration(tmp_path, name="slake_llava_dlg_private")
    protocol = protocol_config(config)
    assert protocol.model.family == "llava"
    assert protocol.model.dtype == "bfloat16"
    assert protocol.training.task == "vqa"
    assert protocol.training.fine_tuning_strategy == "f_cl"
    assert protocol.training.algorithm == "fedsgd"
    assert protocol.attack.method == "dlg_adapted"
    assert protocol.attack.max_evaluations is None


@pytest.mark.parametrize("override", ["fed.secure_aggregation=true",
                                      "knowledge.template_known=true", "defense=safe_template"])
def test_unimplemented_protocols_fail_closed(tmp_path, override):
    with pytest.raises(NotImplementedError):
        protocol_config(configuration(tmp_path, override))


def test_hydra_run_resume_and_integrity(tmp_path):
    config = configuration(tmp_path, "num_runs=2", "start_run_id=1")
    result = run_experiment(config)
    assert [r["run_id"] for r in result["runs"]] == [1]
    assert result["runs"][0]["status"] == "completed"
    assert (tmp_path / "model/model.safetensors").exists()
    config.resume = True
    config.start_run_id = 0
    result = run_experiment(config)
    assert [r["run_id"] for r in result["runs"]] == [0, 1]
    before = (tmp_path / result["runs"][0]["result"]).stat().st_mtime_ns
    run_experiment(config)
    assert (tmp_path / result["runs"][0]["result"]).stat().st_mtime_ns == before
    config.attack.lr = 0.2
    with pytest.raises(ValueError, match="Resume protocol"):
        run_experiment(config)


def test_federated_training_in_hydra_entry(tmp_path):
    result = run_experiment(configuration(
        tmp_path, "fed.rounds=1", "fed=fedavg", "tuning=f_cl"))
    assert result["runs"][0]["status"] == "completed"
    state = read_json(tmp_path / "federation/training.json")
    assert state["round"] == 1
    assert state["history"][0]["algorithm"] == "fedavg"
    assert state["history"][0]["fine_tuning_strategy"] == "f_cl"


def test_two_stage_federation_switches_parameter_groups(tmp_path):
    run_experiment(configuration(
        tmp_path, "fed.rounds=2", "fed=fedavg", "tuning=f_2stage",
        "tuning.two_stage_connector_rounds=1"))
    state = read_json(tmp_path / "federation/training.json")
    assert [row["fine_tuning_stage"] for row in state["history"]] == ["connector", "llm"]
    model = read_json(tmp_path / "federation" / state["checkpoint"] / "model.json")
    assert model["training"]["server_round"] == 2
    assert model["training"]["fine_tuning_strategy"] == "f_2stage"
    observation = read_json(tmp_path / "run-00000/capture/public/observation.json")
    assert observation["training"]["server_round"] == 2
    assert observation["training"]["fine_tuning_strategy"] == "f_2stage"
    assert all("lora_" in name for name in observation["parameter_names"])


def test_hydra_federation_can_start_from_snapshot(tmp_path):
    source_output = tmp_path / "source"
    run_experiment(configuration(source_output))
    snapshot = source_output / "model"

    warm_output = tmp_path / "warm"
    config = configuration(warm_output, "fed.rounds=1")
    config.model_snapshot = str(snapshot)
    result = run_experiment(config)

    training = read_json(warm_output / "federation/training.json")
    source = read_json(snapshot / "model.json")
    imported = read_json(warm_output / "federation/round-0000/model.json")
    trained = read_json(warm_output / "federation" / training["checkpoint"] / "model.json")
    assert result["runs"][0]["status"] == "completed"
    assert training["round"] == 1
    assert training["initialization"]["kind"] == "snapshot"
    assert training["initialization"]["fingerprint"] == source["fingerprint"]
    assert imported["fingerprint"] == source["fingerprint"]
    assert trained["fingerprint"] != source["fingerprint"]


def test_run_failure_is_saved_once_and_stops_experiment(tmp_path, monkeypatch):
    runner = ExperimentRunner(configuration(tmp_path, "num_runs=2"))
    attempts = []

    def fail(run_id):
        attempts.append(run_id)
        raise RuntimeError("injected failure")

    monkeypatch.setattr(runner, "_run_one", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.run_experiments()
    assert attempts == [0]
    assert read_json(tmp_path / "experiment.json")["runs"] == [
        {"run_id": 0, "status": "error", "reason": "injected failure"}]


def test_named_alignment_is_atomic():
    names = ["layer.lora_A.target.weight", "layer.lora_B.target.weight"]
    accumulator = NamedGradientAccumulator([torch.tensor([1.]), torch.tensor([3.])], names)
    accumulator.add_client([torch.tensor([5.]), torch.tensor([7.])],
                           ["layer.lora_B.benign.weight", "layer.lora_A.benign.weight"])
    tensors, metadata = accumulator.finalize()
    assert [v.item() for v in tensors] == [4., 4.]
    assert metadata["num_clients"] == 2
    with pytest.raises(ValueError, match="shape"):
        accumulator.add_client([torch.tensor([10.]), torch.ones(2)], names)
    assert [v.item() for v in accumulator.finalize()[0]] == [4., 4.]


def test_aggregate_requires_shared_basis():
    updates = [{"x": torch.tensor([1.])}, {"x": torch.tensor([3.])}]
    assert simulate_secure_aggregation(updates, ["same", "same"])["x"].item() == 2
    with pytest.raises(ValueError, match="basis"):
        simulate_secure_aggregation(updates, ["one", "two"])


@pytest.mark.parametrize("config", [{"name": "none"}, {"name": "clipping", "max_norm": 2.},
    {"name": "gaussian_dp", "noise_multiplier": 0.}, {"name": "topk_sparsify", "ratio": .5},
    {"name": "sign_sgd"}])
def test_defenses_preserve_names_and_inputs(config):
    value = torch.tensor([3., -4.])
    output = create_defense(config).apply({"x": value})
    assert set(output) == {"x"}
    assert torch.equal(value, torch.tensor([3., -4.]))
    if config["name"] == "clipping":
        assert output["x"].norm().item() == pytest.approx(2.)
    elif config["name"] == "topk_sparsify":
        assert torch.equal(output["x"], torch.tensor([0., -4.]))


def test_defense_is_in_report_condition(tmp_path):
    result = run_experiment(configuration(tmp_path, "defense=clipping"))
    report = read_json(tmp_path / result["runs"][0]["result"])
    assert report["condition"]["defense"]["name"] == "clipping"
    assert report["condition"]["attack_policy"] == "defense_unaware_raw_update_matching"


def test_registry_describes_every_factory_name():
    """METHODS is the benchmark surface; the factory must not accept an undescribed name."""
    source = Path(__file__).resolve().parent.parent / "attacks" / "factory.py"
    accepted = set(re.findall(r'name == "([a-z0-9_]+)"', source.read_text()))
    accepted |= set(re.findall(r'name in \{([^}]*)\}', source.read_text())[0].replace('"', "").replace(" ", "").split(","))
    assert accepted, "factory name extraction failed"
    assert accepted <= set(METHODS), f"undescribed factory names: {sorted(accepted - set(METHODS))}"
    assert set(METHODS) == set(IMPLEMENTED) | set(UNIMPLEMENTED) | set(NOT_APPLICABLE)
    assert not (set(IMPLEMENTED) & set(UNIMPLEMENTED))


def test_factory_does_not_alias_closed_form_april():
    model = build_model(ModelSpec(), TrainingSpec())
    batch = model.batch(torch.rand(1, 3, 8, 8), ["what color"], ["red"])
    observation = capture(model, batch, [], [])
    attacker = create_attacker(model, AttackSpec(method="april"))
    result = attacker.attack(observation.tensors, observation, AdversaryKnowledge())
    assert result.status == "not_implemented"
    with pytest.raises(ValueError, match="knowledge differs"):
        attacker.attack(observation.tensors, observation, AdversaryKnowledge("text_known"))
    with pytest.raises(ValueError, match="token-length knowledge differs"):
        attacker.attack(observation.tensors, observation,
                        AdversaryKnowledge(token_lengths_known=True))
    with pytest.raises(ValueError, match="Unknown attack"):
        create_attacker(model, replace(AttackSpec(), method="typo"))


def test_scheduler_is_argv_only(tmp_path):
    jobs = materialize_jobs(ROOT / "run_yaml/tiny_smoke.yaml", tmp_path)
    assert len(jobs) == 2
    assert jobs[0]["argv"][1:3] == ["-m", "examples.run_attack"]
    assert all(isinstance(part, str) for job in jobs for part in job["argv"])


def test_text_metric_helpers():
    assert token_set_f1([1, 2, 2], [2, 3]) == .5
    assert pii_exact_match_recall(["CANARY-123"], "CANARY-1234") == 0
    assert pii_exact_match_recall(["CANARY-123"], "ID: CANARY-123.") == 1
    assert pii_exact_match_recall([], "anything") is None


def test_hydra_config_roundtrip(tmp_path):
    config = configuration(tmp_path)
    roundtrip = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    assert protocol_config(roundtrip) == protocol_config(config)


def test_attack_oom_is_recorded_and_stops_experiment(tmp_path, monkeypatch):
    from attacks.base import OptimizationAttacker

    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError("injected attack OOM")

    monkeypatch.setattr(OptimizationAttacker, "attack", fail)
    with pytest.raises(torch.OutOfMemoryError):
        run_experiment(configuration(tmp_path))
    assert read_json(tmp_path / "experiment.json")["runs"][0]["status"] == "error"
    assert not (tmp_path / "run-00000/attack/result.json").exists()


def test_paired_prior_only_deltas():
    from copy import deepcopy
    from evaluation.compare import prior_only_deltas
    baseline = {"condition": {"method": "prior_only", "seed": 0}, "status": "completed",
                "observation_id": "same", "evaluation_config": {},
                "samples": [{"sample_id": "a", "metrics": {"psnr": 10., "wer": 1., "flag": False}}]}
    attacked = deepcopy(baseline)
    attacked["condition"]["method"] = "ig_adapted"
    attacked["samples"][0]["metrics"].update(psnr=12., wer=.5)
    result = prior_only_deltas(attacked, baseline)
    assert result["samples"][0]["delta"] == {"psnr": 2., "wer": -.5}
    attacked["observation_id"] = "other"
    with pytest.raises(ValueError, match="observation_id"):
        prior_only_deltas(attacked, baseline)


def test_noise_uses_injected_private_generator():
    defense = create_defense({"name": "gaussian_dp", "noise_multiplier": .2})
    updates = {"x": torch.ones(8)}
    first = defense.apply(updates, generator=torch.Generator().manual_seed(91))
    second = defense.apply(updates, generator=torch.Generator().manual_seed(91))
    assert torch.equal(first["x"], second["x"])
    assert not torch.equal(first["x"], updates["x"] / updates["x"].norm())
