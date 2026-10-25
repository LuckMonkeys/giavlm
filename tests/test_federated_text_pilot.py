import pytest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from core.artifacts import write_json
from core.config import validate_attack_config
from core.experiment import protocol_config
from defenses.factory import create_defense
from evaluation.federated_text_pilot import (
    CONDITIONS, aggregate_rows, capture_pilot, evaluate_pilot, pilot_attack, select_cohort,
    select_pilot, validate_upload_groups,
)


def test_pilot_balances_length_without_selecting_by_reconstruction():
    frozen = {"selected": [{"sample_id": str(i), "length_bin": b}
                            for i, b in enumerate(["1-16"] * 10 + ["17-32"] * 10 + ["33-64"] * 10)]}
    selected = select_pilot(frozen)
    assert [s["sample_id"] for s in selected] == ["0", "10", "1", "11"]
    assert [s["length_bin"] for s in selected] == ["1-16", "17-32"] * 2
    for image in ("gt", "random"):
        single, pair = pilot_attack(image, 1), pilot_attack(image, 2)
        validate_attack_config(single)
        validate_attack_config(pair)
        assert pair.max_evaluations == 2 * single.max_evaluations
        assert pair.seconds == 2 * single.seconds


def test_pilot_requires_all_commits_before_scoring(tmp_path, monkeypatch):
    write_json(tmp_path / "plan.json", {"jobs": [{"id": "a"}, {"id": "b"}]})
    write_json(tmp_path / "runs/a/attacks/gt/result.json", {"status": "completed"})
    def forbidden(*args, **kwargs):
        raise AssertionError("Truth opened before every reconstruction was committed")
    monkeypatch.setattr("evaluation.federated_text_pilot.evaluate", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_pilot(tmp_path)


def test_slake_expansion_preserves_all_frozen_samples_and_order():
    frozen = {"dataset": "slake", "selected": [
        {"sample_id": str(i), "length_bin": b}
        for i, b in enumerate(["33-64"] * 10 + ["17-32"] * 10 + ["1-16"] * 10)]}
    selected = select_cohort(frozen, "slake30")
    assert selected == frozen["selected"] and selected is not frozen["selected"]
    assert sum(len(selected) // (batch * steps) for _, _, batch, steps, _ in CONDITIONS) == 165
    frozen["selected"][1]["sample_id"] = "0"
    with pytest.raises(ValueError, match="distinct QA"):
        select_cohort(frozen, "slake30")


def test_expansion_rejects_mixed_client_uploads_and_partial_capture(tmp_path):
    validate_upload_groups([{"client": 0}, {"client": 0}, {"client": 1}, {"client": 1}])
    with pytest.raises(ValueError, match="different clients"):
        validate_upload_groups([{"client": 0}, {"client": 1}])
    with pytest.raises(ValueError, match="even"):
        validate_upload_groups([{"client": 0}])
    with pytest.raises(ValueError, match="threshold fixed"):
        capture_pilot(tmp_path, None, None, "slake30")
    (tmp_path / "runs").mkdir()
    with pytest.raises(FileExistsError):
        capture_pilot(tmp_path, None, None)


def test_length_aggregation_keeps_failed_samples_in_denominator():
    rows = [dict(condition="batch", image="random", job="a", length_bin="1-16",
                 status="completed", joint_em=1., full_coverage=True, candidate_recall=1.),
            dict(condition="batch", image="random", job="a", length_bin="1-16",
                 status="budget_exhausted", joint_em=0., full_coverage=True, candidate_recall=1.),
            dict(condition="batch", image="random", job="b", length_bin="33-64",
                 status="completed", joint_em=0., full_coverage=False, candidate_recall=.5)]
    overall = aggregate_rows(rows)[0]
    assert overall["n"] == 3 and overall["uploads"] == 2
    assert overall["completed_samples"] == 2 and overall["joint_em"] == pytest.approx(1 / 3)
    short, long = aggregate_rows(rows, by_length=True)
    assert short["n"] == 2 and short["joint_em"] == .5
    assert long["n"] == 1 and long["joint_em"] == 0


def test_experimental_options_are_reachable_through_hydra():
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        config = compose(config_name="config", overrides=[
            "attack=dager_adapted", "attack.dager.pooled_updates=true",
            "attack.dager.defense_mode=unaware", "defense=gaussian_noise"])
    protocol = protocol_config(config)
    assert protocol.attack.dager.pooled_updates
    assert protocol.attack.dager.defense_mode == "unaware"
    defense = create_defense(OmegaConf.to_container(config.defense, resolve=True))
    assert defense.noise_ratio == .01
