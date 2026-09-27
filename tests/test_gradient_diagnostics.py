import numpy as np
from PIL import Image
import pytest
import torch

from core.artifacts import read_json
from core.config import load_config
from core.data import synthetic
from evaluation.gradient_diagnostics import diagnose
from tests.test_workflows import command

TRAINING = ["training.fine_tuning_strategy=f_c", "training.knowledge=text_known"]
ATTACK = [*TRAINING, "attack.iterations=2", "attack.checkpoint_interval=1", "attack.text_method=none"]


def options(*items):
    return [argument for item in items for argument in ("--set", item)]


@pytest.fixture
def captures(tmp_path):
    manifest = synthetic(tmp_path / "data", 12, clients=1)
    paths = []
    for offset in range(2):
        path = tmp_path / f"run-{offset}" / "capture"
        command("capture", "--data", str(manifest), "--output", str(path),
                "--client", "0", "--offset", str(offset), *options(*TRAINING))
        paths.append(path)
    return paths


def attack_and_evaluate(capture, output, *overrides):
    command("attack", "--observation", str(capture / "public"), "--output", str(output),
            *options(*ATTACK, *overrides))
    command("evaluate", "--reconstruction", str(output), "--truth", str(capture / "private"),
            *options(*ATTACK, *overrides, "evaluation.trajectory=true"))
    return read_json(output / "result.json"), read_json(output / "evaluation.json")


def test_truth_replays_the_observed_update(captures):
    report = diagnose(captures[0], other_captures=[captures[1]], gaussian_levels=(0.1,),
                      blur_levels=(1.0,), mix_levels=(0.5, 1.0))
    assert report["uses_private_reference"] is True
    assert report["data_path"] == {"image_max_abs_error": 0.0, "public_questions_match": True,
                                   "public_targets_match": True}
    rows = {(row["family"], row["level"]): row for row in report["variants"]}
    truth = rows["truth", None]
    assert truth["cosine_loss"] < 1e-6 and truth["max_abs_error"] < 1e-6
    assert truth["connector_input"]["cosine"] == pytest.approx(1.0)
    assert truth["image"][0]["psnr_infinite"]
    assert rows["uniform_mix", 1.0]["cosine_loss"] > truth["cosine_loss"]
    assert rows["other_image", "run-1"]["relative_l2"] > truth["relative_l2"]


@pytest.mark.parametrize("overrides,message", [
    (["attack.init_images=x.png"], "Random initialization"),
    (["attack.init_perturbation=blur", "attack.init_level=1"], "Random initialization"),
    (["attack.init_source=public_image"], "requires attack.init_images"),
    (["attack.init_source=private_reference", "attack.init_level=0.5"], "requires an init_perturbation"),
    (["attack.init_source=private_reference", "attack.init_perturbation=uniform_mix",
      "attack.init_level=2"], r"\[0, 1\]"),
    (["attack.init_source=unknown"], "Unknown attack init source"),
])
def test_initialization_config_is_validated(overrides, message):
    with pytest.raises(ValueError, match=message):
        load_config(overrides=[*ATTACK, *overrides])


def test_random_start_is_recorded_for_evaluation(captures, tmp_path):
    result, report = attack_and_evaluate(captures[0], tmp_path / "attack")
    assert result["condition"]["init_source"] == "random"
    assert result["provenance"]["initialization"]["images_sha256"] is None
    assert "init_psnr" in report["samples"][0]["metrics"]
    assert report["init_near_reference"] is None
    assert [row["iteration"] for row in report["trajectory"]] == [1, 2]


def test_private_reference_start_is_labeled(captures, tmp_path):
    result, report = attack_and_evaluate(
        captures[0], tmp_path / "attack", "attack.init_source=private_reference",
        "attack.init_perturbation=uniform_mix", "attack.init_level=0.25")
    condition = result["condition"]
    assert (condition["init_source"], condition["init_perturbation"], condition["init_level"]) == (
        "private_reference", "uniform_mix", 0.25)
    assert condition["init_images_sha256"] == result["provenance"]["initialization"]["images_sha256"]
    # The start is the perturbed reference, not the reference itself.
    assert 0 < report["samples"][0]["metrics"]["init_mse"] < 0.1


def test_public_image_start_and_near_reference_flag(captures, tmp_path):
    template = tmp_path / "template.png"
    Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(template)
    _, report = attack_and_evaluate(captures[0], tmp_path / "public", "attack.init_source=public_image",
                                    f"attack.init_images={template}")
    assert report["init_near_reference"] is False
    # A "public" file that is really the private reference is flagged, not trusted.
    leaked = captures[0] / "private" / "images.safetensors"
    _, report = attack_and_evaluate(captures[0], tmp_path / "leaked", "attack.init_source=public_image",
                                    f"attack.init_images={leaked}")
    assert report["init_near_reference"] is True


def test_runner_requires_images_exactly_for_image_sources(captures):
    from attacks.optim.engine import AttackRunner
    from core.fl import load_observation, restore_model

    observation = load_observation(captures[0] / "public")
    adapter = restore_model(captures[0] / "public" / "model")
    adapter.set_training_spec(observation.training)
    random_spec = load_config(overrides=ATTACK).attack
    with pytest.raises(ValueError, match="exactly when"):
        AttackRunner(adapter, observation, random_spec).run(initial_images=torch.rand(1, 3, 8, 8))
    image_spec = load_config(overrides=[*ATTACK, "attack.init_source=private_reference"]).attack
    with pytest.raises(ValueError, match="exactly when"):
        AttackRunner(adapter, observation, image_spec).run()
