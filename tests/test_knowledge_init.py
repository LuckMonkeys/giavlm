from dataclasses import replace

import pytest
import torch

from attacks.init import perturb_tokens
from core.artifacts import read_json
from core.config import TrainingSpec, load_config
from core.data import synthetic
from core.fl import capture, load_observation, save_observation
from tests.test_protocol import fixture
from tests.test_workflows import command


def options(*items):
    return [argument for item in items for argument in ("--set", item)]


def settings(knowledge, text_method="tag_adapted"):
    return ["training.fine_tuning_strategy=f_c", f"training.knowledge={knowledge}",
            "attack.iterations=2", "attack.checkpoint_interval=1",
            f"attack.text_method={text_method}"]


@pytest.fixture
def manifest(tmp_path):
    return synthetic(tmp_path / "data", 12, clients=1)


def run(manifest, tmp_path, knowledge, *overrides, text_method="tag_adapted"):
    items = [*settings(knowledge, text_method), *overrides]
    capture_dir = tmp_path / "capture"
    command("capture", "--data", str(manifest), "--output", str(capture_dir),
            *options(*settings(knowledge, text_method)))
    attack_dir = tmp_path / "attack"
    command("attack", "--observation", str(capture_dir / "public"), "--output", str(attack_dir),
            *options(*items))
    command("evaluate", "--reconstruction", str(attack_dir), "--truth", str(capture_dir / "private"),
            *options(*items, "evaluation.trajectory=true"))
    return read_json(attack_dir / "result.json"), read_json(attack_dir / "evaluation.json"), attack_dir


def test_public_image_observation_is_validated_and_integrity_checked(tmp_path):
    adapter, batch = fixture(knowledge="image_known")
    observation = capture(adapter, batch, [], [])
    with pytest.raises(ValueError, match="Missing or malformed declared public images"):
        replace(observation, public_images=None).validate()
    private = replace(observation, training=replace(observation.training, knowledge="private"))
    with pytest.raises(ValueError, match="Private images appeared"):
        private.validate()
    save_observation(tmp_path, observation)
    from core.artifacts import write_tensors
    write_tensors(tmp_path / "public_images.safetensors", {"images": torch.zeros(1, 3, 8, 8)})
    with pytest.raises(ValueError, match="Public image integrity"):
        load_observation(tmp_path)


@pytest.mark.parametrize("knowledge,scored", [
    ("image_known", ["question", "target"]),
    ("image_question_known", ["target"]),
])
def test_public_image_conditions_attack_and_score_only_text(manifest, tmp_path, knowledge, scored):
    result, report, attack_dir = run(manifest, tmp_path, knowledge)
    assert result["status"] == "completed" and result["condition"]["knowledge"] == knowledge
    assert not (attack_dir / "images.safetensors").exists()
    assert not (attack_dir / "init.safetensors").exists()
    assert report["scored_fields"] == scored and report["trajectory"] is None
    metrics = report["samples"][0]["metrics"]
    assert "psnr" not in metrics and "init_psnr" not in metrics
    assert ("question_rouge1" in metrics) == ("question" in scored)
    assert "target_rouge1" in metrics and "init_target_rouge1" in metrics


def test_text_known_evaluation_scores_only_the_image(manifest, tmp_path):
    _, report, _ = run(manifest, tmp_path, "text_known", text_method="none")
    assert report["scored_fields"] == ["image"]
    assert not any(key.startswith(("question_", "target_", "init_target"))
                   for key in report["samples"][0]["metrics"])


def test_private_reference_text_start_decodes_to_the_truth(manifest, tmp_path):
    result, report, attack_dir = run(manifest, tmp_path, "image_known",
                                     "attack.init_text_source=private_reference")
    condition = result["condition"]
    assert condition["init_text_source"] == "private_reference" and condition["init_text_sha256"]
    metrics = report["samples"][0]["metrics"]
    assert metrics["init_question_exact_match"] == 1.0 and metrics["init_target_exact_match"] == 1.0
    assert read_json(attack_dir / "init.json")["targets"]


def test_public_text_template_starts_only_its_field(manifest, tmp_path):
    result, report, attack_dir = run(manifest, tmp_path, "image_question_known",
                                     "attack.init_text_source=public_text",
                                     "attack.init_target=red circle")
    assert result["provenance"]["initialization"]["text_source"] == "public_text"
    assert read_json(attack_dir / "init.json") == {"questions": [], "targets": ["red circle"]}


def test_token_replacement_keeps_special_positions():
    ids = torch.tensor([[5, 6, 7, 2, 0, 0]])
    perturbed = perturb_tokens(ids, "replace", 1.0, 0, 30, {0, 1, 2, 3})
    assert perturbed[0, 3:].tolist() == [2, 0, 0]
    assert all(token not in {0, 1, 2, 3} for token in perturbed[0, :3].tolist())
    assert torch.equal(perturb_tokens(ids, "none", 0.0, 0, 30, {0, 1, 2, 3}), ids)


@pytest.mark.parametrize("knowledge,overrides,message", [
    ("image_known", ["attack.init_source=private_reference"], "needs a private image"),
    ("text_known", ["attack.init_text_source=private_reference"], "needs private text"),
    ("question_known", ["attack.init_text_source=public_text", "attack.init_question=what"],
     "private VQA question"),
    ("private", ["attack.init_text_source=public_text"], "requires init_question or init_target"),
    ("private", ["attack.init_target=red"], "Random text initialization"),
    ("private", ["attack.init_text_source=private_reference", "attack.init_target=red"],
     "comes from the capture"),
    ("private", ["attack.init_text_source=private_reference",
                 "attack.init_text_perturbation=replace"], r"\(0, 1\]"),
    ("image_known", ["attack.text_method=none"], "explicit reconstruction component"),
])
def test_initialization_matches_the_knowledge_condition(knowledge, overrides, message):
    base = [item for item in settings(knowledge) if not item.startswith("attack.text_method")
            or not any(o.startswith("attack.text_method") for o in overrides)]
    with pytest.raises(ValueError, match=message):
        load_config(overrides=[*base, *overrides])


def test_caption_rejects_question_only_knowledge():
    for knowledge in ["question_known", "image_question_known"]:
        with pytest.raises(ValueError, match="Caption"):
            load_config(overrides=[f"training.knowledge={knowledge}", "training.task=caption"])
    assert TrainingSpec(task="caption", knowledge="image_known").private_text


@pytest.mark.parametrize("image_dtype", ["float32", "bfloat16"])
def test_candidate_images_keep_their_own_precision(image_dtype):
    from attacks.optim.engine import Candidate
    from core.fl import simulate_update
    adapter, batch = fixture(knowledge="text_known", dtype="float64")
    observation = capture(adapter, batch, adapter.decode(batch.questions),
                          adapter.decode(batch.targets))
    candidate = Candidate(adapter, observation, 0, image_dtype=getattr(torch, image_dtype))
    assert candidate.images.dtype == getattr(torch, image_dtype)
    replayed = candidate.batch()
    assert replayed.images.dtype == torch.float64  # cast to the victim only in the forward pass
    loss = sum(value.square().sum() for value in
               simulate_update(adapter, replayed, observation.training, True).values())
    gradient, = torch.autograd.grad(loss, candidate.images)
    assert gradient.dtype == candidate.images.dtype and gradient.abs().sum() > 0
    with pytest.raises(ValueError, match="attack image dtype"):
        load_config(overrides=["attack.image_dtype=int8"])


def test_bfloat16_candidates_lose_small_steps_that_float32_keeps():
    step = torch.tensor(0.001)
    for dtype, kept in [(torch.float32, True), (torch.bfloat16, False)]:
        pixel = torch.tensor(0.75, dtype=dtype)
        assert bool(pixel - step.to(dtype) != pixel) == kept
