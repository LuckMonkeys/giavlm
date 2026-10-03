import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from core.artifacts import read_json, write_json
from core.config import KNOWLEDGE_FIELDS, load_config
from core.data import synthetic
from core.vlm_wrapper import build_model
from evaluation.discriminability_metrics import replay
from evaluation.text_unknown_discriminability import (execute, load_spec, observation_for,
                                                       text_guesses, worker_binding)


def tiny_spec(manifest):
    return {
        "schema_version": 1,
        "manifest": str(manifest),
        "seed": 42,
        "pilot_count": 2,
        "development_count": 2,
        "validation_count": 2,
        "resources": {"gpu_id": 4, "min_free_mib": 1},
        "protocol": {
            "model": {"image_size": 14, "patch_size": 2},
            "training": {"algorithm": "fedsgd", "task": "vqa", "knowledge": "private",
                         "clients": 1, "clients_per_round": 1},
        },
        "conditions": [{"name": "fc", "strategy": "f_c", "round": 0}],
        "knowledge_conditions": ["text_known", "question_known", "target_known", "private"],
        "candidate_ids": ["truth", "uniform_mix-0-0.001", "uniform_mix-0-0.01",
                          "uniform_mix-0-0.1", "uniform_mix-0-1", "constant-0.5"],
        "text_guesses": {
            "per_family": 1,
            "template_pairs": [{"question": "what is visible?", "target": "normal"}],
            "question_words": ["what", "where", "image"],
            "target_words": ["yes", "no", "unknown"],
            "question_word_count": 3,
            "target_word_count": 2,
        },
    }


@pytest.fixture
def args(tmp_path):
    manifest = synthetic(tmp_path / "data", 80, clients=1)
    path = tmp_path / "spec.json"
    write_json(path, tiny_spec(manifest))
    return SimpleNamespace(spec=str(path), output=str(tmp_path / "study"), stage="prepare",
                           cohort="pilot", model_condition=None, device="cpu", resume=False)


def test_target_known_is_a_declared_public_field():
    assert KNOWLEDGE_FIELDS["target_known"] == frozenset({"target"})
    training = load_config(overrides=["training.knowledge=target_known"]).training
    assert training.target_public and training.question_private and not training.image_public


def test_real_diagnostic_spec_keeps_yaml_boolean_words_as_strings():
    path = Path(__file__).parents[1] / "configs/diagnostics/slake_llava_text_unknown_discriminability.yaml"
    guesses = load_spec(path)["text_guesses"]
    assert all(isinstance(word, str) for word in guesses["target_words"])


def test_text_guesses_use_only_declared_known_fields(args):
    spec = load_spec(args.spec)
    cfg = load_config(overrides=["training.knowledge=private",
                                 "training.fine_tuning_strategy=f_c"])
    adapter = build_model(cfg.model, cfg.training)
    images = adapter.prepare_image(Image.new("RGB", (8, 8))).float()[None]
    batch = adapter.batch(images, ["private question"], ["private answer"])
    update = replay(adapter, images, batch)
    for knowledge in spec["knowledge_conditions"]:
        observation = observation_for(adapter, batch, update, knowledge)
        guesses = text_guesses(spec, observation, knowledge)
        if "question" in KNOWLEDGE_FIELDS[knowledge]:
            assert all(guess["question"] == observation.public_questions[0] for guess in guesses)
        else:
            assert all("private question" not in guess["question"] for guess in guesses)
        if "target" in KNOWLEDGE_FIELDS[knowledge]:
            assert all(guess["target"] == observation.public_targets[0] for guess in guesses)
        else:
            assert all("private answer" not in guess["target"] for guess in guesses)


def test_tiny_score_resume_and_report_do_not_export_text(args):
    execute(args)
    args.stage = "score"
    execute(args)
    root = Path(args.output)
    records = list((root / "records/pilot/fc/image-000/scores").rglob("*.json"))
    # 6 images × (1 oracle + 2 guesses for each of three unknown-text conditions).
    assert len(records) == 42
    stamp = records[0].stat().st_mtime_ns
    args.resume = True
    execute(args)
    assert records[0].stat().st_mtime_ns == stamp
    args.stage = "report"
    execute(args)
    report_path = root / "reports/pilot/fc/report.json"
    report = read_json(report_path)
    assert set(report["summary"]) == {"text_known", "question_known", "target_known", "private"}
    assert report["summary"]["private"]["all"]["aggregate"]["mean_rank"][
        "spearman_mse"]["images"] == 2
    exported = json.dumps(report)
    assert "private question" not in exported and "private answer" not in exported
    assert (report_path.parent / "report.md").exists()
    assert (report_path.parent / "unknown_text_ranking.png").exists()


def test_gpu4_binding_is_exact(monkeypatch):
    inventory = {"4": {"uuid": "GPU-four", "free_mib": 80000, "total_mib": 81920}}
    monkeypatch.setenv("GIAVLM_TEXT_UNKNOWN_GPU", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-four")
    assert worker_binding("cuda:0", inventory=inventory)["physical_gpu"] == 4
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(ValueError, match="UUID"):
        worker_binding("cuda:0", inventory=inventory)
