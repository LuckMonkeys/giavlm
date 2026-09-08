from PIL import Image
import pytest
import torch

from core.data import (MEDICAL_VQA_SOURCES, assignment, prepare_medical_vqa,
                       read_manifest, synthetic)
from evaluation.reconstruction import match_pairs, summarize
from metrics.image import image_metrics
from metrics.text import text_metrics


def test_cross_task_images_never_cross_partitions(tmp_path):
    manifest = synthetic(tmp_path, count=20)
    rows = read_manifest(manifest)
    for row in rows:
        assert (row["split"], row["client"]) == assignment(row["image_id"], 42, 10)
    assert len(rows) == 40
    with pytest.raises(FileExistsError):
        synthetic(tmp_path)


def test_metrics_identity_and_repeated_words():
    image = torch.rand(3, 8, 8)
    score = image_metrics(image, image)
    assert score["ssim"] == 1 and score["psnr_infinite"] and score["mse"] == 0
    assert text_metrics("red red blue", "red blue")["word_recall"] == pytest.approx(2 / 3)
    assert text_metrics("red red blue", "red blue")["wer"] == pytest.approx(1 / 3)
    assert text_metrics("red red blue", "red red blue")["rougeL"] == 1


def test_batch_matching_preserves_whole_pairs():
    images = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8)])
    assignment, agreement = match_pairs(images, images.flip(0), ["red", "blue"], ["blue", "red"])
    assert assignment == [1, 0] and agreement == 1
    _, agreement = match_pairs(images, images.flip(0), ["red", "blue"], ["red", "blue"])
    assert agreement == 0


def test_bootstrap_groups_images_and_preserves_failure_counts():
    report = {"condition": {"model": "tiny", "seed": 0}, "status": "completed", "costs": {},
              "samples": [{"image_id": "a", "metrics": {"ssim": 0.2}},
                          {"image_id": "a", "metrics": {"ssim": 0.4}},
                          {"image_id": "b", "metrics": {"ssim": 0.8}}]}
    failed = {**report, "status": "not_applicable", "samples": []}
    summary = summarize([report, failed], bootstrap=20)["groups"][0]
    assert summary["metrics"]["ssim"]["image_groups"] == 2
    assert summary["metrics"]["ssim"]["mean"] == pytest.approx(0.55)
    assert summary["statuses"] == {"completed": 1, "not_applicable": 1}


class _FakeRows:
    """Stands in for a datasets.Dataset split without touching the network."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, index):
        return self._rows[index]


def test_medical_vqa_groups_questions_by_image_content(tmp_path, monkeypatch):
    red = Image.new("RGB", (8, 8), (200, 10, 10))
    blue = Image.new("RGB", (8, 8), (10, 10, 200))
    # The same picture appears twice, and once more in another upstream split.
    splits = {
        "train": _FakeRows([{"image": red, "question": "q1", "answer": " yes "},
                            {"image": red, "question": "q2", "answer": "no"},
                            {"image": blue, "question": "q3", "answer": "maybe"}]),
        "test": _FakeRows([{"image": red, "question": "q4", "answer": "yes"}]),
    }
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: splits)

    manifest = prepare_medical_vqa(tmp_path / "med", sources=("vqa_rad",), seed=42, clients=4)
    rows = read_manifest(manifest)  # raises if an image group crosses a split or client

    assert len(rows) == 4
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_id"], []).append(row)
    assert len(by_image) == 2, "identical pixels must collapse to one canonical image id"
    shared = max(by_image.values(), key=len)
    assert len(shared) == 3, "rows sharing an image stay in one group across upstream splits"
    assert len({(r["split"], r["client"]) for r in shared}) == 1
    assert len(list((tmp_path / "med" / "images").iterdir())) == 2, "each image written once"

    answers = {r["sample_id"]: r["target"] for r in rows}
    assert answers["vqa_rad:train:0"] == "yes", "answers are stripped"
    assert all(r["references"] == [r["target"]] for r in rows)
    assert {r["source_split"] for r in rows} == {"train", "test"}
    with pytest.raises(FileExistsError):
        prepare_medical_vqa(tmp_path / "med", sources=("vqa_rad",))


def test_medical_vqa_rejects_unknown_source(tmp_path):
    assert set(MEDICAL_VQA_SOURCES) == {"vqa_rad", "slake"}
    with pytest.raises(ValueError, match="Unknown medical VQA source"):
        prepare_medical_vqa(tmp_path / "med", sources=("imagenet",))
