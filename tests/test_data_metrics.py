from PIL import Image
import pytest
import torch

from core.data import (MEDICAL_VQA_SOURCES, assignment, inject_canaries,
                       prepare_medical_vqa, read_manifest, synthetic)
from evaluation.reconstruction import canary_metrics, match_pairs, summarize
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


def test_canary_injection_preserves_partitions_and_marks_entities(tmp_path):
    source = synthetic(tmp_path / "base", count=12)
    before = {r["sample_id"]: (r["split"], r["client"]) for r in read_manifest(source)}

    # A different canary seed must not disturb the inherited partitioning.
    injected = inject_canaries(tmp_path / "base" / "samples.jsonl", tmp_path / "canary",
                               canary_seed=7, rate=1.0, field="question")
    rows = read_manifest(injected)

    assert {r["sample_id"]: (r["split"], r["client"]) for r in rows} == before, \
        "injection must not move a sample between splits or clients"
    for row in rows:
        assert len(row["canary"]) == 3
        assert all(value in row["question"] for value in row["canary"])
    assert len({tuple(r["canary"]) for r in rows}) > 1, "entities vary across samples"
    with pytest.raises(FileExistsError):
        inject_canaries(source, tmp_path / "canary")


def test_canary_injection_rate_and_field(tmp_path):
    source = synthetic(tmp_path / "base", count=40)
    partial = inject_canaries(source, tmp_path / "some", canary_seed=3, rate=0.5, field="target")
    rows = read_manifest(partial)
    marked = [r for r in rows if r["canary"]]
    assert 0 < len(marked) < len(rows), "a rate below one leaves some rows clean"
    for row in marked:
        assert all(value in row["target"] for value in row["canary"])
        assert row["references"] == [row["target"]]
    for bad in [{"field": "image"}, {"rate": 0.0}, {"rate": 1.5}]:
        with pytest.raises(ValueError):
            inject_canaries(source, tmp_path / f"bad-{bad}", **bad)
    (tmp_path / "base" / "samples.meta.json").unlink()
    with pytest.raises(FileNotFoundError, match="inherits its partitioning"):
        inject_canaries(source, tmp_path / "orphan")


def test_canary_recall_scores_only_trained_entities():
    row = {"canary": ["Ellary Northcott", "MRN5548093", "1968-03-16"]}
    # All three reached the trained tokens; the attack recovered two.
    full = canary_metrics(row, "patient Ellary Northcott MRN5548093 dob 1968-03-16 what",
                          "Ellary Northcott MRN5548093 dob unknown")
    assert full == {"canary_declared": 3, "canary_trained": 3, "canary_recall": 2 / 3}

    # Truncation dropped two; recall is over the survivor only, not over three.
    clipped = canary_metrics(row, "patient Ellary Northcott", "Ellary Northcott")
    assert clipped == {"canary_declared": 3, "canary_trained": 1, "canary_recall": 1.0}

    # Nothing survived tokenization, so recall is undefined rather than zero.
    none_trained = canary_metrics(row, "what", "Ellary Northcott")
    assert none_trained["canary_trained"] == 0 and none_trained["canary_recall"] is None

    assert canary_metrics({"canary": []}, "a", "b") == {}
    assert canary_metrics({}, "a", "b") == {}
