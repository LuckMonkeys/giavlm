import pytest
import torch

from core.data import assignment, read_manifest, synthetic
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
