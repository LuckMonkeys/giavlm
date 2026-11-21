from collections import Counter, defaultdict

from evaluation.public_style_generation import generation_cases
from evaluation.public_style_study import controlled_training
from core.config import TrainingSpec


def test_styles_and_queries_are_balanced_and_group_disjoint():
    rows = generation_cases()
    counts = Counter((r["split"], r["style"], r["label"]) for r in rows)
    assert len(rows) == 108
    for split in ["fit", "dev"]:
        for style in ["photo", "line", "render"]:
            for label in ["head", "non-head"]:
                assert counts[split, style, label] == 8
    assert counts["query", "render", "head"] == counts["query", "render", "non-head"] == 6
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    for group in groups.values():
        assert len({r["split"] for r in group}) == 1
        if group[0]["split"] != "query":
            assert {r["style"] for r in group} == {"photo", "line", "render"}
            assert len({r["seed"] for r in group}) == 1


def test_controlled_capture_does_not_inherit_the_snapshot_optimizer():
    snapshot = TrainingSpec(algorithm="fedavg", local_optimizer="adamw", local_steps=10,
                            batch_size=2, gradient_accumulation_steps=4, weight_decay=.01)
    capture = controlled_training(snapshot)
    assert capture.algorithm == "fedsgd" and capture.local_optimizer == "sgd"
    assert capture.sample_count == 1 and capture.weight_decay == 0
    assert capture.knowledge == "text_known" and not capture.image_public
    assert snapshot.local_optimizer == "adamw" and snapshot.sample_count == 80
