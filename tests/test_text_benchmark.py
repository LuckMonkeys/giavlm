from dataclasses import replace

import pytest

from core.artifacts import write_json
from core.config import TrainingSpec, validate_training_config
from evaluation.text_benchmark import aggregate, attack_spec, capture_training, evaluate_committed
from evaluation.text_benchmark_data import select_evaluation, training_schedule


def test_capture_protocol_is_separate_from_training_budget():
    trained = TrainingSpec(algorithm="fedavg", local_optimizer="adamw", batch_size=2,
                           local_steps=10, gradient_accumulation_steps=4,
                           server_round=10, lora_rank=32, lora_alpha=64)
    captured = capture_training(trained)
    validate_training_config(captured)
    assert captured.sample_count == 1 and trained.sample_count == 80
    assert captured.server_round == 10 and captured.lora_rank == 32
    assert captured.knowledge == "image_known" and captured.token_lengths_known
    for family, projections in (("llava", "qkv"), ("qwen3_5", "linear_inputs")):
        assert attack_spec(family, "gt").dager.projections == projections
        assert attack_spec(family, "random").dager.image_source == "random"


def test_frozen_sampling_is_order_independent_and_shortages_are_not_filled():
    rows = [{"sample_id": str(i), "image_id": str(i // 2), "question": f"q{i}",
             "target": "a", "split": "eval", "client": 0} for i in range(12)]
    lengths = {r["sample_id"]: [5 if i < 4 else 20 if i < 10 else 40, 1]
               for i, r in enumerate(rows)}
    chosen, bins = select_evaluation(rows, lengths)
    assert select_evaluation(list(reversed(rows)), lengths) == (chosen, bins)
    assert [r["selected_n"] for r in bins] == [2, 3, 1]
    assert len({rows[int(r["sample_id"])]["image_id"] for r in chosen}) == len(chosen)
    training_rows = [{**row, "split": "train"} for row in rows]
    training = replace(TrainingSpec(), batch_size=2, clients_per_round=1, rounds=3)
    assert training_schedule(training_rows, training) == training_schedule(training_rows[::-1], training)


def test_evaluation_checks_all_commits_before_references(tmp_path, monkeypatch):
    frozen = {"selected": [{"length_bin": "1-16"}]}
    write_json(tmp_path / "run-00000/attacks/gt/result.json", {"status": "completed"})
    def forbidden(*args, **kwargs):
        raise AssertionError("References were opened before the random reconstruction committed")
    monkeypatch.setattr("evaluation.text_benchmark.evaluate", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_committed(tmp_path, frozen, "llava")


def test_aggregate_keeps_budget_failures_in_denominator_and_reports_missing_metrics():
    common = {"model": "llava", "dataset": "slake", "image": "random", "length_bin": "1-16"}
    rows = [{**common, "status": "completed", "joint_em": 1., "average_precision": .8},
            {**common, "status": "budget_exhausted", "joint_em": 0.}]
    summaries = aggregate(rows)
    assert len(summaries) == 2
    for summary in summaries:
        assert summary["n"] == 2 and summary["completed"] == 1
        assert summary["joint_em"] == .5 and summary["joint_em_n"] == 2
        assert summary["average_precision"] == .8 and summary["average_precision_n"] == 1
