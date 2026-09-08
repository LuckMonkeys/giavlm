from pathlib import Path
from types import SimpleNamespace

import pytest

from core.artifacts import read_json, write_json
from core.commands import build_parser
from core.data import synthetic
from utils.suite import execute, materialize


def command(*argv):
    args = build_parser().parse_args(list(argv))
    args.func(args)


def test_train_restore_capture_and_utility(tmp_path):
    manifest = synthetic(tmp_path / "data", 40, clients=2)
    train_output = tmp_path / "federation"
    overrides = ["--set", "training.clients=2", "--set", "training.rounds=2",
                 "--set", "training.snapshots=[0,1,2]", "--set", "training.mode=lora_llm"]
    command("train", "--data", str(manifest), "--output", str(train_output), *overrides)
    before = read_json(train_output / "training.json")
    command("train", "--data", str(manifest), "--output", str(train_output), "--resume", *overrides)
    assert before == read_json(train_output / "training.json")
    capture_output = tmp_path / "capture"
    command("capture", "--data", str(manifest), "--output", str(capture_output),
            "--model", str(train_output / "round-0002"), *overrides)
    command("attack", "--observation", str(capture_output / "public"),
            "--output", str(tmp_path / "attack"), "--set", "attack.iterations=2")
    command("evaluate", "--reconstruction", str(tmp_path / "attack"), "--truth", str(capture_output / "private"))
    assert read_json(tmp_path / "attack" / "evaluation.json")["status"] == "completed"
    utility_output = tmp_path / "utility.json"
    command("utility", "--model", str(train_output / "round-0002"), "--data", str(manifest),
            "--limit", "2", "--output", str(utility_output))
    assert 0 <= read_json(utility_output)["metrics"]["vqa_soft_accuracy"] <= 1


def test_suite_materialization_and_data_integrity(tmp_path):
    manifest = synthetic(tmp_path / "data", 20, clients=1)
    config = tmp_path / "tiny.json"
    write_json(config, {"training": {"clients": 1, "clients_per_round": 1},
                        "attack": {"iterations": 2}})
    args = SimpleNamespace(output=str(tmp_path / "suite"), data=str(manifest), configs=[str(config)],
                            set=[], tasks=["vqa"], modes=["full"], knowledge=["private", "text_known"],
                            samples=1, batch_size=1, local_steps=1, gpus_per_run=1, split="eval",
                            observation="gradient", methods=["ig_adapted", "random"], seeds=[0, 1],
                            pilot_seconds=1.0)
    budget = materialize(args)
    assert budget["jobs"] == 8 and budget["captures"] == 2
    jobs_file = Path(args.output) / "jobs.jsonl"
    run_args = SimpleNamespace(manifest=str(jobs_file), limit=1, threads=1)
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="Dataset manifest changed"):
        execute(run_args)


def test_unknown_baseline_and_missing_prior_are_explicit(tmp_path):
    manifest = synthetic(tmp_path / "data", 20, clients=1)
    capture_dir = tmp_path / "capture"
    command("capture", "--data", str(manifest), "--output", str(capture_dir))
    for method, status in [("dager", "not_implemented"), ("gradvit_adapted", "resource_unavailable")]:
        output = tmp_path / method
        command("attack", "--observation", str(capture_dir / "public"), "--output", str(output),
                "--set", f"attack.method={method}")
        assert read_json(output / "result.json")["status"] == status
        command("evaluate", "--reconstruction", str(output), "--truth", str(capture_dir / "private"))
        assert read_json(output / "evaluation.json")["samples"] == []
