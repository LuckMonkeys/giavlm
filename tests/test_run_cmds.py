from io import StringIO
import json
from pathlib import Path
import signal
import sys

import pytest

from utils import run_cmds


def write_schedule(path, text):
    path.write_text(text)
    return path


def test_schedule_keeps_benchmark_jobs_and_adds_structured_entries(tmp_path):
    schedule = write_schedule(tmp_path / "jobs.yaml", f"""
output_root: outputs
scheduler:
  min_free_mib: 12000
  max_jobs_per_gpu: 2
  poll_seconds: 5
jobs:
  - name: benchmark
    config_name: slake_llava_dlg_private
    overrides: [knowledge=private]
    resources:
      min_free_mib: 50000
  - name: module
    module: analyze.gradient_inspect
    args: [--config-name, inspect_llava]
  - name: script
    script: {Path(__file__).resolve()}
    args: [--help]
""")
    result = run_cmds.load_schedule(schedule)
    benchmark, module, script = result["jobs"]

    assert benchmark["argv"][1:5] == [
        "-m", "examples.run_attack", "--config-name", "slake_llava_dlg_private"]
    assert benchmark["argv"][5] == "knowledge=private"
    assert benchmark["min_free_mib"] == 50000
    assert module["argv"] == [sys.executable, "-m", "analyze.gradient_inspect",
                              "--config-name", "inspect_llava"]
    assert script["argv"] == [sys.executable, str(Path(__file__).resolve()), "--help"]
    assert module["min_free_mib"] == 12000
    assert result["scheduler"] == {
        "min_free_mib": 12000, "max_jobs_per_gpu": 2, "poll_seconds": 5.0,
        "occupancy": {"enabled": False, "script": None, "args": [],
                      "poll_seconds": 30.0}}


def test_multiple_schedules_merge_in_order_and_support_comma_paths(tmp_path, capsys):
    first = write_schedule(tmp_path / "first.yaml", "jobs:\n  - name: first\n")
    second = write_schedule(tmp_path / "second.yaml", "jobs:\n  - name: second\n")

    result = run_cmds.load_schedules([first, second])

    assert [job["name"] for job in result["jobs"]] == ["first", "second"]
    assert result["schedule_files"] == [str(first.resolve()), str(second.resolve())]
    assert run_cmds._schedule_paths([[f"{first},{second}"]]) == [str(first), str(second)]
    assert run_cmds.main([
        "--cmd-config-yaml", str(first), str(second)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert [job["name"] for job in printed["jobs"]] == ["first", "second"]


def test_multiple_schedules_reject_conflicting_names_roots_and_settings(tmp_path):
    first = write_schedule(tmp_path / "first.yaml", "jobs:\n  - name: shared\n")
    duplicate = write_schedule(tmp_path / "duplicate.yaml", "jobs:\n  - name: shared\n")
    with pytest.raises(ValueError, match="Duplicate job name"):
        run_cmds.load_schedules([first, duplicate])

    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_root = write_schedule(other_dir / "root.yaml", "jobs:\n  - name: other\n")
    with pytest.raises(ValueError, match="one output_root"):
        run_cmds.load_schedules([first, other_root])

    settings = write_schedule(tmp_path / "settings.yaml", """
scheduler:
  poll_seconds: 2
jobs:
  - name: settings
""")
    with pytest.raises(ValueError, match="same scheduler"):
        run_cmds.load_schedules([first, settings])


@pytest.mark.parametrize("job,match", [
    ("module: package.tool\n    script: tool.py", "exactly one"),
    ("module: package.tool\n    config_name: config", "Generic jobs support"),
    ("script: missing.py", "does not exist"),
    ("module: invalid-name", "dotted Python module"),
    ("config_name: config\n    overrides: [1]", "list of strings"),
])
def test_schedule_rejects_ambiguous_or_malformed_jobs(tmp_path, job, match):
    path = write_schedule(tmp_path / "jobs.yaml", f"""
jobs:
  - name: invalid
    {job}
""")
    with pytest.raises(ValueError, match=match):
        run_cmds.load_schedule(path)


def test_cli_scheduler_values_override_yaml(tmp_path):
    path = write_schedule(tmp_path / "jobs.yaml", """
scheduler:
  min_free_mib: 1
  max_jobs_per_gpu: 1
  poll_seconds: 1
jobs:
  - name: benchmark
""")
    result = run_cmds.load_schedule(path, scheduler_overrides={
        "min_free_mib": 2, "max_jobs_per_gpu": 3, "poll_seconds": 4})
    assert result["scheduler"] == {
        "min_free_mib": 2, "max_jobs_per_gpu": 3, "poll_seconds": 4.0,
        "occupancy": {"enabled": False, "script": None, "args": [],
                      "poll_seconds": 30.0}}
    assert result["jobs"][0]["min_free_mib"] == 2


def test_occupancy_requires_a_real_python_script(tmp_path):
    missing = write_schedule(tmp_path / "missing.yaml", """
scheduler:
  occupancy:
    enabled: true
jobs:
  - name: benchmark
""")
    with pytest.raises(ValueError, match="script is required"):
        run_cmds.load_schedule(missing)

    invalid = write_schedule(tmp_path / "invalid.yaml", """
scheduler:
  occupancy:
    enabled: true
    script: missing.py
jobs:
  - name: benchmark
""")
    with pytest.raises(ValueError, match="does not exist"):
        run_cmds.load_schedule(invalid)


def test_gpu_query_uses_nvidia_smi_csv(monkeypatch):
    monkeypatch.setattr(run_cmds.subprocess, "check_output",
                        lambda *args, **kwargs: "0, 12000\n2, 34000\n")
    assert run_cmds.query_gpu_memory(["2", "0"]) == {"2": 34000, "0": 12000}
    with pytest.raises(ValueError, match="Unknown GPU"):
        run_cmds.query_gpu_memory(["1"])


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeProcess:
    next_pid = 1000

    def __init__(self, code=0, polls=0):
        self.code = code
        self.remaining = polls
        self.done = False
        self.stdout = StringIO("output\n")
        self.stderr = StringIO("error\n" if code else "")
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1

    def poll(self):
        if self.done:
            return self.code
        if self.remaining:
            self.remaining -= 1
            return None
        self.done = True
        return self.code

    def wait(self):
        self.done = True
        return self.code


class FakeFactory:
    def __init__(self, clock, definitions=None):
        self.clock = clock
        self.definitions = definitions or {}
        self.started = []
        self.processes = {}

    def __call__(self, argv, **kwargs):
        name = argv[-1]
        code, polls = self.definitions.get(name, (0, 0))
        process = FakeProcess(code, polls)
        self.processes[process.pid] = process
        self.started.append({"name": name, "gpu": kwargs["env"].get("CUDA_VISIBLE_DEVICES"),
                             "time": self.clock(), "shell": kwargs.get("shell")})
        return process


def fake_schedule(tmp_path, names, *, maximum=1, minimum=None, poll=1):
    minimum = minimum or {}
    return {
        "output_root": tmp_path,
        "scheduler": {"min_free_mib": 10, "max_jobs_per_gpu": maximum,
                      "poll_seconds": float(poll),
                      "occupancy": {"enabled": False, "script": None, "args": [],
                                    "poll_seconds": 1.0}},
        "jobs": [{"name": name, "kind": "module", "argv": [sys.executable, name],
                  "min_free_mib": minimum.get(name, 10), "output": str(tmp_path / name)}
                 for name in names],
    }


def test_multi_gpu_allows_multiple_jobs_per_card_and_never_uses_shell(tmp_path):
    clock = FakeClock()
    schedule = fake_schedule(tmp_path, ["a", "b", "c", "d"], maximum=2, poll=1)
    factory = FakeFactory(clock, {name: (0, 8) for name in ["a", "b", "c", "d"]})
    runner = run_cmds.JobRunner(schedule, popen=factory, clock=clock,
                                sleep=clock.sleep, gpu_query=lambda ids: {gpu: 100 for gpu in ids})

    assert run_cmds.run_gpu(schedule, ["0", "1"], runner) == 0
    assert [(row["name"], row["gpu"], row["time"]) for row in factory.started] == [
        ("a", "0", 0.0), ("b", "1", 0.0), ("c", "0", 1.0), ("d", "1", 1.0)]
    assert all(row["shell"] is None for row in factory.started)
    summary = json.loads((tmp_path / "scheduler-results.json").read_text())
    assert all(job["status"] == "completed" for job in summary["jobs"])


def test_gpu_queue_skips_large_job_until_memory_is_available(tmp_path):
    clock = FakeClock()
    schedule = fake_schedule(tmp_path, ["large", "small"],
                             minimum={"large": 90, "small": 10})
    factory = FakeFactory(clock)
    samples = iter([20, 100])
    runner = run_cmds.JobRunner(
        schedule, popen=factory, clock=clock, sleep=clock.sleep,
        gpu_query=lambda ids: {"0": next(samples)})

    assert run_cmds.run_gpu(schedule, ["0"], runner) == 0
    assert [row["name"] for row in factory.started] == ["small", "large"]


def test_failures_are_recorded_without_stopping_remaining_jobs(tmp_path):
    clock = FakeClock()
    schedule = fake_schedule(tmp_path, ["first", "fail", "last"])
    factory = FakeFactory(clock, {"fail": (7, 0)})
    runner = run_cmds.JobRunner(schedule, popen=factory, clock=clock, sleep=clock.sleep)

    assert run_cmds.run_serial(schedule, runner) == 1
    assert [row["name"] for row in factory.started] == ["first", "fail", "last"]
    statuses = {row["name"]: (row["status"], row["returncode"])
                for row in runner.results}
    assert statuses == {"first": ("completed", 0), "fail": ("failed", 7),
                        "last": ("completed", 0)}


def test_occupancy_starts_after_jobs_with_isolated_gpu_and_structured_args(tmp_path):
    clock = FakeClock()
    schedule = fake_schedule(tmp_path, [])
    schedule["scheduler"]["occupancy"] = {
        "enabled": True,
        "script": str(Path(__file__).resolve()),
        "args": ["--local", "{gpu}", "--physical", "{physical_gpu}"],
        "poll_seconds": 1.0,
    }
    factory = FakeFactory(clock)
    runner = run_cmds.JobRunner(schedule, popen=factory, clock=clock, sleep=clock.sleep)
    runner.root.mkdir(parents=True, exist_ok=True)
    supervisor = run_cmds.OccupancySupervisor(runner)

    assert supervisor.start(["2", "4"]) == 2
    assert [row["gpu"] for row in factory.started] == ["2", "4"]
    assert all(row["shell"] is None for row in factory.started)
    assert runner.occupancy[0]["argv"][-4:] == ["--local", "0", "--physical", "2"]
    assert runner.occupancy[1]["argv"][-4:] == ["--local", "0", "--physical", "4"]
    assert supervisor.monitor() == 0
    assert all(row["status"] == "completed" for row in runner.occupancy)


def test_occupancy_termination_kills_process_group(tmp_path, monkeypatch):
    clock = FakeClock()
    schedule = fake_schedule(tmp_path, [])
    schedule["scheduler"]["occupancy"] = {
        "enabled": True,
        "script": str(Path(__file__).resolve()),
        "args": [],
        "poll_seconds": 1.0,
    }
    factory = FakeFactory(clock, {str(Path(__file__).resolve()): (0, 1000)})
    runner = run_cmds.JobRunner(schedule, popen=factory, clock=clock, sleep=clock.sleep)
    supervisor = run_cmds.OccupancySupervisor(runner)
    supervisor.start(["1"])
    calls = []

    def kill_group(pid, sent_signal):
        calls.append((pid, sent_signal))
        process = factory.processes[pid]
        process.code = -sent_signal
        process.done = True

    monkeypatch.setattr(run_cmds.os, "killpg", kill_group)
    supervisor.terminate()

    assert calls == [(next(iter(factory.processes)), signal.SIGTERM)]
    assert runner.occupancy[0]["status"] == "cancelled"


def test_terminate_kills_process_group_and_marks_job_cancelled(tmp_path, monkeypatch):
    clock = FakeClock()
    schedule = fake_schedule(tmp_path, ["long"])
    factory = FakeFactory(clock, {"long": (0, 1000)})
    runner = run_cmds.JobRunner(schedule, popen=factory, clock=clock, sleep=clock.sleep)
    running = runner.start(schedule["jobs"][0], "3")
    calls = []

    def kill_group(pid, sent_signal):
        calls.append((pid, sent_signal))
        process = factory.processes[pid]
        process.code = -sent_signal
        process.done = True

    monkeypatch.setattr(run_cmds.os, "killpg", kill_group)
    runner.terminate()

    assert calls == [(running["process"].pid, signal.SIGTERM)]
    assert runner.results[0]["status"] == "cancelled"


def test_dry_run_does_not_query_gpu_or_start_process(tmp_path, monkeypatch, capsys):
    path = write_schedule(tmp_path / "jobs.yaml", "jobs:\n  - name: benchmark\n")
    monkeypatch.setattr(run_cmds, "query_gpu_memory",
                        lambda ids: pytest.fail("dry-run queried GPUs"))
    monkeypatch.setattr(run_cmds.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("dry-run started a process"))

    assert run_cmds.main(["--cmd-config-yaml", str(path), "--gpu-ids", "0,1"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["jobs"][0]["argv"][1:3] == ["-m", "examples.run_attack"]
