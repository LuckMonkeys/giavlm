"""Structured multi-YAML runner with memory-aware GPUs and opt-in occupancy."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIN_FREE_MIB = 10_000
DEFAULT_MAX_JOBS_PER_GPU = 1
DEFAULT_POLL_SECONDS = 30.0
DEFAULT_OCCUPANCY_POLL_SECONDS = 30.0
SUMMARY_SCHEMA = 2


def _positive_number(name, value, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or value <= 0:
        kind = "integer" if integer else "number"
        raise ValueError(f"{name} must be a positive {kind}")
    return int(value) if integer else float(value)


def _string_list(name, values):
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{name} must be a list of strings")
    return values


def _resolve_root(schedule_path, configured, override):
    root = Path(override or configured or "../outputs/batch")
    return root.resolve() if root.is_absolute() else (schedule_path.parent / root).resolve()


def _python_script(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty Python script path")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise ValueError(f"Python script does not exist: {path}")
    return path


def _occupancy_config(value, cli=None):
    value = value or {}
    if not isinstance(value, dict) or set(value) - {
            "enabled", "script", "args", "poll_seconds"}:
        raise ValueError(
            "scheduler.occupancy supports enabled, script, args, and poll_seconds")
    cli = cli or {}
    enabled = cli.get("occupancy_enabled")
    enabled = value.get("enabled", False) if enabled is None else enabled
    if not isinstance(enabled, bool):
        raise ValueError("scheduler.occupancy.enabled must be a boolean")
    script = cli.get("occupancy_script") or value.get("script")
    if enabled and not script:
        raise ValueError("scheduler.occupancy.script is required when occupancy is enabled")
    path = _python_script(script, "scheduler.occupancy.script") if script else None
    args = _string_list("scheduler.occupancy.args", value.get("args", []))
    poll_seconds = _positive_number(
        "scheduler.occupancy.poll_seconds",
        value.get("poll_seconds", DEFAULT_OCCUPANCY_POLL_SECONDS))
    return {"enabled": enabled, "script": str(path) if path else None,
            "args": args, "poll_seconds": poll_seconds}


def _scheduler_config(value, cli=None):
    value = value or {}
    if not isinstance(value, dict) or set(value) - {
            "min_free_mib", "max_jobs_per_gpu", "poll_seconds", "occupancy"}:
        raise ValueError(
            "scheduler supports min_free_mib, max_jobs_per_gpu, poll_seconds, and occupancy")
    cli = cli or {}
    minimum = cli.get("min_free_mib")
    maximum = cli.get("max_jobs_per_gpu")
    interval = cli.get("poll_seconds")
    return {
        "min_free_mib": _positive_number(
            "scheduler.min_free_mib",
            minimum if minimum is not None else value.get("min_free_mib", DEFAULT_MIN_FREE_MIB),
            integer=True),
        "max_jobs_per_gpu": _positive_number(
            "scheduler.max_jobs_per_gpu",
            maximum if maximum is not None else value.get(
                "max_jobs_per_gpu", DEFAULT_MAX_JOBS_PER_GPU), integer=True),
        "poll_seconds": _positive_number(
            "scheduler.poll_seconds",
            interval if interval is not None else value.get("poll_seconds", DEFAULT_POLL_SECONDS)),
        "occupancy": _occupancy_config(value.get("occupancy"), cli),
    }


def _resources(value, default_minimum):
    value = value or {}
    if not isinstance(value, dict) or set(value) - {"min_free_mib"}:
        raise ValueError("resources supports only min_free_mib")
    return {"min_free_mib": _positive_number(
        "resources.min_free_mib", value.get("min_free_mib", default_minimum), integer=True)}


def _generic_argv(job):
    module, script = job.get("module"), job.get("script")
    if bool(module) == bool(script):
        raise ValueError("A generic job requires exactly one of module or script")
    args = _string_list("args", job.get("args", []))
    if module:
        if not isinstance(module, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            raise ValueError("module must be a dotted Python module name")
        return "module", [sys.executable, "-m", module, *args]
    path = _python_script(script, "script")
    return "script", [sys.executable, str(path), *args]


def load_schedule(path, output_root=None, scheduler_overrides=None):
    """Validate and expand a schedule without creating files or starting processes."""
    path = Path(path).resolve()
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict) or set(config) - {"output_root", "scheduler", "jobs"}:
        raise ValueError("Schedule supports only output_root, scheduler, and jobs")
    if not isinstance(config.get("jobs"), list) or not config["jobs"]:
        raise ValueError("Schedule requires a nonempty jobs list")

    scheduler = _scheduler_config(config.get("scheduler"), scheduler_overrides)
    root = _resolve_root(path, config.get("output_root"), output_root)
    jobs, names = [], set()
    for raw in config["jobs"]:
        if not isinstance(raw, dict):
            raise ValueError("Each job must be a mapping")
        name = raw.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in names:
            raise ValueError("Job names must be unique, path-safe identifiers")
        names.add(name)
        resources = _resources(raw.get("resources"), scheduler["min_free_mib"])
        generic = "module" in raw or "script" in raw or "args" in raw
        if generic:
            if set(raw) - {"name", "module", "script", "args", "resources"}:
                raise ValueError("Generic jobs support name, module/script, args, and resources")
            kind, argv = _generic_argv(raw)
        else:
            if set(raw) - {"name", "config_name", "overrides", "resources"}:
                raise ValueError(
                    "Benchmark jobs support name, config_name, overrides, and resources")
            overrides = _string_list("overrides", raw.get("overrides", []))
            config_name = raw.get("config_name", "config")
            if not isinstance(config_name, str) or not config_name:
                raise ValueError("config_name must be a nonempty string")
            if any(value.startswith(("output_dir=", "hydra.run.dir=", "--"))
                   for value in overrides):
                raise ValueError("Output routing and CLI options belong to the scheduler")
            output = (root / name).resolve()
            argv = [sys.executable, "-m", "examples.run_attack", "--config-name",
                    config_name, *overrides,
                    f"output_dir='{output}'", f"hydra.run.dir='{output / 'hydra'}'"]
            kind = "benchmark"
        jobs.append({"name": name, "kind": kind, "argv": argv,
                     "min_free_mib": resources["min_free_mib"],
                     "output": str((root / name).resolve()),
                     "schedule": str(path)})
    return {"output_root": root, "scheduler": scheduler, "jobs": jobs}


def load_schedules(paths, output_root=None, scheduler_overrides=None):
    """Merge schedules in declaration order after validating shared settings."""
    if not paths:
        raise ValueError("At least one schedule YAML is required")
    schedules = [load_schedule(path, output_root, scheduler_overrides) for path in paths]
    first = schedules[0]
    jobs, names = [], set()
    for schedule in schedules:
        if schedule["output_root"] != first["output_root"]:
            raise ValueError(
                "Multiple schedules must resolve to one output_root; use --output-root")
        if schedule["scheduler"] != first["scheduler"]:
            raise ValueError("Multiple schedules must use the same scheduler configuration")
        for job in schedule["jobs"]:
            if job["name"] in names:
                raise ValueError(f"Duplicate job name across schedules: {job['name']}")
            names.add(job["name"])
            jobs.append(job)
    return {"output_root": first["output_root"], "scheduler": first["scheduler"],
            "jobs": jobs, "schedule_files": [str(Path(path).resolve()) for path in paths]}


def materialize_jobs(path, output_root=None):
    """Backward-compatible job expansion used by tests and external tooling."""
    return load_schedule(path, output_root)["jobs"]


def query_gpu_memory(gpu_ids):
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    discovered = {}
    for line in output.strip().splitlines():
        index, free = line.split(",", 1)
        discovered[index.strip()] = int(free.strip())
    missing = set(gpu_ids) - set(discovered)
    if missing:
        raise ValueError(f"Unknown GPU IDs: {sorted(missing)}")
    return {gpu: discovered[gpu] for gpu in gpu_ids}


def available_gpu(gpu_ids, min_free_mib):
    """Return the first eligible GPU; retained for callers of the old helper."""
    for gpu, free in query_gpu_memory(gpu_ids).items():
        if free >= min_free_mib:
            return gpu
    raise RuntimeError("No requested GPU meets the memory threshold")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class JobRunner:
    """Own child processes, logs, and a continuously committed scheduler summary."""

    def __init__(self, schedule, popen=subprocess.Popen, clock=time.monotonic,
                 sleep=time.sleep, gpu_query=query_gpu_memory):
        self.schedule = schedule
        self.popen = popen
        self.clock = clock
        self.sleep = sleep
        self.gpu_query = gpu_query
        self.root = Path(schedule["output_root"])
        self.logs = self.root / "_scheduler_logs"
        self.summary_path = self.root / "scheduler-results.json"
        self.running = []
        self.cancelling = set()
        self.print_lock = threading.Lock()
        self.started_at = _timestamp()
        self.occupancy = []
        self.results = [{"name": job["name"], "kind": job["kind"],
                         "argv": job["argv"], "min_free_mib": job["min_free_mib"],
                         "gpu": None, "status": "pending", "returncode": None,
                         "started_at": None, "ended_at": None, "seconds": None,
                         "stdout": f"_scheduler_logs/{job['name']}.stdout.log",
                         "stderr": f"_scheduler_logs/{job['name']}.stderr.log"}
                        for job in schedule["jobs"]]
        self.by_name = {result["name"]: result for result in self.results}

    def _commit(self):
        _write_json(self.summary_path, {
            "schema_version": SUMMARY_SCHEMA,
            "started_at": self.started_at,
            "updated_at": _timestamp(),
            "scheduler": self.schedule["scheduler"],
            "jobs": self.results,
            "occupancy": self.occupancy,
        })

    def _relay(self, stream, log, name, channel):
        for line in iter(stream.readline, ""):
            log.write(line)
            log.flush()
            with self.print_lock:
                print(f"[{name}][{channel}] {line}", end="", flush=True)
        stream.close()

    def start(self, job, gpu=None):
        self.logs.mkdir(parents=True, exist_ok=True)
        stdout_log = (self.logs / f"{job['name']}.stdout.log").open("w")
        stderr_log = (self.logs / f"{job['name']}.stderr.log").open("w")
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        try:
            process = self.popen(
                job["argv"], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
        except Exception:
            stdout_log.close()
            stderr_log.close()
            result = self.by_name[job["name"]]
            result.update(status="failed_to_start", gpu=gpu, returncode=-1,
                          started_at=_timestamp(), ended_at=_timestamp(), seconds=0.0)
            self._commit()
            return None
        now, wall = self.clock(), _timestamp()
        result = self.by_name[job["name"]]
        result.update(status="running", gpu=gpu, started_at=wall)
        threads = [
            threading.Thread(target=self._relay,
                             args=(process.stdout, stdout_log, job["name"], "stdout"),
                             daemon=True),
            threading.Thread(target=self._relay,
                             args=(process.stderr, stderr_log, job["name"], "stderr"),
                             daemon=True),
        ]
        for thread in threads:
            thread.start()
        running = {"job": job, "gpu": gpu, "process": process, "start": now,
                   "threads": threads, "logs": (stdout_log, stderr_log)}
        self.running.append(running)
        self._commit()
        return running

    def reap(self):
        completed = []
        for running in list(self.running):
            code = running["process"].poll()
            if code is None:
                continue
            for thread in running["threads"]:
                thread.join()
            for log in running["logs"]:
                log.close()
            result = self.by_name[running["job"]["name"]]
            name = running["job"]["name"]
            status = "cancelled" if name in self.cancelling else (
                "completed" if code == 0 else "failed")
            result.update(status=status, returncode=code,
                          ended_at=_timestamp(), seconds=self.clock() - running["start"])
            self.running.remove(running)
            completed.append(running)
        if completed:
            self._commit()
        return completed

    def terminate(self, grace_seconds=5.0):
        self.cancelling.update(item["job"]["name"] for item in self.running)
        for running in self.running:
            process = running["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = self.clock() + grace_seconds
        while self.running and self.clock() < deadline:
            self.reap()
            if self.running:
                self.sleep(0.05)
        for running in self.running:
            process = running["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
        self.reap()
        for result in self.results:
            if result["status"] in {"pending", "running"}:
                result.update(status="cancelled", ended_at=_timestamp())
        self._commit()

    def exit_code(self):
        return int(any(result["status"] != "completed" for result in self.results))


class OccupancySupervisor:
    """Run one opt-in occupancy process per GPU after all real jobs finish."""

    def __init__(self, runner, popen=None):
        self.runner = runner
        self.popen = popen or runner.popen
        self.running = []

    def _argv(self, gpu):
        config = self.runner.schedule["scheduler"]["occupancy"]
        replacements = {"{gpu}": "0", "{physical_gpu}": str(gpu)}
        args = [replacements.get(value, value) for value in config["args"]]
        return [sys.executable, config["script"], *args]

    def start(self, gpu_ids):
        for gpu in gpu_ids:
            name = f"occupancy-gpu-{gpu}"
            stdout_path = self.runner.logs / f"{name}.stdout.log"
            stderr_path = self.runner.logs / f"{name}.stderr.log"
            self.runner.logs.mkdir(parents=True, exist_ok=True)
            stdout_log = stdout_path.open("w")
            stderr_log = stderr_path.open("w")
            argv = self._argv(gpu)
            row = {"gpu": gpu, "argv": argv, "status": "pending", "returncode": None,
                   "started_at": None, "ended_at": None,
                   "stdout": str(stdout_path.relative_to(self.runner.root)),
                   "stderr": str(stderr_path.relative_to(self.runner.root))}
            self.runner.occupancy.append(row)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            try:
                process = self.popen(
                    argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
            except Exception:
                stdout_log.close()
                stderr_log.close()
                row.update(status="failed_to_start", returncode=-1,
                           started_at=_timestamp(), ended_at=_timestamp())
                self.runner._commit()
                continue
            row.update(status="running", started_at=_timestamp())
            threads = [
                threading.Thread(
                    target=self.runner._relay,
                    args=(process.stdout, stdout_log, name, "stdout"), daemon=True),
                threading.Thread(
                    target=self.runner._relay,
                    args=(process.stderr, stderr_log, name, "stderr"), daemon=True),
            ]
            for thread in threads:
                thread.start()
            self.running.append({"gpu": gpu, "process": process, "row": row,
                                 "threads": threads, "logs": (stdout_log, stderr_log)})
            self.runner._commit()
            print(f"[scheduler] GPU {gpu} occupancy started (PID {process.pid})", flush=True)
        return len(self.running)

    def reap(self):
        for item in list(self.running):
            code = item["process"].poll()
            if code is None:
                continue
            self._finish(item, "completed" if code == 0 else "failed", code)

    def _finish(self, item, status, code):
        for thread in item["threads"]:
            thread.join()
        for log in item["logs"]:
            log.close()
        item["row"].update(status=status, returncode=code, ended_at=_timestamp())
        self.running.remove(item)
        self.runner._commit()

    def monitor(self):
        interval = self.runner.schedule["scheduler"]["occupancy"]["poll_seconds"]
        while self.running:
            self.reap()
            if self.running:
                self.runner.sleep(interval)
        return int(any(row["status"] != "completed" for row in self.runner.occupancy))

    def terminate(self, grace_seconds=5.0):
        for item in self.running:
            if item["process"].poll() is None:
                try:
                    os.killpg(item["process"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = self.runner.clock() + grace_seconds
        while self.running and self.runner.clock() < deadline:
            for item in list(self.running):
                code = item["process"].poll()
                if code is not None:
                    self._finish(item, "cancelled", code)
            if self.running:
                self.runner.sleep(0.05)
        for item in list(self.running):
            process = item["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = process.wait()
            self._finish(item, "cancelled", code)


def run_serial(schedule, runner=None):
    runner = runner or JobRunner(schedule)
    runner.root.mkdir(parents=True, exist_ok=True)
    runner._commit()
    for job in schedule["jobs"]:
        running = runner.start(job)
        if running is None:
            continue
        running["process"].wait()
        runner.reap()
    return runner.exit_code()


def run_gpu(schedule, gpu_ids, runner=None):
    runner = runner or JobRunner(schedule)
    runner.root.mkdir(parents=True, exist_ok=True)
    runner._commit()
    pending = list(schedule["jobs"])
    maximum = schedule["scheduler"]["max_jobs_per_gpu"]
    interval = schedule["scheduler"]["poll_seconds"]
    next_poll = runner.clock()
    while pending or runner.running:
        runner.reap()
        now = runner.clock()
        if pending and now >= next_poll:
            memory = runner.gpu_query(gpu_ids)
            active = {gpu: sum(item["gpu"] == gpu for item in runner.running)
                      for gpu in gpu_ids}
            for gpu in gpu_ids:
                if active[gpu] >= maximum:
                    continue
                index = next((i for i, job in enumerate(pending)
                              if memory[gpu] >= job["min_free_mib"]), None)
                if index is None:
                    continue
                job = pending.pop(index)
                if runner.start(job, gpu) is not None:
                    active[gpu] += 1
            states = ", ".join(
                f"GPU {gpu}: {memory[gpu]} MiB free, {active[gpu]}/{maximum} jobs"
                for gpu in gpu_ids)
            print(f"[scheduler] {states}; {len(pending)} jobs pending", flush=True)
            next_poll = now + interval
        if pending or runner.running:
            delay = min(0.25, max(0.01, next_poll - runner.clock()))
            runner.sleep(delay)
    return runner.exit_code()


def _gpu_ids(value):
    if value is None:
        return None
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids or any(not part.isdigit() for part in ids) or len(ids) != len(set(ids)):
        raise ValueError("gpu-ids must be unique comma-separated nonnegative integers")
    return ids


def _schedule_paths(values):
    paths = []
    for group in values:
        for value in group:
            paths.extend(part.strip() for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("At least one schedule YAML is required")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd-config-yaml", "--cmd_config_yaml", required=True,
                        action="append", nargs="+",
                        help="One or more schedule YAML files; comma-separated paths also work")
    parser.add_argument("--output-root")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--gpu-ids", "--gpu_ids",
                        help="Allowed physical GPU indices, comma separated")
    parser.add_argument("--min-free-mib", "--GPU_memory", type=int)
    parser.add_argument("--max-jobs-per-gpu", "--max_procs_per_gpu", type=int)
    parser.add_argument("--poll-seconds", "--sleep_time", type=float)
    occupancy = parser.add_mutually_exclusive_group()
    occupancy.add_argument("--occupy-after-run", "--enable_gpu_occupy",
                           dest="occupancy_enabled", action="store_true", default=None)
    occupancy.add_argument("--no-occupy-after-run", dest="occupancy_enabled",
                           action="store_false")
    parser.add_argument("--occupancy-script", "--gpu_occupy_script")
    args = parser.parse_args(argv)
    schedule = load_schedules(_schedule_paths(args.cmd_config_yaml), args.output_root, {
        "min_free_mib": args.min_free_mib,
        "max_jobs_per_gpu": args.max_jobs_per_gpu,
        "poll_seconds": args.poll_seconds,
        "occupancy_enabled": args.occupancy_enabled,
        "occupancy_script": args.occupancy_script,
    })
    printable = {**schedule, "output_root": str(schedule["output_root"])}
    print(json.dumps(printable, indent=2))
    if not args.execute:
        return 0

    runner = JobRunner(schedule)
    occupancy_supervisor = None
    previous = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    for name in (signal.SIGINT, signal.SIGTERM):
        previous[name] = signal.signal(name, interrupt)
    try:
        gpu_ids = _gpu_ids(args.gpu_ids)
        if schedule["scheduler"]["occupancy"]["enabled"] and not gpu_ids:
            raise ValueError("GPU occupancy requires explicit --gpu-ids")
        result = run_gpu(schedule, gpu_ids, runner) if gpu_ids else run_serial(schedule, runner)
        if schedule["scheduler"]["occupancy"]["enabled"]:
            occupancy_supervisor = OccupancySupervisor(runner)
            occupancy_supervisor.start(gpu_ids)
            print("[scheduler] Real jobs finished; occupancy runs until it exits or is interrupted",
                  flush=True)
            occupancy_result = occupancy_supervisor.monitor()
            result = result or occupancy_result
        return result
    except KeyboardInterrupt:
        if occupancy_supervisor is not None:
            occupancy_supervisor.terminate()
        runner.terminate()
        return 130
    except Exception:
        if occupancy_supervisor is not None:
            occupancy_supervisor.terminate()
        runner.terminate()
        raise
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


if __name__ == "__main__":
    raise SystemExit(main())
