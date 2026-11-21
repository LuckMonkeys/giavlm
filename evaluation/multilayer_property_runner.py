"""Detached single-host GPU launcher; never retry a failed stage or change settings."""

import argparse
import csv
import fcntl
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback

from core.artifacts import file_hash, read_json, write_json


REPO = Path(os.environ.get("GIAVLM_REPO", Path(__file__).resolve().parents[1])).resolve()
DEFAULT_ROOT = REPO / "outputs/diagnostics/multilayer_property_20261101"
GIA = Path("/home/zx/nas/miniconda3/envs/gia/bin/python")
QWEN = Path("/home/zx/nas/miniconda3/envs/qwen-image-21/bin/python")


def gpu_rows():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    return [{"index": int(row[0]), "uuid": row[1].strip(), "used_mib": int(row[2]),
             "free_mib": int(row[3]), "utilization": int(row[4])}
            for row in csv.reader(result.stdout.splitlines())]


def select_gpu(rows, requested=None):
    eligible = [row for row in rows if row["used_mib"] < 512 and row["free_mib"] >= 70 * 1024
                and row["utilization"] == 0 and (requested is None or row["index"] == requested)]
    if not eligible:
        raise RuntimeError("No idle GPU with 70 GiB free; no sharing or automatic retry")
    return min(eligible, key=lambda row: (row["used_mib"], -row["free_mib"], row["index"]))


def alive(pid):
    try:
        os.kill(pid, 0)
        stat = Path(f"/proc/{pid}/stat")
        return not stat.exists() or stat.read_text().split(") ", 1)[1].split()[0] != "Z"
    except ProcessLookupError:
        return False


def snapshot(root, stage):
    destination = root / f"source-{stage}-v1"
    if destination.exists():
        raise FileExistsError("Frozen source already exists; do not overwrite an attempted stage")
    for package in ["core", "attacks", "defenses", "metrics", "evaluation", "utils"]:
        for source in (REPO / package).rglob("*.py"):
            relative = source.relative_to(REPO)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    hashes = {str(path): file_hash(path) for path in destination.rglob("*.py")}
    write_json(root / f"source-{stage}-manifest.json", {"hashes": hashes,
               "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()})
    return destination


def launch(root, stage, requested):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "launcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for folder in root.glob("local-*"):
        if (folder / "launch.json").exists():
            prior = read_json(folder / "launch.json")
            if alive(prior["controller_pid"]):
                raise RuntimeError("An existing local controller is still alive")
    folder = root / f"local-{stage}"
    if folder.exists():
        raise FileExistsError("Stage launch already recorded; never retry or overwrite automatically")
    for python in [GIA, QWEN]:
        if not python.is_file():
            raise FileNotFoundError(python)
    if stage == "probe" and not (root / "public/review.json").is_file():
        raise RuntimeError("Inspect generated public images and commit public/review.json first")
    if stage == "fixed-qa" and not (root / "fresh/complete.json").is_file():
        raise RuntimeError("Natural-QA baseline must be committed before the fixed-QA comparison")
    if stage == "generation" and (root / "generation/started.json").exists():
        raise FileExistsError("Generation already attempted")
    selected = select_gpu(gpu_rows(), requested)
    folder.mkdir()
    source = snapshot(root, stage)
    # Recheck immediately before launching, after snapshot IO.
    fresh = select_gpu(gpu_rows(), selected["index"])
    if fresh["uuid"] != selected["uuid"]:
        raise RuntimeError("Physical GPU changed during launch")
    if stage == "generation":
        plan = read_json(root / "abdomen-generation-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen local generation plan")
        write_json(root / "generation/plan.json", plan)
    elif stage == "probe":
        plan = read_json(root / "probe-plan-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen local probe plan")
        write_json(root / "probe-plan.json", plan)
    elif stage == "fixed-qa":
        plan = read_json(root / "fixed-qa-plan-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen matched fixed-QA plan")
        write_json(root / "fixed-qa/plan.json", plan)
    elif stage == "causal":
        plan = read_json(root / "causal-plan-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen public causal feature check")
        write_json(root / "causal/plan.json", plan)
    elif stage == "multi-qa":
        plan = read_json(root / "multi-qa-plan-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen public multi-QA calibration")
        write_json(root / "multi-qa/plan.json", plan)
    elif stage == "qa-controls":
        plan = read_json(root / "qa-controls-plan-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen public QA and QKV controls")
        write_json(root / "qa-controls/plan.json", plan)
    else:
        plan = read_json(root / "medical-kv-plan-template.json")
        plan.update(gpu_uuid=selected["uuid"], status="frozen private natural-QA medical QKV transfer")
        write_json(root / "medical-kv/plan.json", plan)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": selected["uuid"], "PYTHONPATH": str(source),
                   "GIAVLM_REPO": str(REPO),
                   "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    runner_script = str(source / "evaluation/multilayer_property_runner.py")
    command = [str(GIA), runner_script, "run", "--root", str(root), "--stage", stage]
    with (folder / "controller.log").open("a") as log:
        process = subprocess.Popen(command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"controller_pid": process.pid, "host": socket.gethostname(), "gpu": fresh,
              "command": command, "source": str(source), "python": str(GIA),
              "worker_python": str(QWEN if stage == "generation" else GIA), "stage": stage,
              "code_manifest_sha256": file_hash(root / f"source-{stage}-manifest.json")}
    write_json(folder / "launch.json", record)
    with (folder / "watcher.log").open("a") as log:
        watcher = subprocess.Popen([str(GIA), runner_script, "watch",
                                   "--root", str(root), "--stage", stage, "--pid", str(process.pid)],
                                  cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(folder / "watcher-launch.json", {"pid": watcher.pid, "controller_pid": process.pid})
    print({"stage": stage, "gpu": selected["index"], "controller_pid": process.pid,
           "watcher_pid": watcher.pid, "log": str(folder / "controller.log")}, flush=True)


def run(root, stage):
    folder = root / f"local-{stage}"
    write_json(folder / "started.json", {"pid": os.getpid(), "host": socket.gethostname()})
    child = None

    def stopped(number, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise RuntimeError(f"Controller interrupted by signal {number}")

    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    try:
        for path, sha in read_json(root / f"source-{stage}-manifest.json")["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen source changed")
        if stage == "generation":
            command = [str(QWEN), "-m", "evaluation.public_style_generation", "--root", str(root)]
        elif stage == "probe":
            command = [str(GIA), "-m", "evaluation.multilayer_property_gpu", "--root", str(root)]
        elif stage == "fixed-qa":
            command = [str(GIA), "-m", "evaluation.fixed_qa_property", "--root", str(root)]
        elif stage == "causal":
            command = [str(GIA), "-m", "evaluation.causal_visual_feature_check", "--root", str(root)]
        elif stage == "multi-qa":
            command = [str(GIA), "-m", "evaluation.public_multiqa_calibration", "--root", str(root)]
        elif stage == "qa-controls":
            source = Path(__file__).resolve().parents[1]
            command = [str(GIA), str(source / "evaluation/qa_type_controls.py"), "--root", str(root)]
        else:
            source = Path(__file__).resolve().parents[1]
            command = [str(GIA), str(source / "evaluation/medical_kv_transfer.py"), "--root", str(root)]
        child = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(folder / "worker.json", {"pid": child.pid, "command": command})
        code = child.wait()
        if code != 0:
            raise RuntimeError(f"{stage} worker exited with code {code}; no retry")
        status = "awaiting_public_visual_review" if stage == "generation" else "complete"
        write_json(folder / "complete.json", {"status": status, "worker_pid": child.pid})
        print(status, flush=True)
    except BaseException:
        write_json(folder / "failed.json", {"traceback": traceback.format_exc()})
        raise


def watch(root, stage, pid):
    folder = root / f"local-{stage}"
    while True:
        if (folder / "complete.json").exists():
            write_json(folder / "watcher-final.json", read_json(folder / "complete.json"))
            return
        if (folder / "failed.json").exists() or not alive(pid):
            write_json(folder / "watcher-final.json", {"status": "failed_or_controller_dead"})
            return
        write_json(folder / "watcher-progress.json", {"status": "running", "controller_pid": pid,
                   "worker": read_json(folder / "worker.json") if (folder / "worker.json").exists() else None,
                   "time": time.time()})
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "run", "watch", "status"])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage", choices=["generation", "probe", "fixed-qa", "causal", "multi-qa", "qa-controls", "medical-kv"], default="generation")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pid", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "launch":
        launch(root, args.stage, args.gpu)
    elif args.action == "run":
        run(root, args.stage)
    elif args.action == "watch":
        watch(root, args.stage, args.pid)
    else:
        for folder in root.glob("local-*"):
            if (folder / "launch.json").exists():
                record = read_json(folder / "launch.json")
                print(folder.name, {"controller_alive": alive(record["controller_pid"]),
                      "complete": read_json(folder / "complete.json") if (folder / "complete.json").exists() else None,
                      "failed": (folder / "failed.json").exists()})


if __name__ == "__main__":
    main()
