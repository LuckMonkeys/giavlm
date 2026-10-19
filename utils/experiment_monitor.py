"""Read-only periodic health reports for the remote text benchmark.

Does not start, stop, retry, or change experiments. Only aggregate progress and
process checks are saved; no private text, token IDs, or images are exported.
"""
import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time
from zoneinfo import ZoneInfo


REMOTE_PYTHON = "/home/zx/nas/miniconda3/envs/gia/bin/python"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_matches(pid, needle):
    try:
        command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return bool(command) and needle in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def remote_probe(host, expected):
    code = '''import json,subprocess,socket
from pathlib import Path
expected=json.loads(PAYLOAD)
processes={}
for pid,needle in expected.items():
    try:
        command=Path('/proc/'+pid+'/cmdline').read_bytes().replace(b'\\0',b' ').decode()
        processes[pid]=bool(command) and needle in command
    except (FileNotFoundError,ProcessLookupError):processes[pid]=False
query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
gpus=[]
if query.returncode==0:
    for line in query.stdout.splitlines():
        index,uuid,used,total,util=[x.strip() for x in line.split(',')]
        gpus.append({'index':int(index),'uuid':uuid,'memory_used_mib':int(used),'memory_total_mib':int(total),'utilization_percent':int(util)})
print(json.dumps({'host':socket.gethostname(),'processes':processes,'gpus':gpus,'gpu_probe_ok':query.returncode==0}))
'''.replace("PAYLOAD", repr(json.dumps(expected)))
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=yes",
         host, shlex.join([REMOTE_PYTHON, "-c", code])],
        capture_output=True, text=True, timeout=40, check=True)
    return json.loads(result.stdout)


def progress(root, job):
    phase, dataset, model = job.split("-")
    if phase == "train":
        path = root / "models" / dataset / model / "training.json"
        return {"round": read(path)["round"]} if path.exists() else {"round": None}
    directory = root / "attacks" / dataset / model
    status_file = directory / "state.json"
    state = read(status_file) if status_file.exists() else {}
    result_files = list(directory.glob("run-*/attacks/*/result.json"))
    statuses = Counter(read(path)["status"] for path in result_files)
    frozen = read(root / "data" / dataset / "frozen.json")
    value = {"phase": state.get("phase"), "run": state.get("run"), "image": state.get("image"),
             "captured_samples": len(list(directory.glob("run-*/capture_complete.json"))),
             "planned_samples": len(frozen["selected"]), "finished_attacks": len(result_files),
             "planned_attacks": 2 * len(frozen["selected"]), "statuses": dict(statuses),
             "last_progress_at": status_file.stat().st_mtime if status_file.exists() else None}
    if state.get("phase") == "attack":
        attack = directory / f"run-{state['run']:05d}" / "attacks" / state["image"]
        pointer = attack / "checkpoint.json"
        try:
            if pointer.exists():
                saved = read(attack / "checkpoints" / read(pointer)["generation"] / "state.json")
                search = saved["state"]
                value["checkpoint"] = {"stage": search["stage"], "depth": search["depth"],
                                       "elapsed_seconds": saved["elapsed"],
                                       "vocab_cursor": search["vocab_cursor"]}
                value["last_progress_at"] = pointer.stat().st_mtime
        except FileNotFoundError:
            value["checkpoint"] = {"status": "being_published"}
    return value


def collect(root, host):
    controller = read(root / "controller.json")
    jobs = {p.stem: read(p) for p in sorted((root / "jobs").glob("*.json"))}
    expected = {}
    if controller["status"] == "running":
        expected[str(controller["pid"])] = str(root / "setup/phase-controller.py")
    for name, job in jobs.items():
        if job["status"] == "running":
            expected[str(job["pid"])] = str(root / f"jobs/{name}.json")
            if job.get("child_pid"):
                expected[str(job["child_pid"])] = "-m core.commands" if name.startswith("train-") else "-m evaluation.text_benchmark"
    now = time.time()
    report = {"checked_at": now, "checked_at_local": datetime.fromtimestamp(now, ZoneInfo("Asia/Shanghai")).isoformat(),
              "phase": controller["phase"], "controller_status": controller["status"],
              "jobs": {}, "alerts": [], "planned_attacks": 288, "finished_attacks": 0}
    try:
        report["remote"] = remote_probe(host, expected)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        report["remote"] = {"status": "unverified", "error_type": type(error).__name__}
        report["alerts"].append("SSH 或 GPU 探测失败，本次无法核验远端进程；下一周期重新检查。")
    # Refresh statuses after the probe to allow normal process exits during a check.
    controller = read(root / "controller.json")
    report.update(phase=controller["phase"], controller_status=controller["status"])
    probed = report["remote"].get("processes", {})
    if report["remote"].get("gpu_probe_ok") is False:
        report["alerts"].append("远端 nvidia-smi 探测失败，GPU 状态未核验。")
    if controller["status"] == "running":
        if probed.get(str(controller["pid"])) is False:
            report["alerts"].append("阶段调度器标记 running，但远端 PID 已退出或命令不匹配。")
        if now - controller["updated_at"] > 180:
            report["alerts"].append("阶段调度器超过 3 分钟未更新状态。")
    active_gpus = set()
    for name in jobs:
        job = read(root / f"jobs/{name}.json")
        item = {"status": job["status"], "gpu_uuid": job["gpu_uuid"], **progress(root, name)}
        if job["status"] == "running":
            item["worker_verified"] = probed.get(str(job["pid"]))
            item["child_verified"] = probed.get(str(job.get("child_pid")))
            if item["worker_verified"] is False or item["child_verified"] is False:
                report["alerts"].append(f"{name} 标记 running，但工作进程已退出或命令不匹配。")
            active_gpus.add(job["gpu_uuid"])
            if item.get("last_progress_at") and now - item["last_progress_at"] > 1800:
                report["alerts"].append(f"{name} 超过 30 分钟没有新进度，需要核查。")
        if job["status"] == "failed":
            report["alerts"].append(f"{name} 已失败，未自动重启。")
        report["jobs"][name] = item
        report["finished_attacks"] += item.get("finished_attacks", 0)
    report["assigned_project_gpu_count"] = len(active_gpus)
    if len(active_gpus) > 2:
        report["alerts"].append("项目活跃任务分配超过两张 GPU，需要立即核查。")
    finalizer = read(root / "finalizer.json")
    report["finalizer_status"] = finalizer["status"]
    if finalizer["status"] in {"waiting", "running"}:
        report["finalizer_verified"] = process_matches(finalizer["pid"], str(root / "setup/local-finalizer.py"))
        if not report["finalizer_verified"]:
            report["alerts"].append("本机汇总进程已退出，尚未完成报告。")
    if finalizer["status"] == "failed":
        report["alerts"].append("本机汇总任务失败，需要检查 finalizer.json。")
    report["terminal"] = controller["status"] == "failed" or finalizer["status"] in {"completed", "failed"}
    report["health"] = "attention" if report["alerts"] else "completed" if finalizer["status"] == "completed" else "healthy"
    return report


def markdown(report):
    lines = ["# Benchmark 定期巡检", "", f"- 最近检查：{report['checked_at_local']}",
             f"- 状态：{report['health']}；阶段：{report['phase']}",
             f"- 已结束攻击：{report['finished_attacks']} / {report['planned_attacks']}",
             f"- 项目活跃任务分配 GPU：{report['assigned_project_gpu_count']} 张；上限 2 张",
             f"- 汇总任务：{report['finalizer_status']}", "",
             "| Job | Status | Progress | PID verified |", "|---|---|---|---|"]
    for name, job in report["jobs"].items():
        detail = (f"Round {job.get('round')}" if name.startswith("train-") else
                  f"capture {job['captured_samples']}/{job['planned_samples']}; "
                  f"attack {job['finished_attacks']}/{job['planned_attacks']}")
        verified = job.get("child_verified") if job["status"] == "running" else "—"
        lines.append(f"| {name} | {job['status']} | {detail} | {verified} |")
    lines += ["", "## 需要关注", ""]
    lines.extend(f"- {alert}" for alert in report["alerts"])
    if not report["alerts"]:
        lines.append("- 本次检查未发现异常。")
    lines += ["", "此任务只巡检和记录，不重试实验、不修改参数、不额外占用 GPU。",
              "运行完成数量不代表精确恢复成功数量。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--host", default="zx-wn-a100")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-checks", type=int)
    args = parser.parse_args()
    if args.interval <= 0 or (args.max_checks is not None and args.max_checks <= 0):
        parser.error("interval and max-checks must be positive")
    root = args.root.resolve()
    output = args.output or root / "monitor"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        control = {"pid": os.getpid(), "host": socket.gethostname(), "status": "running", "interval_seconds": args.interval}
        checks = 0
        try:
            while True:
                report = collect(root, args.host)
                write(output / "latest.json", report)
                (output / "latest.md").write_text(markdown(report))
                with (output / "history.jsonl").open("a") as history:
                    history.write(json.dumps(report, ensure_ascii=False) + "\n")
                if report["alerts"]:
                    write(output / "alert.json", report)
                checks += 1
                control.update(checks=checks, last_check_at=report["checked_at"], health=report["health"])
                if report["terminal"] or (args.max_checks is not None and checks >= args.max_checks):
                    control.update(status="finished", next_check_at=None)
                    write(output / "timer.json", control)
                    break
                control["next_check_at"] = time.time() + args.interval
                write(output / "timer.json", control)
                print(f"{report['checked_at_local']} {report['health']} {report['finished_attacks']}/288", flush=True)
                time.sleep(args.interval)
        except BaseException as error:
            control.update(status="failed", error_type=type(error).__name__, next_check_at=None)
            write(output / "timer.json", control)
            raise


if __name__ == "__main__":
    main()
