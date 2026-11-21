"""Evaluation-only continuation; never recapture, refit or modify committed scores."""

import argparse
from pathlib import Path
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, write_json
from evaluation.public_multiqa_calibration import evaluate


def finalize(root):
    out = root / "multi-qa"
    if (out / "cpu-finalize-started.json").exists() or (out / "complete.json").exists():
        raise FileExistsError("No automatic retry or overwrite")
    write_json(out / "cpu-finalize-started.json", {"mode": "CPU evaluation only", "time": time.time()})
    files = [out / name for name in ["predictions.json", "predictions-complete.json", "frozen-rules.json",
                                   "rules-complete.json", "public-audit.json", "public-scores.json"]]
    before = {str(p): file_hash(p) for p in files}
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        plan = read_json(out / "plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen input changed")
        if read_json(out / "runtime-source-audit.json")["status"] != "pass":
            raise ValueError("Original execution source not audited")
        evaluate(root, plan)
        if any(file_hash(path) != sha for path, sha in before.items()):
            raise ValueError("Evaluation changed rules or predictions")
        write_json(out / "evaluation-repair.json", {
            "status": "complete", "cause": "Two raw prototype distances compared with old symmetric signed margins",
            "fix": "Only canonicalize original baselines for equality audit; all converted baseline scores exactly match",
            "no_gpu_rerun": True, "no_refit": True, "committed_scores_unchanged": before,
            "seconds": time.monotonic() - start, "source_sha256": file_hash(Path(__file__))})
        write_json(out / "complete.json", {"status": "complete_after_cpu_evaluation_repair",
                   "original_worker_terminal_state": "evaluation assertion failed after all scores committed",
                   "original_worker_to_score_commit_seconds_estimate": (out / "predictions-complete.json").stat().st_mtime - (out / "started.json").stat().st_mtime,
                   "timing_basis": "File mtimes, approximate; not a recorded GPU kernel or full phase timer",
                   "peak_memory_mib": None, "peak_memory_note": "Original worker stopped before its final memory/timing record; no peak claimed",
                   "cpu_evaluation_seconds": time.monotonic() - start,
                   "report_sha256": file_hash(out / "report.json")})
    except BaseException:
        write_json(out / "cpu-finalize-failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.root.resolve())


if __name__ == "__main__":
    main()
