"""Read attributes from already committed convex weights; no optimization replay."""
import argparse
import os
from pathlib import Path
import time
import traceback

import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from core.fl import restore_model
from evaluation.visual_concept_probe import inject_logits, native_hard_loss


def run(root, gpu_uuid):
    out = root / "convex/readout-v2"
    if (out / "started.json").exists():
        raise FileExistsError("No automatic readout retries")
    write_json(out / "started.json", {"pid": os.getpid(), "gpu_uuid": gpu_uuid})
    start = time.monotonic()
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly the freshly selected GPU")
        if torch.cuda.mem_get_info()[0] < 35 * 2**30:
            raise RuntimeError("Readout needs 35 GiB free; no dtype changes")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        plan = read_json(root / "convex/plan.json")
        committed = read_json(root / "convex/predictions_complete.json")
        for path, sha in committed["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Committed reconstruction changed")
        for path in plan["dictionary"]:
            if file_hash(path) != plan["hashes"][path]:
                raise ValueError("Public dictionary changed")
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        fingerprint = adapter.fingerprint()
        dictionary = torch.stack([read_tensors(p)["visual"].float() for p in plan["dictionary"]]).to("cuda:0")
        q = adapter.encode([plan["readout_question"]], adapter.spec.question_length)
        answers = [adapter.encode([answer], adapter.spec.target_length) for answer in ["yes", "no"]]

        @torch.no_grad()
        def readout(weights):
            visual = torch.einsum("m,mbnd->bnd", weights.to("cuda:0"), dictionary)
            # Same FP32-master -> bf16 boundary used by every matching replay.
            native = visual.to(adapter.dtype)
            return [float(native_hard_loss(inject_logits(adapter, native, q, a), a, adapter.pad)) for a in answers]

        outputs, peaks = [], []
        for index in range(committed["queries"]):
            row = read_json(root / "convex/predictions" / f"q{index:02d}.json")
            scores = {"mean": readout(torch.full((len(dictionary),), 1 / len(dictionary))),
                      "best_single": readout(torch.nn.functional.one_hot(
                          torch.tensor(row["best_single"]), len(dictionary)).float())}
            optimized, initial = [], []
            for attempt in row["runs"]:
                tensors = read_tensors(attempt["weights"])
                optimized.append(readout(tensors["best"]))
                initial.append(readout(tensors["initial"]))
                peaks.extend(step["peak_memory_mib"] for step in attempt["trace"])
            scores["optimized"] = optimized[row["selected"]]
            scores["random_initial_mean"] = torch.tensor(initial).mean(0).tolist()
            outputs.append({"query": index, "scores": scores, "initial_scores": initial,
                            "optimized_scores": optimized, "near_optimal": row["near_optimal"],
                            "selected": row["selected"]})
            write_json(out / "progress.json", {"completed": index + 1, "total": committed["queries"]})
        path = root / "convex/readout_predictions.json"
        if path.exists():
            raise FileExistsError("Do not overwrite existing property predictions")
        write_json(path, outputs)
        if adapter.fingerprint() != fingerprint:
            raise ValueError("Readout mutated the public model")
        elapsed = time.monotonic() - start
        report = {"readout_sha256": file_hash(path),
                  "predictions_sha256": file_hash(root / "convex/predictions_complete.json"),
                  "seconds": committed["seconds"] + elapsed, "readout_seconds": elapsed,
                  "peak_memory_mib": max(max(peaks), torch.cuda.max_memory_allocated() / 2**20),
                  "peak_memory_note": "Maximum recorded optimizer trace peak and replacement-readout peak; original final worker peak was not recorded",
                  "optimization_steps": committed["queries"] * len(plan["seeds"]) * plan["steps"],
                  "readout_uses_attribute_for_optimization": False,
                  "optimization_replayed": False,
                  "optimization_end_fingerprint_recorded": False,
                  "readout_model_unchanged": True, "native_readout_dtype": str(adapter.dtype),
                  "original_readout_failure_sha256": file_hash(root / "convex/failed.json")}
        write_json(out / "complete.json", report)
        write_json(root / "convex/complete.json", report)
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    run(args.root, args.gpu_uuid)


if __name__ == "__main__":
    main()
