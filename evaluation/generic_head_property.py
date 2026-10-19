"""Replay three domain-agnostic banks on already committed head-study inputs."""

import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, write_json, write_tensors
from core.fl import restore_model
from evaluation import synthetic_head_property as legacy
from evaluation.generic_head_generation import cases
from evaluation.image_property_scoring import METHODS, classification_stats, paired_bootstrap


ROOT = Path("outputs/diagnostics/generic_head_n20_gpu5_20261018")
SOURCE = Path("outputs/diagnostics/synthetic_head_n20_gpu5_20261017")


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def prepare():
    """No target labels/text/images read: package reviewed public generations."""
    if (ROOT / "plan.json").exists():
        raise FileExistsError("Plan already frozen")
    design = read_json(ROOT / "design.json")
    if design["cases"] != cases():
        raise ValueError("Predeclared prompts changed")
    review = read_json(ROOT / "generated/visual_review.json")
    if not review["all_72_accepted"] or review["scoring_started"]:
        raise ValueError("All bank cases must be reviewed before scoring")
    generated = read_json(ROOT / "generated/complete.json")["images"]
    if len(generated) != 72 or any(
        any(row[k] != case[k] for k in case)
        for row, case in zip(generated, design["cases"], strict=True)
    ):
        raise ValueError("Generated cases differ from frozen design")
    original = read_json(SOURCE / "plan.json")
    old_complete = read_json(SOURCE / "complete.json")
    if file_hash(SOURCE / "report.json") != old_complete["report_sha256"]:
        raise ValueError("Original report changed")
    hashes = {str(ROOT / "design.json"): file_hash(ROOT / "design.json")}
    # Original code has evolved; validate committed data manifests, not its old code hashes.
    for marker in ["capture_complete.json", "reconstruction/complete.json",
                   "reconstructed/complete.json", "oracle/complete.json"]:
        manifest = read_json(SOURCE / marker)
        committed = {str(SOURCE / name): value for name, value in manifest["hashes"].items()}
        verify_hashes(committed)
        hashes.update(committed)
        hashes[str(SOURCE / marker)] = file_hash(SOURCE / marker)
    for path in [SOURCE / "report.json", SOURCE / "plan.json", SOURCE / "complete.json",
                 SOURCE / "private/targets.json", ROOT / "generated/complete.json",
                 ROOT / "generated/visual_review.json",
                 Path(original["model"]) / "model.json", Path(original["model"]) / "model.safetensors"]:
        hashes[str(path)] = file_hash(path)
    for name in ["model.json", "model.safetensors"]:
        key = str(Path(original["model"]) / name)
        if hashes[key] != original["frozen_hashes"][key]:
            raise ValueError("Original victim model changed")
    for name in ["private/targets.json"]:
        key = str(SOURCE / name)
        if hashes[key] != original["frozen_hashes"][key]:
            raise ValueError("Original evaluation target manifest changed")
    pixels = set()
    for b in range(3):
        folder = ROOT / f"bank-{b}"
        folder.mkdir(exist_ok=False)
        bank, tensors = [], {}
        for row in [r for r in generated if r["bank"] == b]:
            path = ROOT / "generated" / row["file"]
            if file_hash(path) != row["sha256"]:
                raise ValueError("Generated candidate changed")
            hashes[str(path)] = row["sha256"]
            with Image.open(path) as image:
                arr = np.array(image.convert("RGB"), copy=True)
            import hashlib
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            if digest in pixels:
                raise ValueError("Duplicate generated candidate")
            pixels.add(digest)
            identifier = f"candidate-{len(bank):02d}"
            tensors[identifier] = torch.from_numpy(arr)
            bank.append({"id": identifier, "label": row["property_label"],
                         "region_prompt": row["region"], "seed": row["seed"],
                         "pixel_sha256": digest, "generated_file": str(path)})
        if Counter(r["label"] for r in bank) != {"head": 12, "non-head": 12}:
            raise ValueError("Candidate class imbalance")
        write_json(folder / "public/bank.json", bank)
        write_tensors(folder / "public/bank.safetensors", tensors)
        for name in ["bank.json", "bank.safetensors"]:
            path = folder / "public" / name
            hashes[str(path)] = file_hash(path)
        # Public inference reuses observations and committed reconstruction only.
        for name in ["captures", "reconstruction"]:
            (folder / name).symlink_to((SOURCE / name).resolve(), target_is_directory=True)
    plan = {k: original[k] for k in ["classes", "targets", "model", "primary_score", "top_m",
                                    "text_only_prompt", "text_conditions"]}
    plan.update(banks=3, source=str(SOURCE), frozen_hashes=hashes,
                primary_endpoint="Mean per-bank reconstructed centered balanced accuracy; no bank selection",
                statistical_unit="20 target images; three banks are repeated measurements",
                domain_prior="No medical information in generated prompts; recovered QA may reveal domain")
    for b in range(3):
        write_json(ROOT / f"bank-{b}/plan.json", plan)
    write_json(ROOT / "plan.json", plan)
    print("Reviewed banks and reused inputs frozen", flush=True)


def evaluate(plan):
    """Only called after all six prediction phases have committed."""
    truth = read_json(SOURCE / "private/targets.json")
    labels = [r["label"] for r in truth]
    baseline = read_json(SOURCE / "report.json")
    text_em = baseline["text_recovery"]["per_target"]
    aggregate = {"plan_sha256": file_hash(ROOT / "plan.json"), "banks": {},
                 "independent_images": 20, "bank_repeats": 3, "baseline": baseline["phases"]}
    for b in range(3):
        folder = ROOT / f"bank-{b}"
        # The legacy evaluator needs references only now, after predictions exist.
        (folder / "private").symlink_to((SOURCE / "private").resolve(), target_is_directory=True)
        legacy.ROOT = folder
        report = legacy.evaluate(plan)
        for phase in plan["text_conditions"]:
            preds = [read_json(folder / phase / "predictions" / f"{i:05d}-{phase}.json") for i in range(20)]
            old = [read_json(SOURCE / phase / "predictions" / f"{i:05d}-{phase}.json") for i in range(20)]
            for p, o in zip(preds, old, strict=True):
                if p["text_sha256"] != o["text_sha256"] or not np.allclose(p["text_only"], o["text_only"], atol=2e-5, rtol=0):
                    raise ValueError("Shared text or text-only control changed")
            scores = {m: [p["class_scores"][m] for p in preds] for m in METHODS}
            scores.update({m: [p[m] for p in preds] for m in ["model_only", "text_only"]})
            groups = {m: [i for i, r in enumerate(truth) if r["modality"] == m]
                      for m in ["CT", "MRI", "X-Ray"]}
            groups.update({f"joint_em_{flag}": [i for i, r in enumerate(text_em) if r["joint_em"] == flag]
                           for flag in [False, True]})
            groups.update({f"head_terms_{flag}": [i for i, r in enumerate(truth) if r["head_terms_present"] == flag]
                           for flag in [False, True]})
            report["phases"][phase]["groups"] = {
                name: {m: classification_stats([labels[i] for i in ids], [v[i] for i in ids], plan["classes"])
                       for m, v in scores.items()} for name, ids in groups.items()}
            report["phases"][phase]["paired_vs_medical_ci95"] = paired_bootstrap(
                labels, scores["centered"], plan["classes"], [p["class_scores"]["centered"] for p in old])
            matched_wrong = {}
            for kind in ["same_modality", "same_modality_opposite_property"]:
                target_credit = []
                for i, pred in enumerate(preds):
                    credits = []
                    for j, values in zip(pred["shuffled_target_runs"], pred["shuffled_class_scores"], strict=True):
                        if truth[i]["modality"] != truth[j]["modality"]:
                            continue
                        if kind.endswith("opposite_property") and labels[i] == labels[j]:
                            continue
                        s = values["centered"]
                        credits.append(float(s is not None and plan["classes"][int(np.argmin(s))] == labels[i]))
                    if credits:
                        target_credit.append(float(np.mean(credits)))
                matched_wrong[kind] = {"targets": len(target_credit),
                                       "accuracy": float(np.mean(target_credit))}
            report["phases"][phase]["matched_wrong_gradient"] = matched_wrong
        write_json(folder / "report.json", report)
        aggregate["banks"][str(b)] = report["phases"]
    write_json(ROOT / "report.json", aggregate)


def run():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != legacy.GPU5 or torch.cuda.device_count() != 1:
        raise ValueError("Only authorized GPU 5 may be exposed")
    if (ROOT / "started.json").exists():
        raise FileExistsError("No automatic retry or overwrite")
    torch.set_num_threads(1)
    started = time.monotonic()
    write_json(ROOT / "started.json", {"plan_sha256": file_hash(ROOT / "plan.json")})
    try:
        plan = read_json(ROOT / "plan.json")
        source_hashes = read_json(ROOT / "runtime_source.json")["hashes"]
        verify_hashes(source_hashes)
        verify_hashes(plan["frozen_hashes"])
        if torch.cuda.mem_get_info()[0] < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free; stop without changing settings")
        adapter = restore_model(plan["model"], "cuda:0")
        fingerprint = adapter.fingerprint()
        versions = {n: p._version for n, p in adapter.named_parameters()}
        # All reconstructed decisions precede all oracle decisions; no labels read.
        for phase in plan["text_conditions"]:
            for b in range(plan["banks"]):
                legacy.ROOT = ROOT / f"bank-{b}"
                legacy.predict(adapter, plan, phase, fingerprint)
                write_json(ROOT / "progress.json", {"phase": phase, "bank_completed": b,
                           "seconds": time.monotonic() - started})
                gc.collect()
        if adapter.fingerprint() != fingerprint or any(p._version != versions[n] for n, p in adapter.named_parameters()):
            raise ValueError("Victim model changed")
        verify_hashes(plan["frozen_hashes"])
        verify_hashes(source_hashes)
        write_json(ROOT / "predictions_complete.json", {
            "predictions": 120, "model_unchanged": True, "private_reference_labels_read": False,
            "hashes": {str(p): file_hash(p) for p in ROOT.glob("bank-*/*/predictions/*.json")}})
        evaluate(plan)
        write_json(ROOT / "complete.json", {"seconds": time.monotonic() - started,
                   "report_sha256": file_hash(ROOT / "report.json"), "model_unchanged": True,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        print("Three generic banks evaluated", flush=True)
    except BaseException:
        write_json(ROOT / "failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.stage == "prepare":
        try:
            prepare()
        except BaseException:
            write_json(ROOT / "prepare_failed.json", {"traceback": traceback.format_exc()})
            raise
    else:
        run()
