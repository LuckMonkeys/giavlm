"""CPU-only extension of the frozen medical QKV test to the original cohort A."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import io
import os
from pathlib import Path
import random
import shutil
import socket
import time
import traceback
import uuid

import numpy as np
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from core.fl import load_observation
from evaluation.deep_property_probe import HISTORICAL
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.medical_kv_transfer import METHODS, answer_type, score
from evaluation.multilayer_property_data import TERMS
from evaluation.multilayer_property_gpu import CLASSES


def prepare(root):
    out = root / "medical-kv-a"
    if (out / "plan.json").exists():
        raise FileExistsError("Cohort A plan already frozen")
    previous = read_json(root / "medical-kv/plan.json")
    rule_path = root / "medical-kv/public-rule.json"
    rule = read_json(rule_path)
    if rule["primary"] != previous["primary"] or rule["primary"] != "L24_v":
        raise ValueError("Do not reselect the previously frozen public rule")
    uploads, mapping, hashes, protocols = [], {}, {}, []
    for index in range(20):
        folder = HISTORICAL / "captures" / f"run-{index:05d}" / "private"
        metadata = read_json(folder / "observation.json")
        if metadata["model_fingerprint"] != previous["model_fingerprint"]:
            raise ValueError("A and B/C victim models differ")
        protocols.append(metadata["training"])
        identifier = uuid.uuid4().hex
        uploads.append({"id": identifier, "observation": str(folder.resolve())})
        mapping[identifier] = index
        hashes.update({str((folder / name).resolve()): file_hash(folder / name)
                       for name in ("observation.json", "update.safetensors")})
    if any(protocol != protocols[0] for protocol in protocols):
        raise ValueError("Cohort A contains mixed upload protocols")
    training = protocols[0]
    if (training["knowledge"] != "private" or training["algorithm"] != "fedsgd"
            or training["local_steps"] != 1 or training["batch_size"] != 1):
        raise ValueError("Expected original private single-step FedSGD uploads")
    differences = {key: {"A": training.get(key), "BC": previous["training"].get(key)}
                   for key in sorted(set(training) | set(previous["training"]))
                   if training.get(key) != previous["training"].get(key)}
    if set(differences) != {"lr", "rounds", "snapshots", "token_lengths_known"}:
        raise ValueError("Unexpected A versus B/C protocol difference")
    random.SystemRandom().shuffle(uploads)
    write_json(out / "private/upload-list.json", uploads)
    write_json(out / "private/upload-map.json", mapping)
    write_json(out / "public-rule.json", rule)
    references = [Path(previous["head_prototypes"]), root / "public/abdomen-prototypes.safetensors",
                  rule_path, root / "medical-kv/plan.json",
                  Path("outputs/diagnostics/image_property_next_20261028/deep-probe/predictions.json"),
                  out / "private/upload-list.json", out / "private/upload-map.json"]
    hashes.update({str(path.resolve()): file_hash(path) for path in references})
    write_json(out / "plan.json", {"cohort": "A", "images": 20, "primary": rule["primary"],
        "methods": list(METHODS), "model_fingerprint": previous["model_fingerprint"],
        "head_prototypes": previous["head_prototypes"], "training": training,
        "protocol_differences_from_BC": differences, "hashes": hashes,
        "private_evaluation_targets_sha256": file_hash(HISTORICAL / "private/targets.json"),
        "scope": "Repeated original 20-image medical cohort A; no new images or independent confirmation",
        "scoring": "Exactly the existing two-prototype gradient-rowspace distance and argmin; no target tuning",
        "target_text_or_lengths_used_by_classifier": False,
        "replay_absolute_tolerance": 1e-12, "source_sha256": file_hash(Path(__file__))})


def validate_observation(observation, plan):
    if (observation.model_fingerprint != plan["model_fingerprint"]
            or asdict(observation.training) != plan["training"]
            or observation.sample_count != 1 or observation.public_images is not None
            or observation.public_questions or observation.public_targets
            or observation.public_question_ids or observation.public_target_ids):
        raise ValueError("Unexpected original cohort A Observation")


def evaluate(out, predictions):
    """Evaluator joins private labels and QA categories only after prediction commit."""
    plan = read_json(out / "plan.json")
    commit = read_json(out / "predictions-complete.json")
    if file_hash(out / "predictions.json") != commit["sha256"]:
        raise ValueError("Prediction commit changed")
    target_path = HISTORICAL / "private/targets.json"
    if file_hash(target_path) != plan["private_evaluation_targets_sha256"]:
        raise ValueError("Original A evaluation labels changed")
    targets = {row["run"]: row for row in read_json(target_path)}
    mapping = read_json(out / "private/upload-map.json")
    truth = [targets[mapping[row["id"]]] for row in predictions]
    composition = {"n": len(truth), "region": dict(Counter(row["region"] for row in truth)),
                   "modality": dict(Counter(row["modality"] for row in truth)),
                   "answer_type": dict(Counter(answer_type(row) for row in truth))}
    results = {}
    for prop, classes in CLASSES.items():
        ids_by_group = {"all": list(range(len(truth))),
            "no_literal_property_terms": [i for i, row in enumerate(truth)
                if not TERMS[prop].search(row["row"]["question"] + " " + row["row"]["target"])],
            **{m: [i for i, row in enumerate(truth) if row["modality"] == m] for m in ("CT", "MRI", "X-Ray")},
            **{m: [i for i, row in enumerate(truth) if answer_type(row) == m] for m in ("yes_no", "other")}}
        results[prop] = {}
        for name, ids in ids_by_group.items():
            labels = [classes[0] if truth[i]["region"] == prop else classes[1] for i in ids]
            values = {m: [predictions[i]["scores"][prop][m] for i in ids] for m in METHODS}
            metrics = {m: {**classification_stats(labels, v, classes),
                       **(paired_bootstrap(labels, v, classes) or {})} for m, v in values.items()}
            comparisons = {}
            if labels:
                for method in METHODS:
                    baseline = method.split("_")[0] + "_q"
                    comparisons[method] = {"comparator": baseline,
                        "accuracy_gain": metrics[method]["accuracy"] - metrics[baseline]["accuracy"],
                        "balanced_accuracy_gain": metrics[method]["balanced_accuracy"] - metrics[baseline]["balanced_accuracy"]
                            if metrics[method]["balanced_accuracy"] is not None else None,
                        **paired_bootstrap(labels, values[method], classes, values[baseline])}
            results[prop][name] = {"metrics": metrics, "paired_vs_same_layer_q": comparisons}
    return {"cohort": "A", "composition": composition, "properties": results,
        "primary": plan["primary"], "scope": plan["scope"],
        "protocol_differences_from_BC": plan["protocol_differences_from_BC"],
        "prediction_sha256": commit["sha256"], "uses_target_text": False,
        "uses_target_lengths": False, "uses_private_reference": "Evaluation labels and subgroup definitions after prediction commit only"}


def run(root):
    out = root / "medical-kv-a"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(out / "started.json", {"host": socket.gethostname(), "pid": os.getpid(), "device": "cpu"})
    start = time.monotonic()
    plan = read_json(out / "plan.json")
    for path, expected in plan["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen input changed")
    if file_hash(Path(__file__)) != plan["source_sha256"]:
        raise ValueError("Frozen cohort A source changed")
    prototypes = {"head": read_tensors(plan["head_prototypes"]),
                  "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    predictions = []
    for index, item in enumerate(read_json(out / "private/upload-list.json")):
        observation = load_observation(item["observation"], "cpu")
        validate_observation(observation, plan)
        predictions.append({"id": item["id"], "scores": score(observation.tensors, prototypes)})
        del observation
        write_json(out / "progress.json", {"completed": index + 1, "total": 20})
        print(f"Cohort A scoring {index+1}/20", flush=True)
    # The scorer has received only tensor uploads and the unchanged public prototypes.
    write_json(out / "predictions.json", predictions)
    write_json(out / "predictions-complete.json", {"sha256": file_hash(out / "predictions.json"),
        "primary": plan["primary"], "private_labels_read_for_evaluation": False,
        "target_text_or_lengths_used_by_classifier": False})
    # A prior Q replay is a consistency check, never a selection criterion.
    mapping = read_json(out / "private/upload-map.json")
    baseline = read_json("outputs/diagnostics/image_property_next_20261028/deep-probe/predictions.json")
    error = max(abs(a-b) for row in predictions for layer in (24, 32)
        for a, b in zip(row["scores"]["head"][f"L{layer}_q"],
                        baseline[mapping[row["id"]]]["scores"][f"visual_{layer}"], strict=True))
    if error > plan["replay_absolute_tolerance"]:
        raise ValueError("Original A Q distances not reproduced")
    report = evaluate(out, predictions)
    report["original_head_q_replay_max_error"] = error
    write_json(out / "report.json", report)
    write_json(out / "complete.json", {"status": "complete", "device": "cpu", "gpu_used": False,
        "seconds": time.monotonic()-start, "runtime_source": str(Path(__file__).resolve()),
        "report_sha256": file_hash(out / "report.json")})
    print({p: {m: v["correct"] for m, v in g["all"]["metrics"].items()}
           for p, g in report["properties"].items()}, flush=True)


def audit(root):
    """Recompute distances directly from SVD and independently check the metrics."""
    out = root / "medical-kv-a"
    plan = read_json(out / "plan.json")
    predictions = read_json(out / "predictions.json")
    complete, report = read_json(out / "complete.json"), read_json(out / "report.json")
    if (file_hash(out / "predictions.json") != read_json(out / "predictions-complete.json")["sha256"]
            or file_hash(out / "report.json") != complete["report_sha256"]):
        raise ValueError("Committed result changed")
    for path, expected in plan["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen input changed")
    for path, expected in read_json(root / "source-medical-kv-a-manifest.json")["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen source changed")
    if "source-medical-kv-a-v1" not in complete["runtime_source"]:
        raise ValueError("Expected frozen runtime")
    prototypes = {"head": read_tensors(plan["head_prototypes"]),
                  "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    saved = {row["id"]: row["scores"] for row in predictions}
    checked, max_error = 0, 0.
    for item in read_json(out / "private/upload-list.json"):
        observation = load_observation(item["observation"], "cpu")
        validate_observation(observation, plan)
        for method in METHODS:
            layer, letter = method.split("_")
            key = f"backend.language_model.model.layers.{int(layer[1:])-1}.self_attn.{letter}_proj.lora_A.default.weight"
            _, singular, vh = torch.linalg.svd(observation.tensors[key].double(), full_matrices=False)
            basis = vh[singular > max(float(singular.max())*1e-6, 1e-10)]
            for prop in CLASSES:
                p = prototypes[prop][f"visual_{layer[1:]}"]
                actual = (p - (p @ basis.T) @ basis).norm(dim=1).tolist()
                error = float(np.max(np.abs(np.asarray(actual)-saved[item["id"]][prop][method])))
                if error > plan["replay_absolute_tolerance"]:
                    raise ValueError("Independent A SVD scores differ")
                max_error, checked = max(max_error, error), checked + 1
        del observation
    recomputed = evaluate(out, predictions)
    if recomputed["properties"] != report["properties"]:
        raise ValueError("Metric or bootstrap recomputation differs")
    targets = read_json(HISTORICAL / "private/targets.json")
    a_ids = {row["row"]["image_id"] for row in targets}
    overlaps = {}
    for cohort, path in [("B", root / "private/targets.json"), ("C", root / "medical-kv/private/targets.json")]:
        overlaps[cohort] = len(a_ids & {r["row"]["image_id"] for r in read_json(path)})
    if any(overlaps.values()) or len(a_ids) != 20:
        raise ValueError("Unexpected cohort identity or overlap")
    result = {"status": "pass", "cohort": "A", "uploads": 20, "distances_recomputed": checked,
        "max_distance_error": max_error, "original_head_q_replay_max_error": report["original_head_q_replay_max_error"],
        "image_id_overlap_with_BC": overlaps, "same_model_as_BC": True,
        "target_text_or_lengths_used_by_classifier": False,
        "metrics_and_bootstrap_recomputed": True, "prediction_sha256": file_hash(out / "predictions.json"),
        "report_sha256": file_hash(out / "report.json"), "source_manifest_sha256": file_hash(root / "source-medical-kv-a-manifest.json")}
    write_json(out / "integrity-audit.json", result)
    destination = Path("docs/experiments/data/visual-concept-probe")
    write_json(out / "aggregate.json", {**report, "runtime": complete})
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Cohort", "Property", "Layer", "Module", "Primary", "Correct", "Images", "Accuracy", "Balanced accuracy", "Positive recall", "Negative recall", "Positive AUROC"])
    for prop, groups in report["properties"].items():
        for method, value in groups["all"]["metrics"].items():
            layer, module = method.split("_")
            writer.writerow(["A", prop, layer[1:], module, method == plan["primary"], value["correct"], value["n"],
                value["accuracy"], value["balanced_accuracy"], value["recall"][CLASSES[prop][0]],
                value["recall"][CLASSES[prop][1]], value["class_auroc"][CLASSES[prop][0]]])
    (out / "summary.csv").write_text(stream.getvalue())
    files = []
    for name, target in [("aggregate.json", "medical-kv-a-aggregate.json"),
                         ("integrity-audit.json", "medical-kv-a-audit.json"), ("summary.csv", "medical-kv-a-summary.csv")]:
        path = destination / target
        shutil.copy2(out / name, path)
        files.append(path)
    write_json(destination / "medical-kv-a-manifest.json", {"uncommitted": True,
        "uses_private_reference": "Post-prediction evaluation labels only; no private images or text published",
        "artifacts": {str(path): file_hash(path) for path in files},
        "source_manifest_sha256": result["source_manifest_sha256"]})
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        {"prepare": prepare, "run": run, "audit": audit}[args.stage](args.root.resolve())
    except BaseException:
        write_json(args.root / "medical-kv-a" / f"{args.stage}-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
