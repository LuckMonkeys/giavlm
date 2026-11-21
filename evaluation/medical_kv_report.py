"""Independent medical QKV transfer audit, matched statistics and aggregate export."""

import argparse
import csv
import io
from pathlib import Path
import shutil
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, write_json
from core.data import read_manifest
from evaluation.image_property_data import DATA, pixel_hash
from evaluation.image_property_scoring import classification_stats
from evaluation.medical_kv_data import select_public_module
from evaluation.medical_kv_transfer import METHODS, answer_type
from evaluation.multilayer_property_gpu import CLASSES


def audit_and_export(root):
    from core.fl import load_observation
    from core.artifacts import read_tensors
    from evaluation.medical_kv_transfer import score
    torch.set_num_threads(1)
    out = root / "medical-kv"
    plan, complete = read_json(out / "plan.json"), read_json(out / "complete.json")
    if file_hash(out / "report.json") != complete["report_sha256"]:
        raise ValueError("Worker report changed")
    if file_hash(out / "predictions.json") != read_json(out / "predictions-complete.json")["sha256"]:
        raise ValueError("Medical prediction commit changed")
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen input changed")
    for path, sha in read_json(root / "source-medical-kv-manifest.json")["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen source changed")
    for path, sha in read_json(out / "new/capture-complete.json")["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("New anonymous upload changed")
    public_rule = select_public_module(read_json("docs/experiments/data/visual-concept-probe/qa-controls-aggregate.json")["robustness"])
    if public_rule["primary"] != plan["primary"] or public_rule["candidates"] != read_json(out / "public-rule.json")["candidates"]:
        raise ValueError("Public-only selection rule differs")
    if "source-medical-kv-v1" not in complete["runtime_source"]:
        raise ValueError("Actual execution was not frozen source")
    predictions, report = read_json(out / "predictions.json"), read_json(out / "report.json")
    prototypes = {"head": read_tensors(plan["head_prototypes"]), "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    checked, max_error = 0, 0.
    for cohort, path in [("historical", root / "fresh/upload-list.json"), ("new", out / "new/upload-list.json")]:
        by_id = {row["id"]: row for row in predictions[cohort]}
        for item in read_json(path):
            observation = load_observation(item["observation"], "cpu")
            if (observation.training.knowledge != "private" or observation.training.token_lengths_known or
                    observation.public_questions or observation.public_targets or observation.public_images is not None):
                raise ValueError("Reference information in scoring inputs")
            actual = score(observation.tensors, prototypes)
            for prop in CLASSES:
                for method in METHODS:
                    error = float(np.max(np.abs(np.asarray(actual[prop][method]) - by_id[item["id"]]["scores"][prop][method])))
                    max_error = max(max_error, error)
                    if error > 1e-12:
                        raise ValueError("Independent medical distance recomputation differs")
                    checked += 1
    metric_groups = 0
    for cohort, target_path, mapping_path in [("historical", root / "private/targets.json", root / "private/upload-map.json"),
                                               ("new", out / "private/targets.json", out / "private/upload-map.json")]:
        targets = {row["run"]: row for row in read_json(target_path)}
        mapping = read_json(mapping_path)
        truth = [targets[mapping[row["id"]]] for row in predictions[cohort]]
        for prop, classes in CLASSES.items():
            groups = {"all": list(range(len(truth))),
                "no_literal_property_terms": [i for i, row in enumerate(truth) if not row["literal_terms"][prop]],
                **{name: [i for i, row in enumerate(truth) if row["modality"] == name] for name in ("CT", "MRI", "X-Ray")},
                **{name: [i for i, row in enumerate(truth) if answer_type(row) == name] for name in ("yes_no", "other")}}
            for name, ids in groups.items():
                labels = [classes[0] if truth[i]["region"] == prop else classes[1] for i in ids]
                for method in METHODS:
                    actual = classification_stats(labels, [predictions[cohort][i]["scores"][prop][method] for i in ids], classes)
                    recorded = report["properties"][cohort][prop][name]["metrics"][method]
                    if any(value != recorded[key] for key, value in actual.items()):
                        raise ValueError("Medical metric recomputation differs")
                    metric_groups += 1
    # Evaluator-only image-ID and decoded-pixel audit of every previous property target.
    sampling = read_json(out / "private/sampling.json")
    targets = read_json(out / "private/targets.json")
    data = read_manifest(DATA)
    local = {row["image_id"]: row for row in data}
    previous_ids = set(sampling["excluded_ids"])
    previous_pixels = {pixel_hash(Image.open(local[key]["image"])) for key in previous_ids}
    current_ids = {row["row"]["image_id"] for row in targets}
    current_pixels = {pixel_hash(Image.open(row["row"]["image"])) for row in targets}
    if previous_ids & current_ids or previous_pixels & current_pixels or len(current_pixels) != 20:
        raise ValueError("New cohort overlap detected; never resample after scores")
    if {row["row"]["split"] for row in targets} != {"eval"}:
        raise ValueError("Silent target split change")
    audit = {"status": "pass", "uploads_recomputed": 40, "method_distances_recomputed": checked,
        "max_distance_error": max_error, "metric_groups_recomputed": metric_groups,
        "primary_public_rule_reproduced": public_rule["primary"], "new_images": 20,
        "all_previous_image_ids_excluded": len(previous_ids), "image_id_overlap": 0, "pixel_overlap": 0,
        "new_cohort_split": "eval", "private_images_or_text_used_for_inference": False,
        "prediction_sha256": file_hash(out / "predictions.json"), "report_sha256": file_hash(out / "report.json"),
        "source_manifest_sha256": file_hash(root / "source-medical-kv-manifest.json"),
        "analysis_source_sha256": file_hash(Path(__file__))}
    write_json(out / "integrity-audit.json", audit)
    aggregate = {**report, "runtime": complete, "public_prototype_scope": "Existing nonmedical synthetic fit image prototypes; no real medical auxiliary adaptation",
                 "uses_private_reference": "Post-commit property labels, modality and QA category for evaluation only"}
    write_json(out / "aggregate.json", aggregate)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Cohort", "Property", "Layer", "Module", "Primary", "Correct", "Images", "Accuracy", "Balanced accuracy", "Positive recall", "Negative recall", "Positive AUROC"])
    for cohort, props in report["properties"].items():
        for prop, groups in props.items():
            for method, value in groups["all"]["metrics"].items():
                layer, module = method.split("_")
                writer.writerow([cohort, prop, layer[1:], module, method == plan["primary"], value["correct"], value["n"],
                                 value["accuracy"], value["balanced_accuracy"], value["recall"][CLASSES[prop][0]],
                                 value["recall"][CLASSES[prop][1]], value["class_auroc"][CLASSES[prop][0]]])
    (out / "summary.csv").write_text(stream.getvalue())
    destination = Path("docs/experiments/data/visual-concept-probe")
    files = []
    for source, name in [(out / "aggregate.json", "medical-kv-aggregate.json"), (out / "integrity-audit.json", "medical-kv-audit.json"),
                         (out / "summary.csv", "medical-kv-summary.csv")]:
        path = destination / name
        shutil.copy2(source, path)
        files.append(path)
    figure = plot(aggregate)
    files.append(figure)
    source_copy = out / "analysis-source.py"
    shutil.copy2(Path(__file__), source_copy)
    files.append(source_copy)
    write_json(destination / "medical-kv-manifest.json", {"uncommitted": True,
        "uses_private_reference": "Post-prediction property labels and grouping only",
        "artifacts": {str(path.resolve().relative_to(Path.cwd())): file_hash(path) for path in files}})
    print(audit)


def plot(aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharey=True)
    labels = ["24 Q", "24 K", "24 V*", "32 Q", "32 K", "32 V"]
    colors = ["#b0b0b0", "#4c78a8", "#f28e2b", "#b0b0b0", "#4c78a8", "#f28e2b"]
    for i, cohort in enumerate(("historical", "new")):
        for j, prop in enumerate(CLASSES):
            axis = axes[i,j]
            group = aggregate["properties"][cohort][prop]["all"]["metrics"]
            values = [group[method]["balanced_accuracy"] for method in METHODS]
            axis.bar(range(6), values, color=colors)
            axis.axhline(.5, color="#888888", linestyle="--", linewidth=1)
            positive = aggregate["composition"][cohort]["region"][prop]
            axis.set_title(f"{cohort.capitalize()} | {prop.capitalize()} ({positive} positive, {20-positive} negative)")
            axis.set_xticks(range(6), labels)
            axis.set_ylim(0, 1.09)
            axis.grid(axis="y", alpha=.15)
            axis.set_axisbelow(True)
            if j == 0:
                axis.set_ylabel("Balanced accuracy")
            for k, value in enumerate(values):
                axis.text(k, value + .018, f"{value:.2f}", ha="center", fontsize=9)
    figure.suptitle("Medical natural-QA transfer | unchanged public prototypes | *public-selected primary", fontsize=12)
    figure.tight_layout(rect=(0,0,1,.95))
    path = Path("docs/experiments/figures/visual-concept-probe/medical-kv-transfer.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit_and_export(args.root.resolve())
    except BaseException:
        write_json(args.root / "medical-kv/analysis-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
