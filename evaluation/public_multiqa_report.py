"""Independent persisted-score audit and aggregate export for public multi-QA calibration."""

import argparse
import csv
import io
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.deep_property_probe import distance_to_gradient
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.multilayer_property import LAYERS
from evaluation.multilayer_property_gpu import CLASSES
from evaluation.public_multiqa_calibration import calibrated_scores, fit_rules


METHODS = ["original_24", "original_32", "original_fusion", "single_ridge", "multi_ridge"]


def audit_and_export(root):
    torch.set_num_threads(1)
    out = root / "multi-qa"
    plan = read_json(out / "plan.json")
    complete = read_json(out / "complete.json")
    if file_hash(out / "report.json") != complete["report_sha256"]:
        raise ValueError("Completed report changed")
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen input changed")
    for path, sha in read_json(root / "source-multi-qa-manifest.json")["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen source changed")
    public_commit = read_json(out / "public-scores-complete.json")
    for path, sha in public_commit["gradient_hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Public gradient changed")
    if file_hash(out / "public-scores.json") != public_commit["sha256"]:
        raise ValueError("Public score commit changed")
    bank, scores = read_json(plan["public_bank"]), read_json(out / "public-scores.json")
    proto = {"head": read_tensors(plan["head_prototypes"]),
             "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    public_checked, max_error = 0, 0.
    for row in scores:
        data = read_tensors(out / "gradients" / f"{row['image']:03d}-{row['qa']}.safetensors")
        for prop, recorded in row["scores"].items():
            for layer in LAYERS:
                actual = distance_to_gradient(proto[prop][f"visual_{layer}"], data[f"gradient_{layer}"])
                error = float(np.max(np.abs(np.asarray(actual) - recorded[str(layer)])))
                max_error = max(max_error, error)
                if error > 1e-12:
                    raise ValueError("Public score independent recomputation differs")
                public_checked += 1
    public = read_json(out / "public-audit.json")
    deployment = read_json(out / "frozen-rules.json")
    if fit_rules(scores, bank, plan, "fit") != public["rules"] or fit_rules(scores, bank, plan) != deployment:
        raise ValueError("Public calibration rules could not be reproduced")
    original = {"head": read_json(root / "head-rule.json"), "abdomen": read_json(root / "public/abdomen-rule.json")}
    for prop, group in public["properties"].items():
        expected = [row for row in scores if prop in row["scores"] and bank[row["image"]]["calibration_split"] == "audit"]
        if len(expected) != len(group["predictions"]):
            raise ValueError("Public audit rows missing")
        for row, recorded in zip(expected, group["predictions"], strict=True):
            actual = calibrated_scores(row["scores"][prop], public["rules"][prop], original[prop])
            if actual != recorded["scores"]:
                raise ValueError("Held-out public prediction changed")
    # Medical truth is used only for independent evaluation of committed scores.
    predictions = read_json(out / "predictions.json")
    if file_hash(out / "predictions.json") != read_json(out / "predictions-complete.json")["sha256"]:
        raise ValueError("Target predictions changed")
    mapping = read_json(root / "private/upload-map.json")
    target_map = {row["run"]: row for row in read_json(root / "private/targets.json")}
    truth = [target_map[mapping[row["id"]]] for row in predictions]
    target_report = read_json(out / "report.json")
    metrics_checked, paired = 0, {}
    for prop, classes in CLASSES.items():
        labels = [classes[0] if row["region"] == prop else classes[1] for row in truth]
        groups = {"all": list(range(len(truth))),
                  "no_literal_property_terms": [i for i, row in enumerate(truth) if not row["literal_terms"][prop]],
                  **{m: [i for i, row in enumerate(truth) if row["modality"] == m] for m in ("CT", "MRI", "X-Ray")}}
        for name, ids in groups.items():
            for method, recorded in target_report["properties"][prop][name]["metrics"].items():
                actual = classification_stats([labels[i] for i in ids], [predictions[i]["scores"][prop][method] for i in ids], classes)
                if any(recorded[key] != value for key, value in actual.items()):
                    raise ValueError("Target metric independent recomputation differs")
                metrics_checked += 1
        paired[prop] = {}
        primary = [row["scores"][prop]["multi_ridge"] for row in predictions]
        for method in ("single_ridge", "original_24", "original_32", "original_fusion"):
            baseline = [row["scores"][prop][method] for row in predictions]
            a = classification_stats(labels, primary, classes)
            b = classification_stats(labels, baseline, classes)
            paired[prop][method] = {"accuracy_gain": a["accuracy"] - b["accuracy"],
                "balanced_accuracy_gain": a["balanced_accuracy"] - b["balanced_accuracy"],
                **paired_bootstrap(labels, primary, classes, baseline)}
    aggregate = {**target_report, "paired_multi_ridge_gains": paired, "public_rules": deployment,
                 "public_audit": {"note": public["note"], "properties": {
                     prop: {key: value for key, value in group.items() if key != "predictions"}
                     for prop, group in public["properties"].items()}},
                 "public_split_counts": {prop: {split: sum(prop in row["labels"] and row["calibration_split"] == split for row in bank)
                                       for split in ("fit", "audit")} for prop in CLASSES},
                 "runtime": complete}
    write_json(out / "aggregate.json", aggregate)
    write_json(out / "integrity-audit.json", {"status": "pass", "public_gradient_files": len(public_commit["gradient_hashes"]),
               "public_layer_distances_recomputed": public_checked, "public_distance_max_error": max_error,
               "public_rule_recompute": "exact", "held_out_public_prediction_recompute": "exact",
               "target_metric_groups_recomputed": metrics_checked, "target_predictions": len(predictions),
               "target_predictions_sha256": file_hash(out / "predictions.json"),
               "rules_sha256": file_hash(out / "frozen-rules.json"), "plan_sha256": file_hash(out / "plan.json"),
               "source_manifest_sha256": file_hash(root / "source-multi-qa-manifest.json"),
               "report_sha256": file_hash(out / "report.json"), "analysis_source_sha256": file_hash(Path(__file__))})
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Property", "Method", "Correct", "Accuracy", "Balanced accuracy", "Recall", "AUROC"])
    for prop, classes in CLASSES.items():
        for method, value in aggregate["properties"][prop]["all"]["metrics"].items():
            writer.writerow([prop, method, value["correct"], value["accuracy"], value["balanced_accuracy"],
                             value["recall"][classes[0]], value["class_auroc"][classes[0]]])
    (out / "summary.csv").write_text(stream.getvalue())
    destination = Path("docs/experiments/data/visual-concept-probe")
    files = []
    for source, name in [(out / "aggregate.json", "multi-qa-aggregate.json"),
                         (out / "integrity-audit.json", "multi-qa-audit.json"),
                         (out / "summary.csv", "multi-qa-summary.csv")]:
        target = destination / name
        shutil.copy2(source, target)
        files.append(target)
    figure = plot(aggregate)
    files.append(figure)
    source_copy = out / "analysis-source.py"
    shutil.copy2(Path(__file__), source_copy)
    files.append(source_copy)
    write_json(destination / "multi-qa-manifest.json", {"uncommitted": True,
               "base_commit": read_json(root / "source-multi-qa-manifest.json")["base_commit"],
               "uses_private_reference": "Post-prediction property labels for metrics only",
               "artifacts": {str(path.resolve().relative_to(Path.cwd())): file_hash(path) for path in files}})
    print({"status": "pass", "public_layer_distances": public_checked, "target_metric_groups": metrics_checked})


def plot(aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    labels = ["Layer 24", "Layer 32", "Original fusion", "Single-QA ridge", "Multi-QA ridge"]
    colors = ["#9a9a9a", "#717171", "#c6c6c6", "#4c78a8", "#e58939"]
    for i, prop in enumerate(CLASSES):
        public = aggregate["public_audit"]["properties"][prop]["robustness"]
        private = aggregate["properties"][prop]["all"]["metrics"]
        values = [[public[method]["held_out_qa_balanced_accuracy"] for method in METHODS],
                  [private[method]["balanced_accuracy"] for method in METHODS]]
        for j, title in enumerate(("Public: held-out content and QA", "Medical: original private natural QA")):
            axis = axes[i, j]
            axis.bar(range(len(METHODS)), values[j], color=colors)
            axis.axhline(.5, color="#888888", linestyle="--", linewidth=1)
            axis.set_title(f"{prop.capitalize()} | {title}", fontsize=11)
            axis.set_xticks(range(len(METHODS)), labels, rotation=20, ha="right", fontsize=8)
            axis.set_ylim(0, 1.06)
            axis.grid(axis="y", alpha=.15)
            axis.set_axisbelow(True)
            for k, value in enumerate(values[j]):
                axis.text(k, value + .015, f"{value:.2f}", ha="center", fontsize=9)
            if j == 0:
                axis.set_ylabel("Balanced accuracy")
    figure.suptitle("Public multi-QA calibration: unchanged prototypes and medical uploads", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .96))
    path = Path("docs/experiments/figures/visual-concept-probe/multi-qa-calibration.png")
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
        write_json(args.root / "multi-qa/analysis-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
