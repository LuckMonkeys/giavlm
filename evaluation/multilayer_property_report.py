"""Post-commit paired analysis and public aggregate publication for region probes."""

import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from core.artifacts import file_hash, read_json, write_json
from evaluation.image_property_scoring import paired_bootstrap
from evaluation.multilayer_property_gpu import CLASSES


METHODS = ["single_selected", "single_32", "majority", "mean_raw", "mean_standardized"]


def analyze(root):
    predictions_path = root / "fresh/predictions.json"
    if file_hash(predictions_path) != read_json(root / "fresh/predictions-complete.json")["sha256"]:
        raise ValueError("Prediction commit hash mismatch")
    done = read_json(root / "fresh/complete.json")
    if file_hash(root / "fresh/report.json") != done["report_sha256"]:
        raise ValueError("Worker report changed")
    report = read_json(root / "fresh/report.json")
    predictions = read_json(predictions_path)
    mapping = read_json(root / "private/upload-map.json")
    targets = {row["run"]: row for row in read_json(root / "private/targets.json")}
    truth = [targets[mapping[row["id"]]] for row in predictions]
    analysis = {}
    for name, classes in CLASSES.items():
        labels = [classes[0] if row["region"] == name else classes[1] for row in truth]
        score = {method: [row["scores"][name][method] for row in predictions] for method in METHODS}
        y = np.asarray(labels) == classes[0]
        correct = {method: np.asarray(values).argmin(1) == np.where(y, 0, 1)
                   for method, values in score.items()}
        comparisons = {}
        for baseline in ["single_selected", "single_32"]:
            differences = {method: {"accuracy_gain": float((correct[method].astype(float) -
                            correct[baseline].astype(float)).mean()),
                            "balanced_accuracy_gain": float(report["properties"][name]["all"][method]["balanced_accuracy"] -
                             report["properties"][name]["all"][baseline]["balanced_accuracy"]),
                            **paired_bootstrap(labels, values, classes, score[baseline])}
                           for method, values in score.items() if method != baseline}
            comparisons[baseline] = differences
        analysis[name] = {"paired_gains": comparisons,
                          "constant_negative_accuracy": float((~y).mean()),
                          "constant_negative_balanced_accuracy": .5,
                          "unit": "Image; patient dependence unknown"}
    public = {**report, "analysis": analysis,
              "cost": {**done, "generation_seconds": read_json(root / "generation/complete.json")["seconds"]},
              "public_review": {key: value for key, value in read_json(root / "public/review.json").items()
                                if key not in ["retained_ids", "rejected_ids", "reasons"]},
              "analysis_source_sha256": file_hash(Path(__file__)),
              "worker_report_sha256": file_hash(root / "fresh/report.json")}
    write_json(root / "fresh/analysis.json", public)
    write_json("docs/experiments/data/visual-concept-probe/fresh-region-aggregate.json", public)
    lines = ["# 新队列头部与腹部属性", "", "| Property | Method | Correct | Balanced accuracy | Recall | AUROC |",
             "|---|---|---:|---:|---:|---:|"]
    for name, classes in CLASSES.items():
        for method in METHODS:
            value = report["properties"][name]["all"][method]
            lines.append(f"| {name} | {method} | {value['correct']}/20 | {value['balanced_accuracy']:.3f} | "
                         f"{value['recall'][classes[0]]:.3f} | {value['class_auroc'][classes[0]]:.3f} |")
    lines += ["", "图像 ID 与旧属性队列隔离，但模态与部位混淆，患者关系未知；头部五正例、腹部十正例。",
              "主方法 mean_standardized 及零阈值在新目标评分前冻结；不在新目标上选层或权重。"]
    (root / "fresh/report-draft.md").write_text("\n".join(lines) + "\n")
    write_json(root / "fresh/analysis-complete.json", {"sha256": file_hash(root / "fresh/analysis.json")})


def analyze_fixed(root):
    """Only aggregate results enter this plot/export; original QA never enters it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = root / "fixed-qa"
    report = read_json(out / "report.json")
    if file_hash(out / "report.json") != read_json(out / "complete.json")["report_sha256"]:
        raise ValueError("Fixed-QA report changed")
    methods = ["single_selected", "single_32", "mean_standardized"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for axis, prop in zip(axes, ["head", "abdomen"], strict=True):
        group = report["properties"][prop]["all"]["metrics"]
        for index, (condition, color) in enumerate([("natural", "#4c78a8"), ("fixed", "#f58518")]):
            values = [group[condition][method]["balanced_accuracy"] for method in methods]
            intervals = np.asarray([group[condition][method]["supported_class_ba_ci95"] for method in methods])
            error = np.maximum(0, np.stack([np.asarray(values) - intervals[:, 0], intervals[:, 1] - values]))
            positions = np.arange(3) + (index - .5) * .34
            axis.bar(positions, values, .32, color=color, label=condition.capitalize() + " QA",
                     yerr=error, capsize=3, error_kw={"linewidth": 1})
            for position, value in zip(positions, values, strict=True):
                axis.text(position, value + .018, f"{value:.2f}", ha="center", fontsize=9)
        positive = 5 if prop == "head" else 10
        axis.set_title(f"{prop.capitalize()} ({positive} positive, {20-positive} negative)")
        axis.set_xticks(range(3), ["Layer 24", "Layer 32", "Layer fusion"])
        axis.axhline(.5, color="#777777", linestyle="--", linewidth=1)
        axis.set_ylim(0, 1.12)
        axis.set_yticks(np.arange(0, 1.01, .2))
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(loc="upper left", fontsize=9)
    figure.suptitle("Matched fixed-QA diagnostic: same 20 medical images", y=.99)
    figure.tight_layout(rect=(0, 0, 1, .94))
    figure.savefig(out / "comparison.png", dpi=200, bbox_inches=None)
    plt.close(figure)
    with (out / "summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Property", "Text condition", "Method", "Correct", "Accuracy", "Balanced accuracy", "Positive recall", "AUROC"])
        for prop in ["head", "abdomen"]:
            for condition in ["natural", "fixed"]:
                for method in METHODS:
                    value = report["properties"][prop]["all"]["metrics"][condition][method]
                    writer.writerow([prop, condition, method, value["correct"], value["accuracy"],
                                     value["balanced_accuracy"], value["recall"][prop], value["class_auroc"][prop]])
    public = Path("docs/experiments/data/visual-concept-probe")
    figure_path = Path("docs/experiments/figures/visual-concept-probe/fixed-qa-region.png")
    shutil.copy2(out / "comparison.png", figure_path)
    shutil.copy2(out / "summary.csv", public / "fixed-qa-region-summary.csv")
    manifest = {"report_sha256": file_hash(out / "report.json"),
                "public_report_sha256": file_hash(public / "fixed-qa-region-aggregate.json"),
                "source_sha256": file_hash(Path(__file__)),
                "figure_sha256": file_hash(figure_path), "summary_sha256": file_hash(public / "fixed-qa-region-summary.csv"),
                "scope": report["scope"]}
    write_json(out / "analysis-complete.json", manifest)
    write_json(public / "fixed-qa-region-manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixed-qa", action="store_true")
    args = parser.parse_args()
    (analyze_fixed if args.fixed_qa else analyze)(args.root.resolve())


if __name__ == "__main__":
    main()
