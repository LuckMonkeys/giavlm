"""Aggregate bank repeats without treating repeated targets as independent images."""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.artifacts import file_hash, read_json, write_json
from evaluation.generic_head_property import ROOT, SOURCE, verify_hashes
from evaluation.image_property_scoring import METHODS


def clustered_interval(credits, labels, other=None, repeats=5000):
    """Average fixed bank repeats within image, then stratified target bootstrap."""
    values = np.asarray(credits, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(labels):
        raise ValueError("Expected bank by target credits")
    per_image = values.mean(0)
    if other is not None:
        reference = np.asarray(other, dtype=float)
        if reference.ndim == 2:
            reference = reference.mean(0)
        if reference.shape != per_image.shape:
            raise ValueError("Paired target count differs")
        per_image = per_image - reference
    y = np.asarray(labels)
    groups = [np.flatnonzero(y == c) for c in sorted(set(labels))]
    rng = np.random.default_rng(20261018)
    samples = [per_image[rng.choice(g, (repeats, len(g)), replace=True)].mean(1) for g in groups]
    metric = np.mean(samples, axis=0)
    return {"mean_balanced_credit": float(np.mean([per_image[g].mean() for g in groups])),
            "ci95": np.quantile(metric, [.025, .975]).tolist(),
            "independent_images": len(labels), "fixed_banks": values.shape[0],
            "bootstrap": "Stratified target bootstrap, banks averaged within target; not population bank uncertainty"}


def summarize():
    complete = read_json(ROOT / "complete.json")
    if file_hash(ROOT / "report.json") != complete["report_sha256"]:
        raise ValueError("Report changed")
    verify_hashes(read_json(ROOT / "plan.json")["frozen_hashes"])
    verify_hashes(read_json(ROOT / "predictions_complete.json")["hashes"])
    report = read_json(ROOT / "report.json")
    labels = [r["label"] for r in read_json(SOURCE / "private/targets.json")]
    classes = ["head", "non-head"]
    methods = [*METHODS, "model_only", "text_only"]
    summary = {"source_report_sha256": file_hash(ROOT / "report.json"), "targets": 20,
               "banks": 3, "phases": {}, "runtime": complete}
    for phase in ["reconstructed", "oracle"]:
        rows, credits = [], {m: [] for m in methods}
        for b in range(3):
            results = report["banks"][str(b)][phase]
            rows.append({"bank": b, "metrics": results["metrics"],
                         "wrong_gradient_accuracy": results["wrong_gradient"]["centered"]["accuracy"],
                         "matched_wrong_gradient": results["matched_wrong_gradient"],
                         "groups": results["groups"]})
            preds = [read_json(ROOT / f"bank-{b}" / phase / "predictions" / f"{i:05d}-{phase}.json") for i in range(20)]
            for m in methods:
                scores = [p["class_scores"][m] if m in METHODS else p[m] for p in preds]
                credits[m].append([float(s is not None and classes[int(np.argmin(s))] == y)
                                   for s, y in zip(scores, labels, strict=True)])
        old = [read_json(SOURCE / phase / "predictions" / f"{i:05d}-{phase}.json") for i in range(20)]
        old_credits = [float(classes[int(np.argmin(p["class_scores"]["centered"]))] == y)
                       for p, y in zip(old, labels, strict=True)]
        means = {}
        for m in methods:
            accuracies = [r["metrics"][m]["accuracy"] for r in rows]
            aucs = [r["metrics"][m]["class_auroc"]["head"] for r in rows]
            means[m] = {"accuracy_mean": float(np.mean(accuracies)),
                        "accuracy_min": min(accuracies), "accuracy_max": max(accuracies),
                        "head_auc_mean": float(np.mean(aucs)),
                        "head_recall_mean": float(np.mean([r["metrics"][m]["recall"]["head"] for r in rows])),
                        "target_interval": clustered_interval(credits[m], labels)}
        summary["phases"][phase] = {"per_bank": rows, "means": means,
            "medical": report["baseline"][phase]["metrics"],
            "paired_vs_medical": clustered_interval(credits["centered"], labels, old_credits),
            "paired_vs_model_only": clustered_interval(credits["centered"], labels, credits["model_only"]),
            "wrong_gradient_mean": float(np.mean([r["wrong_gradient_accuracy"] for r in rows]))}
    write_json(ROOT / "aggregate.json", summary)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for ax, phase in zip(axes, ["reconstructed", "oracle"], strict=True):
        result = summary["phases"][phase]
        x = np.arange(4)
        for k, (m, label, color) in enumerate([
            ("centered", "Gradient (primary)", "#2372b5"),
            ("model_only", "Model-only", "#b88a73"),
        ]):
            values = [result["medical"][m]["accuracy"]] + [r["metrics"][m]["accuracy"] for r in result["per_bank"]]
            ax.bar(x+(k-.5)*.32, values, width=.32, label=label, color=color)
            for j, v in enumerate(values):
                ax.text(j+(k-.5)*.32, v+.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x, ["Medical", "Generic 1", "Generic 2", "Generic 3"])
        ax.set_ylabel("Accuracy (same 20 images)")
        ax.set_ylim(0, 1.16)
        ax.set_title("Reconstructed text" if phase == "reconstructed" else "Correct text (diagnostic)")
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(figures / "generic-bank-accuracy.png", dpi=180)
    plt.close(fig)
    dest = Path("docs/experiments/figures/synthetic-head-property")
    shutil.copyfile(figures / "generic-bank-accuracy.png", dest / "generic-bank-accuracy.png")
    destination = Path("docs/experiments/data/synthetic-head-property/generic-bank-aggregate.json")
    write_json(destination, summary)
    write_json(ROOT / "report_complete.json", {"aggregate_sha256": file_hash(ROOT / "aggregate.json"),
               "figure_sha256": file_hash(figures / "generic-bank-accuracy.png")})
    for phase, result in summary["phases"].items():
        print(phase, {m: r["accuracy_mean"] for m, r in result["means"].items()}, flush=True)


if __name__ == "__main__":
    try:
        summarize()
    except BaseException as error:
        write_json(ROOT / "report_failed.json", {"type": type(error).__name__, "reason": str(error)})
        raise
