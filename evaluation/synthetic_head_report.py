"""Post-commit audit and subgroup evaluation of the 20-image head experiment."""

from collections import Counter
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.artifacts import file_hash, read_json, write_json
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.synthetic_head_property import ROOT


def binary_details(labels, scores, classes):
    """Positive=head; undefined precision and AUC remain undefined."""
    stats = classification_stats(labels, scores, classes)
    if not labels:
        return stats
    positive = classes.index("head")
    y = np.array([label == "head" for label in labels])
    pred = np.array([np.argmin(s) if s is not None else -1 for s in scores])
    tp = int((y & (pred == positive)).sum())
    fp = int((~y & (pred == positive)).sum())
    stats.update(precision=tp / (tp + fp) if tp + fp else None,
                 coverage=float((pred >= 0).mean()),
                 **paired_bootstrap(labels, scores, classes))
    n = len(labels)
    p = stats["accuracy"]
    z = 1.959963984540054
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    stats["accuracy_wilson_ci95"] = [float(center-half), float(center+half)]
    stats["ci_caveat"] = "Image-level intervals; patient independence unknown"
    return stats


def summarize():
    complete = read_json(ROOT / "complete.json")
    plan = read_json(ROOT / "plan.json")
    report = read_json(ROOT / "report.json")
    if file_hash(ROOT / "report.json") != complete["report_sha256"]:
        raise ValueError("Completed report changed")
    for path, expected in plan["frozen_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen input changed")
    for marker in ["capture_complete.json", "reconstruction/complete.json", "oracle/complete.json",
                   "reconstructed/complete.json"]:
        for path, expected in read_json(ROOT / marker)["hashes"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("Committed phase artifact changed")
    truth = read_json(ROOT / "private/targets.json")
    labels = [r["label"] for r in truth]
    classes = plan["classes"]
    n = len(truth)
    if n != 20:
        raise ValueError("Expected all twenty targets")
    for i in range(n):
        private = read_json(ROOT / "captures" / f"run-{i:05d}" / "private/observation.json")
        if any(private[key] for key in ["public_questions", "public_targets", "public_question_ids", "public_target_ids"]):
            raise ValueError("Reference text exposed in private observation")
    analysis = {"source_report_sha256": file_hash(ROOT / "report.json"), "phases": {},
                "text_recovery": report["text_recovery"], "data": read_json(ROOT / "data_audit.json")}
    text_em = report["text_recovery"]["per_target"]
    for phase in ["reconstructed", "oracle"]:
        predictions = [read_json(ROOT / phase / "predictions" / f"{i:05d}-{phase}.json") for i in range(n)]
        for p in predictions:
            if "scores_file" in p and file_hash(ROOT / phase / p["scores_file"]) != p["scores_sha256"]:
                raise ValueError("Prediction score source changed")
        scores = {m: [p["class_scores"][m] for p in predictions]
                  for m in ["centered", "global", "equal_tensor", "centered_equal_tensor"]}
        scores.update({m: [p[m] for p in predictions] for m in ["model_only", "text_only"]})
        groups = {"all": list(range(n))}
        for modality in ["CT", "MRI", "X-Ray"]:
            groups[modality] = [i for i, row in enumerate(truth) if row["modality"] == modality]
        for flag in [False, True]:
            groups[f"head_terms_{flag}"] = [i for i, row in enumerate(truth) if row["head_terms_present"] == flag]
            groups[f"joint_em_{flag}"] = [i for i, row in enumerate(text_em) if row["joint_em"] == flag]
        subgroup = {name: {m: binary_details([labels[i] for i in ids], [values[i] for i in ids], classes)
                          for m, values in scores.items()} for name, ids in groups.items()}
        wrong = {}
        for kind in ["all", "same_modality", "same_modality_opposite_property", "opposite_property"]:
            per_target, pairs = [], 0
            for i, pred in enumerate(predictions):
                credits = []
                for j, values in zip(pred["shuffled_target_runs"], pred["shuffled_class_scores"], strict=True):
                    same_modality = truth[i]["modality"] == truth[j]["modality"]
                    opposite = labels[i] != labels[j]
                    if ((kind.startswith("same_modality") and not same_modality)
                            or ("opposite_property" in kind and not opposite)):
                        continue
                    score = values["centered"]
                    credits.append(float(score is not None and classes[int(np.argmin(score))] == labels[i]))
                if credits:
                    per_target.append({"run": i, "accuracy": float(np.mean(credits)), "pairs": len(credits)})
                    pairs += len(credits)
            wrong[kind] = {"accuracy": float(np.mean([r["accuracy"] for r in per_target])) if per_target else None,
                           "targets": len(per_target), "pairs": pairs, "per_target": per_target}
        rows = [{**r, "modality": truth[i]["modality"], "head_terms_present": truth[i]["head_terms_present"],
                 "joint_em": text_em[i]["joint_em"]} for i, r in enumerate(report["phases"][phase]["per_target"])]
        analysis["phases"][phase] = {"groups": subgroup, "wrong_gradient": wrong,
            "paired_gain_ci95": report["phases"][phase]["paired_gain_ci95"], "per_target": rows}
    write_json(ROOT / "analysis.json", analysis)
    write_json(ROOT / "integrity_audit.json", {
        "status": "pass", "frozen_inputs_match": True, "targets": n, "generated_candidates": 24,
        "predictions": 2*n, "reconstruction_status": dict(Counter(r["status"] for r in text_em)),
        "max_exact_replay_relative_l2": read_json(ROOT / "capture_complete.json")["max_relative_l2"],
        "model_unchanged": complete["model_unchanged"], "private_observations_expose_text": False,
        "analysis_sha256": file_hash(ROOT / "analysis.json"), "report_sha256": file_hash(ROOT / "report.json"),
    })
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    names = ["Gradient\n(primary)", "Model-only", "Text-only", "Wrong\ngradient"]
    for ax, phase in zip(axes, ["reconstructed", "oracle"], strict=True):
        result = analysis["phases"][phase]
        values = [result["groups"]["all"][m]["accuracy"] for m in ["centered", "model_only", "text_only"]]
        values.append(result["wrong_gradient"]["all"]["accuracy"])
        ax.bar(names, values, color=["#2372b5", "#8da8b8", "#b3bdc6", "#b88a73"])
        for j, v in enumerate(values):
            ax.text(j, v + .025, f"{v:.2f}", ha="center")
        ax.set_ylim(0, 1.12)
        ax.axhline(.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_ylabel("Accuracy (20 images)")
        ax.set_title("Reconstructed text" if phase == "reconstructed" else "Correct text (diagnostic)")
    fig.savefig(figures / "accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, phase in zip(axes, ["reconstructed", "oracle"], strict=True):
        result = analysis["phases"][phase]
        for label, color in [("head", "#2372b5"), ("non-head", "#ca7b43")]:
            rows = [r for r in result["per_target"] if r["label"] == label and r["primary_scores"] is not None]
            x = [r["run"] + 1 for r in rows]
            y = [r["primary_scores"][1] - r["primary_scores"][0] for r in rows]
            ax.scatter(x, y, label=label, color=color)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(range(1, 21))
        ax.set_xlabel("Anonymous target index")
        ax.set_ylabel("Non-head loss minus head loss")
        ax.set_title("Reconstructed text" if phase == "reconstructed" else "Correct text")
        ax.legend()
    fig.savefig(figures / "margins.png", dpi=180)
    plt.close(fig)
    destination = Path("docs/experiments/figures/synthetic-head-property")
    destination.mkdir(parents=True, exist_ok=True)
    for path in figures.glob("*.png"):
        shutil.copyfile(path, destination / path.name)
    write_json(figures / "manifest.json", {"analysis_sha256": file_hash(ROOT / "analysis.json"),
               "files": {p.name: file_hash(p) for p in figures.glob("*.png")}})
    print("Audit and aggregate report complete", flush=True)


if __name__ == "__main__":
    try:
        summarize()
    except BaseException as error:
        write_json(ROOT / "report_failed.json", {"type": type(error).__name__, "reason": str(error)})
        raise
