"""Audit completed synthetic CT predictions and plot aggregate results only."""

from collections import defaultdict
import hashlib
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.artifacts import file_hash, read_json, write_json
from evaluation.image_property_scoring import classification_stats
from evaluation.synthetic_ct_property import ROOT


def summarize():
    complete = read_json(ROOT / "complete.json")
    plan = read_json(ROOT / "plan.json")
    if file_hash(ROOT / "report.json") != complete["report_sha256"]:
        raise ValueError("Completed report changed")
    for path, expected in plan["frozen_hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen source changed")
    for directory in [ROOT, ROOT / "generated"]:
        for row in read_json(ROOT / "generated/complete.json")["images"] if directory == ROOT / "generated" else []:
            path = directory / row["file"]
            with Image.open(path) as image:
                image.verify()
            if file_hash(path) != row["sha256"]:
                raise ValueError("Generated PNG mismatch")
    for marker in [ROOT / "capture_complete.json", ROOT / "reconstruction/complete.json",
                   ROOT / "oracle/complete.json", ROOT / "reconstructed/complete.json"]:
        for path, expected in read_json(marker)["hashes"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("Committed artifact changed")
    report = read_json(ROOT / "report.json")
    labels = [r["label"] for r in read_json(ROOT / "private/targets.json")]
    classes = plan["classes"]
    oracle_keys = []
    for i in range(10):
        obs = read_json(ROOT / "captures" / f"run-{i:05d}" / "oracle/observation.json")
        payload = repr([obs["public_question_ids"], obs["public_target_ids"]]).encode()
        oracle_keys.append(hashlib.sha256(payload).hexdigest())
        private = read_json(ROOT / "captures" / f"run-{i:05d}" / "private/observation.json")
        if any(private[k] for k in ["public_question_ids", "public_target_ids", "public_questions", "public_targets"]):
            raise ValueError("Private observation exposes reference text")
    same_text = {}
    for phase in ["oracle", "reconstructed"]:
        predictions = [read_json(ROOT / phase / "predictions" / f"{i:05d}-{phase}.json") for i in range(10)]
        groups = defaultdict(list)
        for i in range(10):
            key = (oracle_keys[i], predictions[i]["text_sha256"] if phase == "reconstructed" else "oracle")
            groups[key].append(i)
        groups = [ids for ids in groups.values() if len({labels[i] for i in ids}) == 2]
        rows = []
        for ids in groups:
            scores = [predictions[i]["class_scores"][plan["primary_score"]] for i in ids]
            mismatch, opposite = [], []
            for i in ids:
                pred = predictions[i]
                wrong = dict(zip(pred["shuffled_target_runs"], pred["shuffled_class_scores"], strict=True))
                def credit(j):
                    values = wrong[j][plan["primary_score"]]
                    return float(values is not None and classes[int(np.argmin(values))] == labels[i])
                mismatch.append(float(np.mean([credit(j) for j in ids if j != i])))
                opposite.append(float(np.mean([credit(j) for j in ids if labels[j] != labels[i]])))
            rows.append({"runs": ids, "correct_gradient": classification_stats([labels[i] for i in ids], scores, classes),
                         "wrong_gradient_accuracy": float(np.mean(mismatch)),
                         "opposite_class_wrong_accuracy": float(np.mean(opposite)),
                         "wrong_gradient_per_target_accuracy": mismatch})
        same_text[phase] = {"groups": rows, "images": sum(len(ids) for ids in groups),
                            "rule": "All exact-oracle-QA groups with both classes; reconstructed phase also requires identical reconstructed tokens"}
    write_json(ROOT / "same_text_control.json", same_text)
    write_json(ROOT / "integrity_audit.json", {
        "status": "pass", "frozen_inputs_match": True, "generated_pngs": 10,
        "paired_captures": 10, "reconstructions": 10, "predictions": 20,
        "private_observations_expose_reference_text": False,
        "model_unchanged": complete["model_unchanged"],
        "exact_replay_max_relative_l2": read_json(ROOT / "capture_complete.json")["max_relative_l2"],
        "report_sha256": file_hash(ROOT / "report.json"),
        "same_text_plan_sha256": file_hash(ROOT / "same_text_control_plan.json"),
        "same_text_report_sha256": file_hash(ROOT / "same_text_control.json"),
    })
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, phase, title in zip(axes, ["oracle", "reconstructed"], ["Correct text", "Reconstructed text"], strict=True):
        values = report["phases"][phase]
        methods = ["centered", "global", "equal_tensor", "centered_equal_tensor", "model_only", "text_only"]
        names = ["Centered\n(primary)", "Global", "Equal\ntensor", "Centered\nequal tensor", "Model\nonly", "Text\nonly"]
        heights = [values["metrics"][m]["accuracy"] for m in methods]
        ax.bar(np.arange(6), heights, color=["#2372b5"] + ["#80b4d5"] * 3 + ["#999999"] * 2)
        ax.axhline(.5, color="black", linestyle=":", label="Balanced chance (0.50)")
        wrong = values["wrong_gradient"]["centered"]["accuracy"]
        ax.axhline(wrong, color="#c65c35", linestyle="--", label=f"Wrong-gradient primary ({wrong:.2f})")
        for i, h in enumerate(heights):
            ax.text(i, h + .025, f"{round(h * 10)}/10", ha="center", fontsize=9)
        ax.set(xticks=np.arange(6), xticklabels=names, ylim=(0, 1.25), yticks=np.linspace(0, 1, 6),
               ylabel="Property accuracy", title=title)
        ax.legend(loc="upper left", fontsize=7)
    fig.savefig(figure_dir / "accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, phase, title in zip(axes, ["oracle", "reconstructed"], ["Correct text", "Reconstructed text"], strict=True):
        for row in report["phases"][phase]["per_target"]:
            scores = row["primary_scores"]
            true = classes.index(row["label"])
            margin = scores[1 - true] - scores[true]
            ax.scatter(row["run"] + 1, margin, c="#2372b5" if true == 0 else "#d67b27", s=65)
        ax.axhline(0, color="black", linewidth=1)
        ax.set(xticks=range(1, 11), xlabel="Anonymous target index (1–5 head; 6–10 abdomen)",
               ylabel="Other-class loss minus correct-class loss", title=title)
    fig.savefig(figure_dir / "margins.png", dpi=180)
    plt.close(fig)
    destination = Path("docs/experiments/figures/synthetic-ct-property")
    destination.mkdir(parents=True, exist_ok=True)
    for path in figure_dir.glob("*.png"):
        shutil.copyfile(path, destination / path.name)
    write_json(figure_dir / "manifest.json", {
        "source_report_sha256": file_hash(ROOT / "report.json"),
        "figures": {str(p): file_hash(p) for p in figure_dir.glob("*.png")},
        "contains_private_images_or_text": False,
    })
    print({phase: {m: value["accuracy"] for m, value in r["metrics"].items()}
           for phase, r in report["phases"].items()}, flush=True)
    print("Same-text control:", same_text, flush=True)


if __name__ == "__main__":
    summarize()
