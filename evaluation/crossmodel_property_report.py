"""Independent distance/prototype audits and aggregate-only cross-model report."""

import argparse
from collections import Counter
import csv
import io
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from core.fl import load_observation
from evaluation.crossmodel_property import METHODS, PUBLIC, module_key, validate_observation
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.multilayer_property_data import TERMS


CLASSES = ["head", "non-head"]


def fixed_swap_control(truth, scores):
    """Evaluator-only opposite-property swap inside the same modality and QA."""
    ids, other_ids, pair_counts = [], [], {}
    for modality in ("CT", "MRI", "X-Ray"):
        positive = [
            i
            for i, row in enumerate(truth)
            if row["modality"] == modality and row["label"] == "head"
        ]
        negative = [
            i
            for i, row in enumerate(truth)
            if row["modality"] == modality and row["label"] == "non-head"
        ]
        pairs = list(zip(positive, negative))
        pair_counts[modality] = len(pairs)
        for a, b in pairs:
            ids.extend((a, b))
            other_ids.extend((b, a))
    labels = [truth[i]["label"] for i in ids]
    return {
        "pairs_by_modality": pair_counts,
        "paired_images": len(ids),
        "correct_gradient": classification_stats(labels, [scores[i] for i in ids], CLASSES),
        "swapped_gradient": classification_stats(labels, [scores[i] for i in other_ids], CLASSES),
        "scope": "Post-commit diagnostic only; pairs are dependent, fixed QA unchanged, no new uploads or method selection",
    }


def independent_distances(prototype, gradient):
    _, singular, vh = torch.linalg.svd(gradient.double().cpu(), full_matrices=False)
    basis = vh[singular > max(float(singular[0]) * 1e-6, 1e-10)]
    values = prototype.double().cpu()
    return (values - (values @ basis.T) @ basis).norm(dim=1).tolist()


def audit(root):
    plan = read_json(root / "plan.json")
    for path, expected in plan["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen input changed")
    for path, expected in read_json(root / "source-crossmodel-property-manifest.json")[
        "hashes"
    ].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen execution source changed")
    # No target labels are joined until both models have committed their complete scores.
    for family in ("llava", "qwen35"):
        out = root / family
        complete = read_json(out / "complete.json")
        if (
            file_hash(out / "predictions.json") != complete["prediction_sha256"]
            or file_hash(out / "predictions.json")
            != read_json(out / "predictions-complete.json")["sha256"]
        ):
            raise ValueError("Medical predictions changed")
        if "source-crossmodel-property-v1" not in complete["runtime_source"]:
            raise ValueError("Worker did not use frozen source")
    max_error, checked, prototypes_checked = 0.0, 0, 0
    for family in ("llava", "qwen35"):
        out = root / family
        runtime = read_json(out / "runtime.json")
        gate = read_json(out / "native-capture-gate.json")
        if not gate["all_tensors_exact"] or gate["fingerprint"] != runtime["fingerprint"]:
            raise ValueError("Original native capture gate failed")
        keys = {method: module_key(family, method) for method in METHODS}
        if keys != runtime["keys"]:
            raise ValueError("Unexpected architecture mapping")
        for path, expected in read_json(out / "capture-complete.json")["hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("Native upload file changed")
        prototype = read_tensors(out / "public/prototypes.safetensors")
        if (
            file_hash(out / "public/prototypes.safetensors")
            != read_json(out / "public/predictions-complete.json")["prototype_sha256"]
        ):
            raise ValueError("Public prototypes changed after prediction")
        means, labels = (
            read_tensors(out / "public/visual-means.safetensors"),
            read_json(PUBLIC / "fit_labels.json"),
        )
        rows = read_json(root / "public-bank.json")
        for method in METHODS:
            centers = torch.stack(
                [
                    torch.stack(
                        [
                            means[f"{row['id']}_{method}"]
                            for row in rows
                            if row["split"] == "fit" and labels[row["id"]] == label
                        ]
                    ).mean(0)
                    for label in CLASSES
                ]
            )
            centers = torch.nn.functional.normalize(centers, dim=1)
            if not torch.equal(centers, prototype[method]):
                raise ValueError("Public prototype refit differs")
            prototypes_checked += 1
        medical = {row["id"]: row["scores"] for row in read_json(out / "predictions.json")}
        for row in read_json(out / "upload-list.json"):
            observation = load_observation(row["observation"], "cpu")
            validate_observation(observation, runtime["fingerprint"])
            for method in METHODS:
                actual = independent_distances(prototype[method], observation.tensors[keys[method]])
                error = float(np.max(np.abs(np.asarray(actual) - medical[row["id"]][method])))
                if error > 1e-12:
                    raise ValueError("Independent medical distance mismatch")
                checked, max_error = checked + 1, max(max_error, error)
            del observation
        public = {row["id"]: row["scores"] for row in read_json(out / "public/predictions.json")}
        if (
            file_hash(out / "public/predictions.json")
            != read_json(out / "public/predictions-complete.json")["sha256"]
        ):
            raise ValueError("Public predictions changed")
        for row in read_json(out / "public/gradient-list.json"):
            gradients = read_tensors(row["path"])
            for method in METHODS:
                actual = independent_distances(prototype[method], gradients[method])
                error = float(np.max(np.abs(np.asarray(actual) - public[row["id"]][method])))
                if error > 1e-12:
                    raise ValueError("Independent public distance mismatch")
                checked, max_error = checked + 1, max(max_error, error)
    result = {
        "status": "pass",
        "native_medical_uploads": 160,
        "public_uploads": 96,
        "distances_recomputed": checked,
        "prototypes_recomputed": prototypes_checked,
        "max_distance_error": max_error,
        "target_text_or_lengths_used_by_classifier": False,
        "predictions_committed_before_evaluation": True,
        "primary": "Q32",
    }
    write_json(root / "integrity-audit.json", result)
    return result


def report(root):
    integrity = audit(root)
    targets = {
        ds: {row["run"]: row for row in read_json(root / "private" / f"{ds}-targets.json")}
        for ds in ("slake", "rad")
    }
    medical, public, composition, paired, aligned, swaps = {}, {}, {}, {}, {}, {}
    for ds, rows in targets.items():
        composition[ds] = {
            "images": len(rows),
            "regions": dict(Counter(r["region"] for r in rows.values())),
            "modalities": dict(Counter(r["modality"] for r in rows.values())),
            "answer_types": dict(
                Counter(
                    "yes_no"
                    if r["row"]["target"].strip().lower().rstrip(".!? ") in ("yes", "no")
                    else "other"
                    for r in rows.values()
                )
            ),
        }
    csv_rows = []
    for family in ("llava", "qwen35"):
        out = root / family
        mapping = read_json(out / "private/upload-map.json")
        predictions = read_json(out / "predictions.json")
        medical[family], aligned[family], paired[family], swaps[family] = {}, {}, {}, {}
        for ds, by_run in targets.items():
            truth = [by_run[run] for run in sorted(by_run)]
            labels = [r["label"] for r in truth]
            medical[family][ds], aligned[family][ds] = {}, {}
            for qa in ("natural", "fixed"):
                by_id = {
                    mapping[r["id"]]["run"]: r
                    for r in predictions
                    if r["dataset"] == ds and r["qa"] == qa
                }
                values = {m: [by_id[run]["scores"][m] for run in sorted(by_run)] for m in METHODS}
                aligned[family][ds][qa] = values
                groups = {
                    "all": list(range(20)),
                    **{
                        name: [i for i, r in enumerate(truth) if r["modality"] == name]
                        for name in ("CT", "MRI", "X-Ray")
                    },
                    "no_literal_head_terms": [
                        i
                        for i, r in enumerate(truth)
                        if not TERMS["head"].search(r["row"]["question"] + " " + r["row"]["target"])
                    ],
                    **{
                        name: [
                            i
                            for i, r in enumerate(truth)
                            if (r["row"]["target"].strip().lower().rstrip(".!? ") in ("yes", "no"))
                            == (name == "yes_no")
                        ]
                        for name in ("yes_no", "other")
                    },
                }
                medical[family][ds][qa] = {}
                for group, ids in groups.items():
                    y = [labels[i] for i in ids]
                    metrics = {
                        m: {
                            **classification_stats(y, [v[i] for i in ids], CLASSES),
                            **(paired_bootstrap(y, [v[i] for i in ids], CLASSES) or {}),
                        }
                        for m, v in values.items()
                    }
                    medical[family][ds][qa][group] = metrics
                for method, value in medical[family][ds][qa]["all"].items():
                    manual = np.zeros((2, 3), dtype=int)
                    for label, pair in zip(labels, values[method], strict=True):
                        manual[CLASSES.index(label), int(np.argmin(pair))] += 1
                    if (
                        manual.tolist() != value["confusion"]
                        or int(np.trace(manual[:, :2])) != value["correct"]
                    ):
                        raise ValueError("Independent confusion count differs")
                    csv_rows.append(
                        [
                            family,
                            ds,
                            qa,
                            method,
                            method == "Q32",
                            value["correct"],
                            value["n"],
                            value["accuracy"],
                            value["balanced_accuracy"],
                            value["recall"]["head"],
                            value["recall"]["non-head"],
                            value["class_auroc"]["head"],
                        ]
                    )
            natural, fixed = [aligned[family][ds][qa]["Q32"] for qa in ("natural", "fixed")]
            paired[family][ds] = {
                "fixed_vs_natural": {
                    "accuracy_gain": float(
                        np.mean(np.argmin(fixed, 1) == np.array([CLASSES.index(y) for y in labels]))
                        - np.mean(
                            np.argmin(natural, 1) == np.array([CLASSES.index(y) for y in labels])
                        )
                    ),
                    **paired_bootstrap(labels, fixed, CLASSES, natural),
                }
            }
            swaps[family][ds] = fixed_swap_control(truth, fixed)
        public_labels = read_json(PUBLIC / "evaluation_labels.json")
        p = read_json(out / "public/predictions.json")
        public[family] = {
            m: classification_stats(
                [public_labels[r["id"]] for r in p], [r["scores"][m] for r in p], CLASSES
            )
            for m in METHODS
        }
    comparisons = {}
    for ds, by_run in targets.items():
        labels = [by_run[run]["label"] for run in sorted(by_run)]
        comparisons[ds] = {
            qa: paired_bootstrap(
                labels, aligned["qwen35"][ds][qa]["Q32"], CLASSES, aligned["llava"][ds][qa]["Q32"]
            )
            for qa in ("natural", "fixed")
        }
    runtimes = {
        family: {
            "model_fingerprint": read_json(root / family / "runtime.json")["fingerprint"],
            "model_family": read_json(root / family / "runtime.json")["model"]["family"],
            "image_size": read_json(root / family / "runtime.json")["model"]["image_size"],
            "question_length": 64,
            "target_length": 64,
            "round": 10,
            **{
                key: read_json(root / family / "complete.json")[key]
                for key in ("seconds", "peak_memory_mib")
            },
        }
        for family in ("llava", "qwen35")
    }
    aggregate = {
        "primary": "Q32",
        "medical": medical,
        "public": public,
        "composition": composition,
        "paired": paired,
        "model_comparison_ci": comparisons,
        "fixed_qa_same_modality_swap": swaps,
        "runtime": runtimes,
        "uses_private_reference": "Post-commit evaluation labels and subgroup definitions only",
        "uncommitted": True,
        "audit": integrity,
        "scope": "Same SLAKE-trained snapshot per model; SLAKE A repeated; VQA-RAD20 newly tested; native resolutions and architectures differ",
    }
    write_json(root / "aggregate.json", aggregate)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Model",
            "Dataset",
            "QA",
            "Method",
            "Primary",
            "Correct",
            "Images",
            "Accuracy",
            "Balanced accuracy",
            "Head recall",
            "Non-head recall",
            "Head AUROC",
        ]
    )
    writer.writerows(csv_rows)
    (root / "summary.csv").write_text(stream.getvalue())
    figure = plot(aggregate, root)
    destination = Path("docs/experiments/data/gradient-prototype-transfer")
    destination.mkdir(parents=True, exist_ok=True)
    published = []
    for source, name in [
        (root / "aggregate.json", "crossmodel-aggregate.json"),
        (root / "summary.csv", "crossmodel-summary.csv"),
        (root / "integrity-audit.json", "crossmodel-audit.json"),
    ]:
        target = destination / name
        shutil.copy2(source, target)
        published.append(target)
    target = Path("docs/experiments/figures/gradient-prototype-transfer/crossmodel-head.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure, target)
    published.append(target)
    shutil.copy2(Path(__file__), root / "analysis-source.py")
    write_json(
        destination / "crossmodel-manifest.json",
        {
            "uncommitted": True,
            "artifacts": {str(p): file_hash(p) for p in published},
            "analysis_source_sha256": file_hash(Path(__file__)),
            "execution_source_manifest_sha256": file_hash(
                root / "source-crossmodel-property-manifest.json"
            ),
        },
    )
    write_json(
        root / "report-complete.json",
        {
            "aggregate_sha256": file_hash(root / "aggregate.json"),
            "audit_sha256": file_hash(root / "integrity-audit.json"),
            "status": "complete",
        },
    )
    for family in ("llava", "qwen35"):
        print(
            family,
            {
                ds: {
                    qa: aggregate["medical"][family][ds][qa]["all"]["Q32"]["correct"]
                    for qa in ("natural", "fixed")
                }
                for ds in ("slake", "rad")
            },
            flush=True,
        )


def plot(aggregate, root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, ds in zip(axes, ("slake", "rad"), strict=True):
        x = np.arange(2)
        for i, (qa, color) in enumerate((("natural", "#4c78a8"), ("fixed", "#f28e2b"))):
            values = [
                aggregate["medical"][family][ds][qa]["all"]["Q32"]["balanced_accuracy"]
                for family in ("llava", "qwen35")
            ]
            axis.bar(
                x + (i - 0.5) * 0.34,
                values,
                width=0.32,
                label="Original QA" if qa == "natural" else "Fixed QA",
                color=color,
            )
            for j, value in enumerate(values):
                axis.text(
                    j + (i - 0.5) * 0.34, value + 0.018, f"{value:.2f}", ha="center", fontsize=10
                )
        axis.set_xticks(x, ["LLaVA-1.5-7B", "Qwen3.5-4B"])
        axis.axhline(0.5, color="#999999", linestyle="--", linewidth=1)
        axis.set_title("SLAKE A (repeated 20 images)" if ds == "slake" else "VQA-RAD (20 images)")
        axis.set_ylim(0, 1.13)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Head balanced accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Q32 primary | generic public prototypes | no text reconstruction", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    path = root / "crossmodel-head.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        report(args.root.resolve())
    except BaseException:
        write_json(args.root / "analysis-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
