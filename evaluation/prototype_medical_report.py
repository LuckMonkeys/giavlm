"""Evaluate the frozen generic versus synthetic-medical prototype control."""

import argparse
import csv
import io
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.crossmodel_property import METHODS, PUBLIC
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.prototype_boundary_report import group_labels, score_pairs
from evaluation.prototype_boundary_study import SOURCE, load_gradients
from evaluation.prototype_method_report import manual_auc, quality


CLASSES = ["head", "non-head"]


def groups_for(root, family):
    cases = read_json(root / family / "bank-cases.json")
    index = {r["id"]: i for i, r in enumerate(cases)}
    generic_labels = {
        **read_json(PUBLIC / "fit_labels.json"),
        **read_json(PUBLIC / "evaluation_labels.json"),
    }
    groups = {}
    for bank, rows in (
        ("generic", read_json(root / "public-generic-bank.json")),
        ("medical", read_json(root / "public-medical-bank.json")),
    ):
        for split in ("calibration", "audit"):
            selected = [r for r in rows if r["split"] == split]
            groups[f"{bank}_{split}"] = (
                [index[r["id"]] for r in selected],
                [generic_labels[r["id"]] if bank == "generic" else r["label"] for r in selected],
            )
    old_cases, old_groups, mapping = group_labels(family)
    for name, (ids, labels) in old_groups.items():
        if not name.startswith("public_"):
            groups[name] = ([index[old_cases[i]["id"]] for i in ids], labels)
    for dataset in ("slake", "rad"):
        targets = {r["run"]: r for r in read_json(SOURCE / "private" / f"{dataset}-targets.json")}
        for qa in ("natural", "fixed"):
            ids, labels = groups[f"{dataset}_{qa}"]
            for modality in ("CT", "MRI", "X-Ray"):
                selected = [
                    (i, y)
                    for i, y in zip(ids, labels, strict=True)
                    if targets[mapping[cases[i]["id"]]["run"]]["modality"] == modality
                ]
                if selected:
                    groups[f"{dataset}_{qa}_{modality}"] = (
                        [i for i, _ in selected],
                        [y for _, y in selected],
                    )
    return cases, groups


def verify_commits(root):
    for family in ("llava", "qwen35"):
        out = root / family
        commit = read_json(out / "bank-predictions-complete.json")
        for name, key in (
            ("bank-predictions.json", "sha256"),
            ("bank-rules.json", "rules_sha256"),
            ("bank-cases.json", "cases_sha256"),
            ("bank-prototypes.safetensors", "prototype_sha256"),
        ):
            if file_hash(out / name) != commit[key]:
                raise ValueError("Frozen bank predictions or rules changed")
        if commit["target_labels_read"]:
            raise ValueError("Target labels used by scoring worker")


def evaluate(root):
    # Both model commits are checked before the evaluator joins any target labels.
    verify_commits(root)
    metrics, comparisons, distributions = {}, {}, {}
    csv_rows = []
    for family in ("llava", "qwen35"):
        cases, groups = groups_for(root, family)
        predictions = read_json(root / family / "bank-predictions.json")
        if [r["id"] for r in predictions] != [r["id"] for r in cases]:
            raise ValueError("Case order differs from committed predictions")
        rules = read_json(root / family / "bank-rules.json")
        metrics[family], comparisons[family], distributions[family] = {}, {}, {}
        for method in METHODS:
            all_scores, margins = {}, {}
            for bank in ("generic", "medical"):
                key = f"{bank}_{method}"
                d = np.asarray([r["distances"][key] for r in predictions])
                margins[bank] = d[:, 1] - d[:, 0]
                for boundary in ("zero", "public"):
                    tau = 0.0 if boundary == "zero" else rules[key]["threshold"]
                    all_scores[f"{bank}_{boundary}"] = score_pairs(
                        margins[bank], tau, np.ones(len(cases), dtype=bool)
                    )
            metrics[family][method], comparisons[family][method], distributions[family][method] = (
                {},
                {},
                {},
            )
            for group, (ids, labels) in groups.items():
                results = {}
                for name, values in all_scores.items():
                    selected = [values[i] for i in ids]
                    result = classification_stats(labels, selected, CLASSES)
                    result.update(paired_bootstrap(labels, selected, CLASSES))
                    results[name] = result
                    csv_rows.append(
                        [
                            family,
                            method,
                            group,
                            name,
                            result["n"],
                            result["correct"],
                            result["balanced_accuracy"],
                            result["recall"]["head"],
                            result["recall"]["non-head"],
                            result["class_auroc"]["head"],
                        ]
                    )
                metrics[family][method][group] = results
                comparisons[family][method][group] = {}
                for boundary in ("zero", "public"):
                    generic = [all_scores[f"generic_{boundary}"][i] for i in ids]
                    medical = [all_scores[f"medical_{boundary}"][i] for i in ids]
                    delta = (
                        None
                        if results[f"medical_{boundary}"]["balanced_accuracy"] is None
                        else (
                            results[f"medical_{boundary}"]["balanced_accuracy"]
                            - results[f"generic_{boundary}"]["balanced_accuracy"]
                        )
                    )
                    comparisons[family][method][group][boundary] = {
                        "balanced_accuracy_gain": delta,
                        **paired_bootstrap(labels, medical, CLASSES, generic),
                    }
                from evaluation.prototype_boundary_math import scalar_stats

                distributions[family][method][group] = {
                    bank: {
                        label: scalar_stats(
                            [
                                margins[bank][i]
                                for i, y in zip(ids, labels, strict=True)
                                if y == label
                            ]
                        )
                        for label in CLASSES
                        if label in labels
                    }
                    for bank in ("generic", "medical")
                }
    aggregate = {
        "primary": "Q24",
        "secondary": "Q32",
        "primary_qa": "natural",
        "condition": "Equal-size wholly synthetic medical-appearance bank versus existing synthetic generic photography",
        "metrics": metrics,
        "comparisons": comparisons,
        "distributions": distributions,
        "rules": {f: read_json(root / f / "bank-rules.json") for f in ("llava", "qwen35")},
        "unique_target_images": 40,
        "new_target_images": 0,
        "public_medical_synthetic_images": 32,
        "public_generic_images": 32,
        "evidence": "Repeated-target mechanism diagnosis; new public generated bank",
        "domain_prior": "Medical imaging and CT/MRI appearance prompts declared",
        "clinical_validity": False,
        "uncommitted": True,
    }
    write_json(root / "aggregate.json", aggregate)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Model",
            "Module",
            "Group",
            "Algorithm",
            "N",
            "Correct",
            "Balanced accuracy",
            "Head recall",
            "Negative recall",
            "Head AUROC",
        ]
    )
    writer.writerows(csv_rows)
    (root / "summary.csv").write_text(stream.getvalue())
    return aggregate


def audit(root, aggregate):
    verify_commits(root)
    generated = read_json(root / "generation/complete.json")
    if generated["plan_sha256"] != file_hash(root / "generation/plan.json"):
        raise ValueError("Generation plan changed")
    if not read_json(root / "visual-review.json")["all_32_accepted"]:
        raise ValueError("Synthetic bank failed pre-score visual review")
    for row in generated["images"]:
        if file_hash(row["file"]) != row["sha256"]:
            raise ValueError("Generated image bytes changed")
    plan = read_json(root / "plan.json")
    for p, sha in plan["hashes"].items():
        if file_hash(p) != sha:
            raise ValueError("Frozen input changed")
    for name in ("source-medical-generation-manifest.json", "source-medical-probe-manifest.json"):
        for p, sha in read_json(root / name)["hashes"].items():
            if file_hash(p) != sha:
                raise ValueError("Frozen execution source changed")
    generic_rows = read_json(root / "public-generic-bank.json")
    medical_rows = read_json(root / "public-medical-bank.json")
    public_labels = {
        **read_json(PUBLIC / "fit_labels.json"),
        **read_json(PUBLIC / "evaluation_labels.json"),
    }
    metric_count, distance_count, prototype_count, max_error = 0, 0, 0, 0.0
    for family in ("llava", "qwen35"):
        out = root / family
        if not read_json(out / "native-capture-gate.json")["all_tensors_exact"]:
            raise ValueError("Native capture gate failed")
        gpu_complete = read_json(out / "gpu-complete.json")
        if "source-medical-probe-v1" not in gpu_complete["runtime_source"]:
            raise ValueError("GPU worker did not run the frozen source")
        if (
            gpu_complete["fingerprint"]
            != read_json(SOURCE / family / "runtime.json")["fingerprint"]
        ):
            raise ValueError("Victim model snapshot differs")
        if gpu_complete["training"] != read_json(SOURCE / family / "runtime.json")["training"]:
            raise ValueError("Public replay protocol differs from original medical uploads")
        generic_features = read_tensors(SOURCE / family / "public/visual-means.safetensors")
        medical_features = read_tensors(out / "visual-means.safetensors")
        stored = read_tensors(out / "bank-prototypes.safetensors")
        for bank, rows, features in (
            ("generic", generic_rows, generic_features),
            ("medical", medical_rows, medical_features),
        ):
            for method in METHODS:
                centers = torch.stack(
                    [
                        torch.stack(
                            [
                                features[f"{r['id']}_{method}"]
                                for r in rows
                                if r["split"] == "fit"
                                and (public_labels[r["id"]] if bank == "generic" else r["label"])
                                == label
                            ]
                        ).mean(0)
                        for label in CLASSES
                    ]
                )
                expected = centers / centers.norm(dim=1, keepdim=True)
                error = float((expected - stored[f"{bank}_{method}"]).abs().max())
                if error > 1e-12:
                    raise ValueError("Independently reconstructed prototype differs")
                max_error = max(max_error, error)
                prototype_count += 1
        cases, groups = groups_for(root, family)
        rows = read_json(out / "bank-predictions.json")
        rules = read_json(out / "bank-rules.json")
        for method in METHODS:
            for bank, bank_rows in (("generic", generic_rows), ("medical", medical_rows)):
                key = f"{bank}_{method}"
                d = np.asarray([r["distances"][key] for r in rows])
                values = d[:, 1] - d[:, 0]
                cal_ids, cal_labels = groups[f"{bank}_calibration"]
                vals = values[cal_ids]
                labels = [y == "head" for y in cal_labels]
                unique = np.unique(vals)
                candidates = [
                    0.0,
                    float(np.nextafter(unique[0], -np.inf)),
                    float(np.nextafter(unique[-1], np.inf)),
                ]
                candidates += [
                    float(a + (b - a) / 2) for a, b in zip(unique[:-1], unique[1:], strict=True)
                ]
                tau = max(
                    set(candidates),
                    key=lambda t: (
                        *quality(vals, labels, ["fixed"] * len(vals), t)[:2],
                        -abs(t),
                        -t,
                    ),
                )
                if tau != rules[key]["threshold"]:
                    raise ValueError("Independent public threshold differs")
                for group, (ids, ys) in groups.items():
                    y = np.asarray([v == "head" for v in ys])
                    for boundary in ("zero", "public"):
                        prediction = values[ids] >= (0.0 if boundary == "zero" else tau)
                        correct = int((prediction == y).sum())
                        both = bool(y.any() and not y.all())
                        ba = (
                            float((prediction[y].mean() + (~prediction[~y]).mean()) / 2)
                            if both
                            else None
                        )
                        auc = manual_auc(y, values[ids]) if both else None
                        actual = aggregate["metrics"][family][method][group][f"{bank}_{boundary}"]
                        if (
                            actual["correct"] != correct
                            or (
                                abs(actual["balanced_accuracy"] - ba) > 1e-12
                                if both
                                else actual["balanced_accuracy"] is not None
                            )
                            or actual["class_auroc"]["head"] != auc
                        ):
                            raise ValueError("Independent confusion/AUROC differs")
                        metric_count += 1
        # Every stored score is independently projected; no target labels enter this step.
        for i, case in enumerate(cases):
            gradients = load_gradients(case, family)
            for method in METHODS:
                _, s, vh = torch.linalg.svd(gradients[method].double(), full_matrices=False)
                basis = vh[s > max(1e-10, float(s[0]) * 1e-6)]
                for bank in ("generic", "medical"):
                    mu = stored[f"{bank}_{method}"]
                    expected = (mu - (mu @ basis.T) @ basis).norm(dim=1).numpy()
                    error = float(
                        np.max(
                            np.abs(expected - np.asarray(rows[i]["distances"][f"{bank}_{method}"]))
                        )
                    )
                    if error > 1e-12:
                        raise ValueError("Independent direct SVD distance differs")
                    max_error = max(max_error, error)
                    distance_count += 1
    return {
        "status": "pass",
        "native_capture_models": 2,
        "prototypes_rebuilt": prototype_count,
        "independent_distance_pairs": distance_count,
        "manual_metric_groups": metric_count,
        "public_only_thresholds_reproduced": True,
        "both_prediction_commits_verified": True,
        "max_distance_error": max_error,
        "targets_used_for_fitting": False,
    }


def figures(root, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    names = ["slake_natural", "rad_natural", "slake_fixed", "rad_fixed"]
    labels = ["SLAKE natural", "RAD natural", "SLAKE fixed", "RAD fixed"]
    colors = ["#9eaeb7", "#486d83", "#dfb695", "#af633b"]
    for i, family in enumerate(("llava", "qwen35")):
        for j, method in enumerate(("Q24", "Q32")):
            ax = axes[i, j]
            for k, algorithm in enumerate(
                ("generic_zero", "generic_public", "medical_zero", "medical_public")
            ):
                values = [
                    aggregate["metrics"][family][method][n][algorithm]["balanced_accuracy"]
                    for n in names
                ]
                ax.bar(
                    np.arange(4) + (k - 1.5) * 0.18,
                    values,
                    0.17,
                    label=algorithm.replace("_", " "),
                    color=colors[k],
                )
            ax.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8)
            ax.set_xticks(range(4), labels)
            ax.set_ylim(0, 1.08)
            ax.set_title(f"{family} · {method}")
            ax.set_ylabel("Balanced accuracy")
            ax.legend(fontsize=8, ncol=2)
    path = root / "synthetic-medical-transfer.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    torch.set_num_threads(1)
    try:
        aggregate = evaluate(root)
        audited = audit(root, aggregate)
        write_json(root / "integrity-audit.json", audited)
        images = figures(root, aggregate)
        repo = Path.cwd()
        data = repo / "docs/experiments/data/gradient-prototype-transfer"
        dest = repo / "docs/experiments/figures/gradient-prototype-transfer"
        published = {}
        for name in ("aggregate.json", "summary.csv", "integrity-audit.json"):
            path = data / f"synthetic-medical-{name}"
            shutil.copy2(root / name, path)
            published[str(path.relative_to(repo))] = file_hash(path)
        for image in images:
            path = dest / image.name
            shutil.copy2(image, path)
            published[str(path.relative_to(repo))] = file_hash(path)
        write_json(
            data / "synthetic-medical-manifest.json",
            {
                "artifacts": published,
                "analysis_source_sha256": file_hash(Path(__file__)),
                "execution_source_manifest_sha256": file_hash(
                    root / "source-medical-probe-manifest.json"
                ),
                "generation_source_manifest_sha256": file_hash(
                    root / "source-medical-generation-manifest.json"
                ),
                "uncommitted": True,
            },
        )
        write_json(
            root / "report-complete.json",
            {
                "status": "complete",
                "aggregate_sha256": file_hash(root / "aggregate.json"),
                "audit_sha256": file_hash(root / "integrity-audit.json"),
                "runtime_source": str(Path(__file__).resolve()),
            },
        )
        print(audited, flush=True)
    except BaseException:
        write_json(root / "report-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
