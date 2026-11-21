"""Evaluator-only boundary statistics, independent checks and aggregate plots."""

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
from evaluation.prototype_boundary_math import decompose, paired_interval, scalar_stats
from evaluation.prototype_boundary_study import PREVIOUS, SOURCE, load_gradients
from evaluation.prototype_method_report import manual_auc, quality


CLASSES = ["head", "non-head"]


def group_labels(family):
    cases = read_json(PREVIOUS / f"{family}-cases.json")
    old_plan = read_json(PREVIOUS / "plan.json")
    public_labels = read_json(PUBLIC / "evaluation_labels.json")
    index = {r["id"]: i for i, r in enumerate(cases)}
    groups = {}
    for name in ("calibration", "audit"):
        rows = old_plan[f"{name}_rows"]
        groups[f"public_{name}"] = (
            [index[r["id"]] for r in rows],
            [public_labels[r["id"]] for r in rows],
        )
    mapping = read_json(SOURCE / family / "private/upload-map.json")
    for dataset in ("slake", "rad"):
        targets = {r["run"]: r for r in read_json(SOURCE / "private" / f"{dataset}-targets.json")}
        for qa in ("natural", "fixed"):
            ids = [
                i
                for i, r in enumerate(cases)
                if r["kind"] == "medical" and r["dataset"] == dataset and r["qa"] == qa
            ]
            groups[f"{dataset}_{qa}"] = (
                ids,
                [targets[mapping[cases[i]["id"]]["run"]]["label"] for i in ids],
            )
    return cases, groups, mapping


def score_pairs(values, threshold, valid):
    return [
        [-(float(v) - threshold), float(v) - threshold] if ok else None
        for v, ok in zip(values, valid, strict=True)
    ]


def evaluate(root):
    commit = read_json(root / "cpu/predictions-complete.json")
    if file_hash(root / "cpu/predictions.json") != commit["sha256"]:
        raise ValueError("Committed CPU predictions changed")
    predictions = read_json(root / "cpu/predictions.json")
    results, distributions, paired = {}, {}, {}
    csv_rows, dist_rows = [], []
    for family in ("llava", "qwen35"):
        cases, groups, mapping = group_labels(family)
        rows = predictions[family]["rows"]
        if [r["id"] for r in rows] != [r["id"] for r in cases]:
            raise ValueError("Anonymous upload ordering changed")
        results[family], distributions[family], paired[family] = {}, {}, {}
        for m in METHODS:
            z = decompose([r["distances"][m] for r in rows])
            rule = predictions[family]["rules"][m]
            tau, rho = rule["raw_threshold"], rule["relative_rule"]["threshold"]
            all_scores = {
                "raw_zero": score_pairs(z["margin"], 0.0, np.ones(len(rows), dtype=bool)),
                "raw_public": score_pairs(z["margin"], tau, np.ones(len(rows), dtype=bool)),
                "relative_zero": score_pairs(z["relative"], 0.0, z["valid"]),
                "relative_public": score_pairs(z["relative"], rho, z["valid"]),
            }
            results[family][m], distributions[family][m], paired[family][m] = {}, {}, {}
            for group, (ids, labels) in groups.items():
                results[family][m][group] = {}
                for algorithm, scores in all_scores.items():
                    selected = [scores[i] for i in ids]
                    stats = classification_stats(labels, selected, CLASSES)
                    stats.update(paired_bootstrap(labels, selected, CLASSES))
                    stats["gain_vs_raw_public"] = paired_bootstrap(
                        labels, selected, CLASSES, [all_scores["raw_public"][i] for i in ids]
                    )
                    results[family][m][group][algorithm] = stats
                    csv_rows.append(
                        [
                            family,
                            m,
                            group,
                            algorithm,
                            stats["n"],
                            stats["correct"],
                            stats["invalid"],
                            stats["balanced_accuracy"],
                            stats["recall"]["head"],
                            stats["recall"]["non-head"],
                            stats["class_auroc"]["head"],
                        ]
                    )
                distributions[family][m][group] = {}
                for label in CLASSES:
                    selected = [i for i, y in zip(ids, labels, strict=True) if y == label]
                    fields = {
                        key: scalar_stats(z[key][selected])
                        for key in ("margin", "coverage", "delta", "relative")
                    }
                    fields.update(
                        {
                            key: scalar_stats([rows[i]["geometry"][m][key] for i in selected])
                            for key in (
                                "rank",
                                "singular_max",
                                "singular_min_retained",
                                "common_energy",
                                "contrast_energy",
                                "four_cross",
                            )
                        }
                    )
                    fields["above_raw_threshold"] = float(np.mean(z["margin"][selected] >= tau))
                    fields["above_relative_threshold"] = float(
                        np.mean(z["relative"][selected] >= rho)
                    )
                    distributions[family][m][group][label] = fields
                    for key in ("margin", "coverage", "delta", "relative"):
                        d = fields[key]
                        dist_rows.append(
                            [
                                family,
                                m,
                                group,
                                label,
                                key,
                                d["n"],
                                d["median"],
                                d["q25"],
                                d["q75"],
                                d["min"],
                                d["max"],
                            ]
                        )
            for dataset in ("slake", "rad"):
                natural, ys = groups[f"{dataset}_natural"]
                fixed, _ = groups[f"{dataset}_fixed"]
                fixed_by_run = {mapping[cases[i]["id"]]["run"]: i for i in fixed}
                pairs = [
                    (i, fixed_by_run[mapping[cases[i]["id"]]["run"]], y)
                    for i, y in zip(natural, ys, strict=True)
                ]
                paired[family][m][dataset] = {
                    label: {
                        key: paired_interval(
                            [
                                z[key][i] - z[key][j]
                                for i, j, y in pairs
                                if label == "all" or y == label
                            ]
                        )
                        for key in ("margin", "coverage", "delta", "relative")
                    }
                    for label in ("all", *CLASSES)
                }
    aggregate = {
        "primary": "Q24",
        "secondary": "Q32",
        "metrics": results,
        "distributions": distributions,
        "paired_natural_minus_fixed": paired,
        "rules": {f: predictions[f]["rules"] for f in predictions},
        "medical_bank_gate": read_json(root / "medical-bank-stopped.json"),
        "unique_target_images": 40,
        "new_target_images": 0,
        "gpu_used": False,
        "evidence": "Repeated-target mechanism diagnosis; no independent confirmation",
        "uncommitted": True,
    }
    write_json(root / "aggregate.json", aggregate)
    for name, columns, rows in (
        (
            "summary.csv",
            [
                "Model",
                "Module",
                "Group",
                "Algorithm",
                "N",
                "Correct",
                "Invalid",
                "Balanced accuracy",
                "Head recall",
                "Negative recall",
                "Head AUROC",
            ],
            csv_rows,
        ),
        (
            "distributions.csv",
            [
                "Model",
                "Module",
                "Group",
                "Class",
                "Variable",
                "N",
                "Median",
                "Q25",
                "Q75",
                "Min",
                "Max",
            ],
            dist_rows,
        ),
    ):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)
        (root / name).write_text(stream.getvalue())
    return aggregate


def audit(root, aggregate):
    plan = read_json(root / "plan.json")
    for p, expected in plan["hashes"].items():
        if file_hash(p) != expected:
            raise ValueError("Original frozen boundary input changed")
    for p, expected in read_json(root / "source-boundary-manifest.json")["hashes"].items():
        if file_hash(p) != expected:
            raise ValueError("Execution source changed")
    commit = read_json(root / "cpu/predictions-complete.json")
    if file_hash(root / "cpu/predictions.json") != commit["sha256"]:
        raise ValueError("Prediction commitment changed")
    if "source-boundary-v1" not in commit["runtime_source"]:
        raise ValueError("CPU did not run frozen worker")
    predictions = read_json(root / "cpu/predictions.json")
    old = read_json(PREVIOUS / "plan.json")
    public_labels = read_json(PUBLIC / "evaluation_labels.json")
    distances_checked, metrics_checked, direct_checked, max_error = 0, 0, 0, 0.0
    for family in ("llava", "qwen35"):
        cases, groups, _ = group_labels(family)
        rows = predictions[family]["rows"]
        previous = read_tensors(PREVIOUS / f"{family}-scores.safetensors")["scores"][
            :, old["full_bank"]
        ].numpy()
        for j, m in enumerate(METHODS):
            d = np.asarray([r["distances"][m] for r in rows])
            error = float(np.max(np.abs(d - previous[:, j])))
            if error > 1e-12:
                raise ValueError("Baseline distances changed")
            max_error = max(max_error, error)
            distances_checked += len(rows)
            raw = d[:, 1] - d[:, 0]
            energy = np.maximum(1 - d * d, 0)
            if np.max(np.abs((energy[:, 0] - energy[:, 1]) - raw * d.sum(1))) > 1e-12:
                raise ValueError("Margin-energy decomposition differs")
            valid = energy.sum(1) > 1e-12
            relative = np.divide(
                energy[:, 0] - energy[:, 1], energy.sum(1), out=np.zeros(len(d)), where=valid
            )
            rule = predictions[family]["rules"][m]
            index = {r["id"]: i for i, r in enumerate(cases)}
            cal = [index[r["id"]] for r in old["calibration_rows"]]
            labels = [public_labels[r["id"]] == "head" for r in old["calibration_rows"]]
            styles = [r["style"] for r in old["calibration_rows"]]
            vals = relative[cal]
            unique = np.unique(vals)
            thresholds = [
                0.0,
                float(np.nextafter(unique[0], -np.inf)),
                float(np.nextafter(unique[-1], np.inf)),
            ]
            thresholds += [
                float(a + (b - a) / 2) for a, b in zip(unique[:-1], unique[1:], strict=True)
            ]
            threshold = max(
                set(thresholds), key=lambda t: (*quality(vals, labels, styles, t)[:2], -abs(t), -t)
            )
            if threshold != rule["relative_rule"]["threshold"]:
                raise ValueError("Independent public relative threshold differs")
            for group, (ids, ys) in groups.items():
                for name, values, tau in (
                    ("raw_zero", raw, 0.0),
                    ("raw_public", raw, rule["raw_threshold"]),
                    ("relative_zero", relative, 0.0),
                    ("relative_public", relative, threshold),
                ):
                    y = np.array([v == "head" for v in ys])
                    keep = (
                        valid[ids] if name.startswith("relative") else np.ones(len(ids), dtype=bool)
                    )
                    head = values[ids] >= tau
                    correct = int((keep & (head == y)).sum())
                    ba = (np.mean((keep & head)[y]) + np.mean((keep & ~head)[~y])) / 2
                    actual = aggregate["metrics"][family][m][group][name]
                    if (
                        actual["correct"] != correct
                        or abs(actual["balanced_accuracy"] - ba) > 1e-12
                    ):
                        raise ValueError("Independent manual confusion counts differ")
                    if keep.all() and actual["class_auroc"]["head"] != manual_auc(y, values[ids]):
                        raise ValueError("Independent pairwise AUROC differs")
                    metrics_checked += 1
        presets = [0, 13, 26, 39]
        presets += [
            next(
                i
                for i, c in enumerate(cases)
                if c["kind"] == "medical" and c["dataset"] == ds and c["qa"] == qa
            )
            for ds in ("slake", "rad")
            for qa in ("natural", "fixed")
        ]
        protos = read_tensors(SOURCE / family / "public/prototypes.safetensors")
        for i in presets:
            gs = load_gradients(cases[i], family)
            for m in METHODS:
                _, s, vh = torch.linalg.svd(gs[m].double(), full_matrices=False)
                basis = vh[s > max(1e-10, float(s[0]) * 1e-6)]
                mu = protos[m].double()
                projected = mu @ basis.T
                d = (mu - projected @ basis).norm(dim=1).numpy()
                error = float(np.max(np.abs(d - np.asarray(rows[i]["distances"][m]))))
                p, v = (mu[0] + mu[1]) / 2, (mu[0] - mu[1]) / 2
                common = float((p @ basis.T).square().sum())
                contrast = float((v @ basis.T).square().sum())
                cross = 4 * float((p @ basis.T) @ (v @ basis.T))
                expected = rows[i]["geometry"][m]
                if (
                    error > 1e-12
                    or max(
                        abs(expected[k] - val)
                        for k, val in (
                            ("common_energy", common),
                            ("contrast_energy", contrast),
                            ("four_cross", cross),
                        )
                    )
                    > 1e-12
                ):
                    raise ValueError("Independent SVD geometry differs")
                max_error = max(max_error, error)
                direct_checked += 1
    return {
        "status": "pass",
        "full_bank_distances": distances_checked,
        "manual_metric_groups": metrics_checked,
        "independent_svd_geometry": direct_checked,
        "max_distance_error": max_error,
        "public_thresholds_reproduced": True,
        "target_labels_used_for_fit": False,
        "gpu_used": False,
        "medical_bank_stopped_before_capture": True,
    }


def plots(root, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = read_json(root / "cpu/predictions.json")
    names = ["public_audit", "slake_fixed", "slake_natural", "rad_fixed", "rad_natural"]
    labels = ["Public fixed", "SLAKE fixed", "SLAKE natural", "RAD fixed", "RAD natural"]
    figures = []
    for filename, mode in (("boundary-margins.png", "margin"), ("boundary-energy.png", "energy")):
        fig, axes = plt.subplots(2, 2, figsize=(16, 8.8), constrained_layout=True)
        for i, family in enumerate(("llava", "qwen35")):
            _, groups, _ = group_labels(family)
            rows = predictions[family]["rows"]
            for j in range(2):
                method = ("Q24", "Q32")[j] if mode == "margin" else "Q24"
                variable = "margin" if mode == "margin" else ("coverage", "relative")[j]
                ax = axes[i, j]
                z = decompose([r["distances"][method] for r in rows])
                for c, color in (("head", "#c76846"), ("non-head", "#317999")):
                    for k, group in enumerate(names):
                        ids, ys = groups[group]
                        values = np.asarray(
                            [z[variable][a] for a, y in zip(ids, ys, strict=True) if y == c]
                        )
                        x = k + (-0.13 if c == "head" else 0.13)
                        ax.scatter(
                            x + np.linspace(-0.045, 0.045, len(values)),
                            values,
                            s=17,
                            alpha=0.5,
                            color=color,
                            label=c if k == 0 else None,
                        )
                        q = np.quantile(values, [0.25, 0.5, 0.75])
                        ax.errorbar(
                            x,
                            q[1],
                            yerr=[[q[1] - q[0]], [q[2] - q[1]]],
                            fmt="_",
                            markersize=13,
                            capsize=4,
                            color=color,
                            linewidth=2,
                        )
                if variable in ("margin", "relative"):
                    rule = predictions[family]["rules"][method]
                    tau = (
                        rule["raw_threshold"]
                        if variable == "margin"
                        else rule["relative_rule"]["threshold"]
                    )
                    ax.axhline(
                        tau, color="#333333", linestyle="--", linewidth=1, label="Public threshold"
                    )
                ax.set_title(f"{family} · {method} · {variable}")
                ax.set_xticks(range(5), labels, rotation=15)
                ax.grid(axis="y", alpha=0.15)
                ax.legend(fontsize=8)
        path = root / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(path)
    return figures


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
        figures = plots(root, aggregate)
        repo = Path.cwd()
        data = repo / "docs/experiments/data/gradient-prototype-transfer"
        dest = repo / "docs/experiments/figures/gradient-prototype-transfer"
        data.mkdir(parents=True, exist_ok=True)
        dest.mkdir(parents=True, exist_ok=True)
        published = {}
        for name in ("aggregate.json", "summary.csv", "distributions.csv", "integrity-audit.json"):
            p = data / f"boundary-{name}"
            shutil.copy2(root / name, p)
            published[str(p.relative_to(repo))] = file_hash(p)
        for path in figures:
            p = dest / path.name
            shutil.copy2(path, p)
            published[str(p.relative_to(repo))] = file_hash(p)
        write_json(
            data / "boundary-manifest.json",
            {
                "artifacts": published,
                "execution_source_manifest_sha256": file_hash(
                    root / "source-boundary-manifest.json"
                ),
                "analysis_source_sha256": file_hash(Path(__file__)),
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
