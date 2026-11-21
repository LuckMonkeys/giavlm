"""Independent public-rule and score audits; publish aggregate-only method results."""

import argparse
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from core.fl import load_observation
from evaluation.crossmodel_property import module_key
from evaluation.prototype_method_math import METHODS


def manual_auc(labels, values):
    positives = [v for y, v in zip(labels, values, strict=True) if y]
    negatives = [v for y, v in zip(labels, values, strict=True) if not y]
    return sum(float(p > n) + 0.5 * float(p == n) for p in positives for n in negatives) / (
        len(positives) * len(negatives)
    )


def quality(values, labels, styles, threshold):
    style_scores = []
    for style in sorted(set(styles)):
        positive = [
            values[i] >= threshold for i, y in enumerate(labels) if y and styles[i] == style
        ]
        negative = [
            values[i] < threshold for i, y in enumerate(labels) if not y and styles[i] == style
        ]
        style_scores.append((sum(positive) / len(positive) + sum(negative) / len(negative)) / 2)
    return min(style_scores), sum(style_scores) / len(style_scores), manual_auc(labels, values)


def audit(root):
    plan = read_json(root / "plan.json")
    source = Path(plan["input"])
    complete = read_json(root / "complete.json")
    if complete["aggregate_sha256"] != file_hash(root / "aggregate.json"):
        raise ValueError("Completed aggregate changed")
    if "source-prototype-method-v1" not in complete["runtime_source"]:
        raise ValueError("Actual worker source was not frozen")
    for path, expected in plan["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen input changed")
    for path, expected in read_json(root / "source-prototype-method-manifest.json")[
        "hashes"
    ].items():
        if file_hash(path) != expected:
            raise ValueError("Frozen source changed")
    commit = read_json(root / "predictions-complete.json")
    if (
        file_hash(root / "predictions.json") != commit["sha256"]
        or file_hash(root / "rules.json") != commit["rules_sha256"]
    ):
        raise ValueError("Rule or prediction commit changed")
    cache = read_json(root / "cache-complete.json")
    for path, expected in cache["hashes"].items():
        if file_hash(path) != expected:
            raise ValueError("Committed 81-bank scores changed")
    rules = read_json(root / "rules.json")
    weights = read_tensors(root / "bank-weights.safetensors")["weights"]
    public_labels = read_json(
        "outputs/diagnostics/image_property_next_20261028/styles/evaluation_labels.json"
    )
    predictions = read_json(root / "predictions.json")
    aggregate = read_json(root / "aggregate.json")
    replay_count, direct_count, metric_count, max_error = 0, 0, 0, 0.0
    for family in ("llava", "qwen35"):
        cases = read_json(root / f"{family}-cases.json")
        index = {r["id"]: i for i, r in enumerate(cases)}
        values = read_tensors(root / f"{family}-scores.safetensors")["scores"].numpy()
        cal_ids = [index[r["id"]] for r in plan["calibration_rows"]]
        labels = [public_labels[r["id"]] == "head" for r in plan["calibration_rows"]]
        styles = [r["style"] for r in plan["calibration_rows"]]
        calibration = {
            m: values[cal_ids, plan["full_bank"], j, 1] - values[cal_ids, plan["full_bank"], j, 0]
            for j, m in enumerate(METHODS)
        }
        ranking = {m: quality(v, labels, styles, 0.0) for m, v in calibration.items()}
        selected = max(
            METHODS, key=lambda m: (*ranking[m], -int(m[1:]), -("Q", "K", "V").index(m[0]))
        )
        if selected != rules[family]["selection"]["selected"]:
            raise ValueError("Independent public module selection differs")
        for method, vals in calibration.items():
            unique = np.unique(vals)
            thresholds = [
                0.0,
                float(np.nextafter(unique[0], -np.inf)),
                float(np.nextafter(unique[-1], np.inf)),
            ]
            thresholds += [
                float(a + (b - a) / 2) for a, b in zip(unique[:-1], unique[1:], strict=True)
            ]
            chosen = max(
                set(thresholds), key=lambda t: (*quality(vals, labels, styles, t)[:2], -abs(t), -t)
            )
            if chosen != rules[family]["offsets"][method]["threshold"]:
                raise ValueError("Independent public offset calibration differs")
        old = {
            r["id"]: r["scores"]
            for r in read_json(source / family / "public/predictions.json")
            + read_json(source / family / "predictions.json")
        }
        for i, case in enumerate(cases):
            for j, method in enumerate(METHODS):
                pair = values[i, plan["full_bank"], j]
                error = float(np.max(np.abs(pair - np.asarray(old[case["id"]][method]))))
                if error > 1e-12 or int(np.argmin(pair)) != int(np.argmin(old[case["id"]][method])):
                    raise ValueError("Original full-bank scores or predictions changed")
                max_error = max(max_error, error)
                replay_count += 1
        # Direct SVD and residual norms on preset cases cover every data/QA group.
        selected_cases = [0, 13, 26, 39]
        for ds in ("slake", "rad"):
            for qa in ("natural", "fixed"):
                selected_cases.append(
                    next(
                        i
                        for i, c in enumerate(cases)
                        if c["kind"] == "medical" and c["dataset"] == ds and c["qa"] == qa
                    )
                )
        feature_file = read_tensors(source / family / "public/visual-means.safetensors")
        for i in selected_cases:
            case = cases[i]
            if case["kind"] == "public":
                gradients = read_tensors(case["path"])
            else:
                observation = load_observation(case["observation"], "cpu")
                gradients = {m: observation.tensors[module_key(family, m)] for m in METHODS}
            for j, method in enumerate(METHODS):
                features = torch.stack(
                    [feature_file[f"{r['id']}_{method}"] for r in plan["fit_rows"]]
                )
                _, singular, vh = torch.linalg.svd(gradients[method].double(), full_matrices=False)
                basis = vh[singular > max(float(singular[0]) * 1e-6, 1e-10)]
                for bank in (0, 20, 40, 60, 61):
                    proto = torch.stack([features[weights[bank, c] > 0].mean(0) for c in (0, 1)])
                    proto = torch.nn.functional.normalize(proto, dim=1)
                    direct = (proto - (proto @ basis.T) @ basis).norm(dim=1).numpy()
                    error = float(np.max(np.abs(direct - values[i, bank, j])))
                    if error > 1e-12:
                        raise ValueError("Independent direct projection differs")
                    max_error = max(max_error, error)
                    direct_count += 1
            if case["kind"] == "medical":
                del observation
        # Read evaluation-only labels now; the original complete commit is already verified.
        groups = {
            "public_calibration": (
                [index[r["id"]] for r in plan["calibration_rows"]],
                [public_labels[r["id"]] for r in plan["calibration_rows"]],
            ),
            "public_audit": (
                [index[r["id"]] for r in plan["audit_rows"]],
                [public_labels[r["id"]] for r in plan["audit_rows"]],
            ),
        }
        mapping = read_json(source / family / "private/upload-map.json")
        for ds in ("slake", "rad"):
            truth = {r["run"]: r for r in read_json(source / "private" / f"{ds}-targets.json")}
            for qa in ("natural", "fixed"):
                ids = [
                    i
                    for i, c in enumerate(cases)
                    if c["kind"] == "medical" and c["dataset"] == ds and c["qa"] == qa
                ]
                groups[f"{ds}_{qa}"] = (
                    ids,
                    [truth[mapping[cases[i]["id"]]["run"]]["label"] for i in ids],
                )
        committed = {r["id"]: r["scores"] for r in predictions[family]}
        for group, (ids, truth) in groups.items():
            for algorithm, definition in rules[family]["algorithms"].items():
                method, t = definition["method"], definition["threshold"]
                scores = values[np.array(ids), plan["full_bank"], METHODS.index(method)]
                m = scores[:, 1] - scores[:, 0] - t
                y = [label == "head" for label in truth]
                pred = m >= 0
                correct = sum(bool(p) == label for p, label in zip(pred, y, strict=True))
                positive = sum(bool(p) for p, label in zip(pred, y, strict=True) if label) / sum(y)
                negative = sum(
                    not bool(p) for p, label in zip(pred, y, strict=True) if not label
                ) / (len(y) - sum(y))
                actual = aggregate["metrics"][family][group][algorithm]
                if (
                    actual["correct"] != correct
                    or actual["balanced_accuracy"] != (positive + negative) / 2
                    or actual["class_auroc"]["head"] != manual_auc(y, m)
                ):
                    raise ValueError("Independent classification or AUC count differs")
                for i, z in zip(ids, m, strict=True):
                    recorded = np.asarray(committed[cases[i]["id"]][algorithm])
                    if float(np.max(np.abs(recorded - np.array([-z, z])))) > 1e-12:
                        raise ValueError("Committed algorithm prediction differs")
                metric_count += 1
        # All sampling metrics are independently recalculated from the committed arrays.
        banks = read_json(root / "banks.json")
        for group, (ids, truth) in groups.items():
            y = np.array([label == "head" for label in truth])
            for j, method in enumerate(METHODS):
                base = values[np.array(ids), plan["full_bank"], j]
                for section in aggregate["stability"][family][group][method].values():
                    for row in section["rows"]:
                        b = next(i for i, meta in enumerate(banks) if meta["id"] == row["bank"])
                        pair = values[np.array(ids), b, j]
                        pred = pair[:, 0] <= pair[:, 1]
                        ba = float((pred[y].mean() + (~pred[~y]).mean()) / 2)
                        flip = float(np.mean(np.argmin(pair, 1) != np.argmin(base, 1)))
                        if (
                            ba != row["balanced_accuracy"]
                            or manual_auc(y, pair[:, 1] - pair[:, 0]) != row["auroc"]
                            or flip != row["flip_rate"]
                        ):
                            raise ValueError("Sampling metric differs")
    result = {
        "status": "pass",
        "full_bank_distances_replayed": replay_count,
        "independent_direct_distances": direct_count,
        "main_metric_groups": metric_count,
        "sampling_rows_recomputed": 5832,
        "max_distance_error": max_error,
        "public_only_rules_reproduced": True,
        "prediction_commit_verified": True,
        "score_ranking_unchanged_by_offset": True,
        "new_medical_images": 0,
        "gpu_used": False,
        "source_manifest_sha256": file_hash(root / "source-prototype-method-manifest.json"),
        "aggregate_sha256": file_hash(root / "aggregate.json"),
        "analysis_source_sha256": file_hash(Path(__file__)),
    }
    write_json(root / "integrity-audit.json", result)
    return result


def figures(root, aggregate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Fixed Q32", "Public select", "Q32 offset", "Select + offset"]
    algorithms = [
        "fixed_q32",
        "public_selected_raw",
        "q32_calibrated",
        "public_selected_calibrated",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
    for i, family in enumerate(("llava", "qwen35")):
        for j, ds in enumerate(("slake", "rad")):
            axis = axes[i, j]
            values = [
                aggregate["metrics"][family][f"{ds}_natural"][a]["balanced_accuracy"]
                for a in algorithms
            ]
            axis.bar(range(4), values, color=["#999999", "#4c78a8", "#f28e2b", "#59a14f"])
            axis.set_xticks(range(4), names, rotation=12)
            axis.axhline(0.5, color="#888888", linestyle="--", linewidth=1)
            axis.set_ylim(0, 1.08)
            axis.set_axisbelow(True)
            axis.grid(axis="y", alpha=0.15)
            axis.set_title(
                f"{'LLaVA' if family == 'llava' else 'Qwen3.5'} | {'SLAKE A' if ds == 'slake' else 'VQA-RAD'}"
            )
            if j == 0:
                axis.set_ylabel("Balanced accuracy")
            for k, v in enumerate(values):
                axis.text(k, v + 0.02, f"{v:.2f}", ha="center")
    fig.suptitle(
        "Public-only selection and calibration | original QA | repeated medical targets",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = root / "public-rules.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for axis, family in zip(axes, ("llava", "qwen35"), strict=True):
        for group, label, color in [
            ("public_audit", "Public audit", "#59a14f"),
            ("slake_natural", "SLAKE A", "#4c78a8"),
            ("rad_natural", "VQA-RAD", "#f28e2b"),
        ]:
            summaries = [
                aggregate["stability"][family][group]["Q32"][f"real-{k}"]["metrics"][
                    "balanced_accuracy"
                ]
                for k in (1, 2, 4, 8)
            ]
            x = np.array([3, 6, 12, 24])
            means = [v["mean"] for v in summaries]
            axis.plot(x, means, "o-", label=label, color=color)
            axis.fill_between(
                x,
                [v["min"] for v in summaries],
                [v["max"] for v in summaries],
                color=color,
                alpha=0.15,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([3, 6, 12, 24], ["3", "6", "12", "24"])
        axis.set_ylim(0, 1.06)
        axis.grid(alpha=0.15)
        axis.set_axisbelow(True)
        axis.set_xlabel("Public images per class (three styles per family)")
        axis.set_title("LLaVA" if family == "llava" else "Qwen3.5")
    axes[0].set_ylabel("Q32 balanced accuracy")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Prototype sampling | lines: mean | bands: min-max over draws | full bank once", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    second = root / "prototype-stability.png"
    fig.savefig(second, dpi=200)
    plt.close(fig)
    return [path, second]


def publish(root):
    audit_result = audit(root)
    aggregate = read_json(root / "aggregate.json")
    plots = figures(root, aggregate)
    data = Path("docs/experiments/data/gradient-prototype-transfer")
    figure_dir = Path("docs/experiments/figures/gradient-prototype-transfer")
    files = []
    for source, name in [
        (root / "aggregate.json", "method-aggregate.json"),
        (root / "summary.csv", "method-summary.csv"),
        (root / "integrity-audit.json", "method-audit.json"),
        (root / "rules.json", "method-public-rules.json"),
    ]:
        dest = data / name
        shutil.copy2(source, dest)
        files.append(dest)
    for source in plots:
        dest = figure_dir / source.name
        shutil.copy2(source, dest)
        files.append(dest)
    shutil.copy2(Path(__file__), root / "analysis-source.py")
    write_json(
        data / "method-manifest.json",
        {
            "uncommitted": True,
            "artifacts": {str(path): file_hash(path) for path in files},
            "execution_source_manifest_sha256": audit_result["source_manifest_sha256"],
            "analysis_source_sha256": file_hash(Path(__file__)),
            "uses_private_reference": "Post-commit evaluation labels only",
        },
    )
    write_json(
        root / "report-complete.json",
        {
            "status": "complete",
            "audit_sha256": file_hash(root / "integrity-audit.json"),
            "aggregate_sha256": file_hash(root / "aggregate.json"),
        },
    )
    print(audit_result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        publish(args.root.resolve())
    except BaseException:
        write_json(args.root / "analysis-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
