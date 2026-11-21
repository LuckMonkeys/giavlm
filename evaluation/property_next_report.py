"""Aggregate the public next-stage diagnostics after predictions are committed."""
import argparse
from pathlib import Path

import numpy as np

from core.artifacts import file_hash, read_json, write_json
from evaluation.image_property_scoring import classification_stats


def paired_interval(labels, first, second, repeats=5000):
    y = np.asarray(labels) == "head"
    a = np.asarray(first).argmin(1) == (~y).astype(int)
    b = np.asarray(second).argmin(1) == (~y).astype(int)
    difference = a.astype(float) - b.astype(float)
    rng = np.random.default_rng(20261028)
    parts = [difference[rng.choice(np.flatnonzero(y == category),
             (repeats, int((y == category).sum())), replace=True)].mean(1) for category in [True, False]]
    return {"gain": float((difference[y].mean() + difference[~y].mean()) / 2),
            "ci95": np.quantile((parts[0] + parts[1]) / 2, [.025, .975]).tolist(),
            "unit": "independent public query"}


def paired_swap_interval(labels, first, second, mapping, repeats=5000):
    """Swapped outputs share two queries, so resample whole opposite-label pairs."""
    y = np.asarray(labels) == "head"
    difference = ((np.asarray(first).argmin(1) == (~y).astype(int)).astype(float) -
                  (np.asarray(second).argmin(1) == (~y).astype(int)).astype(float))
    pairs = [(i, j) for i, j in enumerate(mapping) if i < j]
    if len(pairs) * 2 != len(labels) or any(mapping[j] != i or y[i] == y[j] for i, j in pairs):
        raise ValueError("Swaps need disjoint opposite-property pairs")
    values = np.asarray([(difference[i] + difference[j]) / 2 for i, j in pairs])
    rng = np.random.default_rng(20261028)
    samples = values[rng.integers(0, len(pairs), size=(repeats, len(pairs)))].mean(1)
    return {"gain": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "unit": "opposite-property upload pair", "pairs": len(pairs)}


def summarize(root):
    sources = {}

    def load(relative):
        path = root / relative
        sources[str(path)] = file_hash(path)
        return read_json(path)

    result = {"scope": "Public controlled known-QA mechanism study, not real private-image recovery",
              "D0": load("d0/complete.json")}
    if (root / "d0-calibration/gate.json").exists():
        result["D0_calibration"] = load("d0-calibration/complete.json")
        result["D0_gate"] = load("d0-calibration/gate.json")
    if (root / "styles/development_complete.json").exists():
        done = load("styles/development_complete.json")
        report = load("styles/development_report.json")
        if done["report_sha256"] != sources[str(root / "styles/development_report.json")]:
            raise ValueError("Development report changed")
        result["public_development"] = report
    if (root / "convex/complete.json").exists():
        done = load("convex/complete.json")
        marker = load("convex/predictions_complete.json")
        for path, sha in marker["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Committed recovery changed")
        output = load("convex/readout_predictions.json")
        recovery_plan = load("convex/plan.json")
        if done["readout_sha256"] != sources[str(root / "convex/readout_predictions.json")]:
            raise ValueError("Readout changed")
        # Independent label join follows both recovery and readout commits.
        metadata = read_json(root / "styles/public_manifest.json")
        evaluation = read_json(root / "styles/evaluation_labels.json")
        order = load("evaluation/convex-input-order.json")
        query_metadata = [r for r in metadata if r["split"] == "query"]
        labels = [evaluation[query_metadata[i]["id"]] for i in order["original_capture_indices"]]
        scores = {method: [row["scores"][method] for row in output] for method in output[0]["scores"]}
        # A selected public prototype also has a verified public label. Report
        # this stronger candidate-selection baseline alongside its model readout.
        fit_labels = read_json(root / "styles/fit_labels.json")
        dict_indices = [i for i, row in enumerate(metadata) if row["split"] == "fit" and
                        row["style"] in ["photo", "line"]]
        traces = [read_json(root / "convex/predictions" / f"q{i:02d}.json") for i in range(12)]
        chosen_labels = [fit_labels[metadata[dict_indices[t["best_single"]]]["id"]] for t in traces]
        scores["best_single_public_label"] = [[0., 1.] if label == "head" else [1., 0.] for label in chosen_labels]
        # All queries share identical known training QA and the same initialization
        # seeds, so swapped-upload inference is exactly a fixed reindexing.
        wrong = order["wrong_gradient_indices"]
        scores["wrong_gradient"] = [scores["optimized"][i] for i in wrong]
        stats = {method: classification_stats(labels, value, ["head", "non-head"])
                 for method, value in scores.items()}
        differences = {method: paired_interval(labels, scores["optimized"], scores[method])
                       for method in ["mean", "random_initial_mean", "best_single",
                                      "best_single_public_label"]}
        differences["wrong_gradient"] = paired_swap_interval(labels, scores["optimized"],
                                                              scores["wrong_gradient"], wrong)
        conflict = [len({np.argmin(row["optimized_scores"][i]) for i in row["near_optimal"]}) > 1
                    for row in output]
        candidate_conflict = [len({np.argmin(s) for s in row["optimized_scores"]}) > 1 for row in output]
        result["convex"] = {"metrics": stats, "paired_gains": differences,
                            "near_optimal_conflict_rate": float(np.mean(conflict)),
                            "all_initialization_conflict_rate": float(np.mean(candidate_conflict)),
                            "mean_selected_matching": {key: float(np.mean([t["selected_matching"][key] for t in traces]))
                                                       for key in traces[0]["selected_matching"]},
                            "heldout_better_than_best_single_fraction": float(np.mean([
                                t["selected_matching"]["heldout"] < t["single_errors"][t["best_single"]]["heldout"]
                                for t in traces])),
                            "near_optimal_counts": [len(row["near_optimal"]) for row in output],
                            "cost": {**done, "gradient_replays": len(output) * (
                                len(recovery_plan["dictionary"]) + 2 +
                                len(recovery_plan["seeds"]) * (recovery_plan["steps"] + 1))},
                            "queries": len(output), "seeds": len(recovery_plan["seeds"]),
                            "heldout_selection_rule": "Minimum 89-block matching error, including step zero",
                            "ambiguity_rule": "Best per-seed solutions within 5% of best heldout loss plus 1e-6",
                            "wrong_gradient": "Same controlled QA, opposite-property same-render upload exchange; inference reused by exact mapping"}
    result["sources"] = sources
    result["analysis_source_sha256"] = file_hash(Path(__file__))
    result["incomplete_routes"] = ["C text-guided generation", "D1c PCA", "D2 free tokens",
                                   "Independent real targets with reconstructed natural QA"]
    write_json(root / "aggregate.json", result)
    public = Path("docs/experiments/data/visual-concept-probe/next-stage-aggregate.json")
    write_json(public, result)
    write_json(root / "report_complete.json", {"aggregate_sha256": file_hash(root / "aggregate.json"),
               "public_sha256": file_hash(public), "scope": result["scope"]})
    return result


def draft(root, result):
    lines = ["# 下一阶段公开机制检查", "", result["scope"], ""]
    if "public_development" in result:
        report = result["public_development"]
        lines += ["## 跨风格评分", "", "| Heldout | Method | Balanced accuracy | Head AUROC |",
                  "|---|---|---:|---:|"]
        for style, methods in report["folds"].items():
            for method, stats in methods.items():
                lines.append(f"| {style} | {method} | {stats['balanced_accuracy']:.3f} | {stats['class_auroc']['head']:.3f} |")
        lines += ["", f"公开开发选择：`{report['selected']}`。只有通过预定门槛的配置才进入独立确认。", ""]
        for key, title in [("layer_readout", "深层前向读出"), ("prototype_probes", "同模态原型与梯度探针")]:
            lines += [f"## {title}", "", "| Heldout | Method | Balanced accuracy | Head AUROC |",
                      "|---|---|---:|---:|"]
            for style, methods in report[key].items():
                for method, stats in methods.items():
                    lines.append(f"| {style} | {method} | {stats['balanced_accuracy']:.3f} | {stats['class_auroc']['head']:.3f} |")
            lines += [""]
    if "convex" in result:
        report = result["convex"]
        lines += ["## 凸组合恢复", "", "| Method | Balanced accuracy | Head AUROC | Head recall |",
                  "|---|---:|---:|---:|"]
        for method, stats in report["metrics"].items():
            lines.append(f"| {method} | {stats['balanced_accuracy']:.3f} | {stats['class_auroc']['head']:.3f} | {stats['recall']['head']:.3f} |")
        lines += ["", "| Baseline | Paired gain | CI95 |", "|---|---:|---|"]
        for method, values in report["paired_gains"].items():
            lines.append(f"| {method} | {values['gain']:.3f} | {values['ci95']} |")
        lines += ["", f"近最优解属性冲突率：{report['near_optimal_conflict_rate']:.3f}。", ""]
    lines += ["## 边界", "", "- 只用公开合成图像和受控已知问答；不代表医学目标或恢复文本下的准确率。",
              "- 三个风格共享生成器和内容族；开发三折不是独立确认。",
              "- 原始生成和一次质量修订完整保留；最终负类为肢体为主。",
              "- 前向读出、梯度原型距离和从上传恢复 tokens 是不同证据。"]
    (root / "report-draft.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    draft(args.root, summarize(args.root))


if __name__ == "__main__":
    main()
