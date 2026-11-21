"""Recompute public QKV control metrics and publish aggregate-only evidence."""

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
from evaluation.image_property_scoring import classification_stats
from evaluation.multilayer_property_gpu import CLASSES


def summary(values):
    return {"n": len(values), "min": float(np.min(values)), "mean": float(np.mean(values)), "max": float(np.max(values))}


def stats(labels, scores, classes):
    values = np.asarray(scores)
    margin = values[:, 1] - values[:, 0]
    positive = np.asarray(labels) == classes[0]
    return {**classification_stats(labels, scores, classes),
            "positive_margin_mean": float(margin[positive].mean()),
            "negative_margin_mean": float(margin[~positive].mean()),
            "class_mean_gap": float(margin[positive].mean() - margin[~positive].mean())}


def paired(labels, first, second, classes, groups):
    """Keep public content families together; styles/seeds are correlated repeats."""
    a = np.asarray(first).argmin(1) == np.asarray([classes.index(label) for label in labels])
    b = np.asarray(second).argmin(1) == np.asarray([classes.index(label) for label in labels])
    label_array, group_array = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(20261101)
    per_class = []
    for label in classes:
        families = sorted(set(group_array[label_array == label]))
        family_gains = np.asarray([(a[group_array == family].astype(float) - b[group_array == family].astype(float)).mean()
                                   for family in families])
        per_class.append(family_gains[rng.integers(len(families), size=(2000, len(families)))].mean(1))
    interval = np.quantile(np.mean(per_class, axis=0), [.025, .975]).tolist()
    return {"first_only_correct": int((a & ~b).sum()), "second_only_correct": int((b & ~a).sum()),
            "both_correct": int((a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "prediction_disagreement": int((np.asarray(first).argmin(1) != np.asarray(second).argmin(1)).sum()),
            "class_equal_content_family_gain_ci95": interval,
            "bootstrap": "Stratified class-equal content-family bootstrap; descriptive, not an independent validation",
            "families": len(set(groups))}


def report(root):
    torch.set_num_threads(1)
    out = root / "qa-controls"
    plan, complete = read_json(out / "plan.json"), read_json(out / "complete.json")
    commit = read_json(out / "predictions-complete.json")
    if file_hash(out / "predictions.json") != commit["sha256"] or complete["predictions_sha256"] != commit["sha256"]:
        raise ValueError("Control prediction commit changed")
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen input changed")
    for path, sha in commit["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Recorded control artifact changed")
    for path, sha in read_json(root / "source-qa-controls-manifest.json")["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen experiment source changed")
    if complete["source_sha256"] != plan["source_sha256"] or "source-qa-controls-v1" not in complete["runtime_source"]:
        raise ValueError("Actual runtime source must be the frozen worker")
    bank, rows = read_json(out / "public-bank.json"), read_json(out / "predictions.json")
    expected = {(image, case["id"]) for image in range(len(bank)) for case in plan["cases"]}
    if {(row["image"], row["case"]) for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicated controls")
    prototype = {"head": read_tensors(plan["head_prototypes"]),
                 "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    checked, old_checked, score_error, add_error, pooled_checked = 0, 0, 0., 0., 0
    pooled = {}
    token_variants = []
    for row in rows:
        public = bank[row["image"]]
        tensors = read_tensors(out / "gradients" / f"{row['image']:03d}-{row['case']}.safetensors")
        for layer in plan["layers"]:
            key = (row["image"], layer)
            if key not in pooled:
                pooled[key] = tensors[f"mean_{layer}"]
            if not torch.equal(pooled[key], tensors[f"mean_{layer}"]):
                raise ValueError("Pooled visual features changed across fixed-slot QA")
            pooled_checked += 1
        for module, scores in row["scores"].items():
            layer = int(module.split("_")[0][1:])
            for prop in scores:
                actual = distance_to_gradient(prototype[prop][f"visual_{layer}"], tensors[module])
                error = float(np.max(np.abs(np.asarray(actual) - scores[prop])))
                score_error = max(score_error, error)
                if error > 1e-12:
                    raise ValueError("Saved control distance differs")
                checked += 1
        if "old_q_gradient_errors" in row:
            case = next(case for case in plan["cases"] if case["id"] == row["case"])
            old = read_tensors(root / "multi-qa/gradients" / f"{public['old_image']:03d}-{case['old_qa']}.safetensors")
            for layer in plan["layers"]:
                if not torch.equal(tensors[f"L{layer}_q"], old[f"gradient_{layer}"]):
                    raise ValueError("Original Q gradient equality audit failed")
                old_checked += 1
        if row["token_diagnostic"]:
            tokens = read_tensors(out / "token-gradients" / f"{row['image']:03d}-{row['case']}.safetensors")
            for module, recorded in row["token_additivity"].items():
                summed = sum(tokens[f"token_{i:02d}_{module}"].double() for i in range(row["supervised_tokens"]))
                original = tensors[module].double()
                actual = float((summed - original).norm() / original.norm().clamp_min(1e-30))
                add_error = max(add_error, abs(actual - recorded["relative_l2"]))
                if abs(actual - recorded["relative_l2"]) > 1e-12:
                    raise ValueError("Token additivity diagnostic differs")
                indices = {"token_sum": list(range(row["supervised_tokens"])),
                           "without_eos": list(range(row["supervised_tokens"] - 1)),
                           "first_content": [0], "eos_only": [row["supervised_tokens"] - 1],
                           "content_words": [token["index"] for token in row["token_diagnostic"] if token["group"] == "content"]}
                layer = int(module.split("_")[0][1:])
                for variant, ids in indices.items():
                    gradient = sum(tokens[f"token_{i:02d}_{module}"].double() for i in ids)
                    token_variants.append({"image": row["image"], "case": row["case"], "module": module, "variant": variant,
                        "scores": {prop: distance_to_gradient(prototype[prop][f"visual_{layer}"], gradient)
                                   for prop in public["labels"]}})
    metrics, contrasts, robustness = {}, {}, {}
    old_cases = [case["id"] for case in plan["cases"] if case["old_qa"]]
    for prop, classes in CLASSES.items():
        ids = [i for i, public in enumerate(bank) if prop in public["labels"]]
        labels = [bank[i]["labels"][prop] for i in ids]
        groups = [bank[i]["group"] for i in ids]
        case_rows = {case["id"]: {row["image"]: row for row in rows if row["case"] == case["id"]} for case in plan["cases"]}
        metrics[prop], contrasts[prop], robustness[prop] = {}, {}, {}
        for layer in plan["layers"]:
            for letter in plan["modules"]:
                module = f"L{layer}_{letter}"
                by_case = {name: [entries[i]["scores"][module][prop] for i in ids] for name, entries in case_rows.items()}
                metrics[prop][module] = {name: stats(labels, values, classes) for name, values in by_case.items()}
                comparisons = [("long_yes_vs_yes", "present_long_yes", "present"),
                               ("sentence_vs_yes", "present_description", "present"),
                               ("question_present_vs_description", "present_description", "description"),
                               ("question_examine_vs_description", "examine_description", "description")]
                contrasts[prop][module] = {name: {
                    "balanced_accuracy_gain": metrics[prop][module][first]["balanced_accuracy"] - metrics[prop][module][second]["balanced_accuracy"],
                    **paired(labels, by_case[first], by_case[second], classes, groups)} for name, first, second in comparisons}
                if letter != "q":
                    q_module = f"L{layer}_q"
                    for case in ("present", "present_long_yes", "description"):
                        baseline = [case_rows[case][i]["scores"][q_module][prop] for i in ids]
                        contrasts[prop][module][f"vs_q_{case}"] = paired(labels, by_case[case], baseline, classes, groups)
                predictions = np.stack([np.asarray(by_case[name]).argmin(1) for name in old_cases])
                robustness[prop][module] = {"old_six_qa_worst_balanced_accuracy": min(metrics[prop][module][name]["balanced_accuracy"] for name in old_cases),
                    "old_six_qa_mean_balanced_accuracy": float(np.mean([metrics[prop][module][name]["balanced_accuracy"] for name in old_cases])),
                    "old_six_qa_flip_images": int((predictions.min(0) != predictions.max(0)).sum()), "images": len(ids)}
    support, spectra = {}, {}
    for case in plan["cases"]:
        selected = [row for row in rows if row["case"] == case["id"]]
        support[case["id"]] = {module: {key: summary([row["loss_support"][module][key] for row in selected])
                              for key in selected[0]["loss_support"][module]} for module in selected[0]["loss_support"]}
        spectra[case["id"]] = {module: {key: summary([row["spectra"][module][key] for row in selected])
                              for key in selected[0]["spectra"][module]} for module in selected[0]["spectra"]}
    token_summary, token_metrics = {}, {}
    for case in plan["cases"]:
        selected = [row for row in rows if row["case"] == case["id"] and row["token_diagnostic"]]
        if not selected:
            continue
        token_summary[case["id"]] = {}
        for module in selected[0]["token_additivity"]:
            tokens_by_position = []
            for index in range(selected[0]["supervised_tokens"]):
                values = [row["token_diagnostic"][index] for row in selected]
                tokens_by_position.append({"index": index, "text": values[0]["token_text"], "group": values[0]["group"],
                    "nll": summary([value["nll"] for value in values]),
                    "gradient_norm": summary([value["gradients"][module]["norm"] for value in values]),
                    "norm_relative_to_full": summary([value["gradients"][module]["norm"] / max(row["spectra"][module]["norm"], 1e-30)
                                                     for value, row in zip(values, selected, strict=True)]),
                    "cosine_to_full": summary([value["gradients"][module]["cosine_to_full"] for value in values])})
            token_summary[case["id"]][module] = {"images": len(selected), "tokens": tokens_by_position,
                "token_sum_relative_l2": summary([row["token_additivity"][module]["relative_l2"] for row in selected])}
        token_metrics[case["id"]] = {}
        for prop, classes in CLASSES.items():
            token_metrics[case["id"]][prop] = {}
            for module in selected[0]["token_additivity"]:
                token_metrics[case["id"]][prop][module] = {}
                for variant in ("token_sum", "without_eos", "first_content", "eos_only", "content_words"):
                    subset = [row for row in token_variants if row["case"] == case["id"] and row["module"] == module
                              and row["variant"] == variant and prop in row["scores"]]
                    labels = [bank[row["image"]]["labels"][prop] for row in subset]
                    token_metrics[case["id"]][prop][module][variant] = stats(labels, [row["scores"][prop] for row in subset], classes)
                native_rows = [row for row in selected if prop in bank[row["image"]]["labels"]]
                native_labels = [bank[row["image"]]["labels"][prop] for row in native_rows]
                native_scores = [row["scores"][module][prop] for row in native_rows]
                token_metrics[case["id"]][prop][module]["native_full"] = stats(native_labels, native_scores, classes)
                summed = {row["image"]: row["scores"][prop] for row in token_variants if row["case"] == case["id"]
                          and row["module"] == module and row["variant"] == "token_sum" and prop in row["scores"]}
                token_metrics[case["id"]][prop][module]["token_sum_vs_native_decision_disagreements"] = sum(
                    int(np.argmin(row["scores"][module][prop]) != np.argmin(summed[row["image"]])) for row in native_rows)
    aggregate = {"scope": plan["scope"], "public_images": len(bank), "cases": plan["cases"],
                 "primary_contrast": plan["primary_contrast"], "question_contrast": plan["question_contrast"],
                 "metrics": metrics, "paired_contrasts": contrasts, "robustness": robustness,
                 "loss_support": support, "spectra": spectra, "token_loss": token_summary,
                 "token_loss_subset_metrics": token_metrics,
                 "token_loss_subset_scope": "Six public images only; full token gradients are divided by original K. Changing loss and using known token positions is a mechanism diagnostic, not an attack available from private uploads.",
                 "token_images": len(plan["token_images"]), "runtime": complete,
                 "uses_private_reference": False, "interpretation": plan["interpretation"],
                 "token_additivity_policy": plan["token_additivity_policy"], "predictions_sha256": commit["sha256"]}
    write_json(out / "aggregate.json", aggregate)
    audit = {"status": "pass", "conditions": len(rows), "distances_recomputed": checked,
             "max_distance_error": score_error, "old_q_gradients_exact": old_checked,
             "token_additivity_metric_error": add_error, "frozen_actual_runtime_source": complete["runtime_source"],
             "pooled_visual_feature_comparisons_equal": pooled_checked,
             "plan_sha256": file_hash(out / "plan.json"), "predictions_sha256": commit["sha256"],
             "source_manifest_sha256": file_hash(root / "source-qa-controls-manifest.json"),
             "analysis_source_sha256": file_hash(Path(__file__))}
    write_json(out / "integrity-audit.json", audit)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Property", "Layer", "Module", "QA", "Correct", "Images", "Balanced accuracy", "Positive AUROC"])
    for prop, modules in metrics.items():
        for module, by_case in modules.items():
            layer, letter = module.split("_")
            for case, values in by_case.items():
                writer.writerow([prop, layer[1:], letter, case, values["correct"], values["n"], values["balanced_accuracy"], values["class_auroc"][CLASSES[prop][0]]])
    (out / "summary.csv").write_text(stream.getvalue())
    destination = Path("docs/experiments/data/visual-concept-probe")
    targets = []
    for source, name in [(out / "aggregate.json", "qa-controls-aggregate.json"),
                         (out / "integrity-audit.json", "qa-controls-audit.json"), (out / "summary.csv", "qa-controls-summary.csv")]:
        path = destination / name
        shutil.copy2(source, path)
        targets.append(path)
    figure = plot(aggregate)
    targets.append(figure)
    source_copy = out / "analysis-source.py"
    shutil.copy2(Path(__file__), source_copy)
    targets.append(source_copy)
    write_json(destination / "qa-controls-manifest.json", {"uses_private_reference": False, "uncommitted": True,
        "artifacts": {str(path.resolve().relative_to(Path.cwd())): file_hash(path) for path in targets},
        "source_manifest_sha256": audit["source_manifest_sha256"]})
    print(audit)


def plot(aggregate):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    question_labels = ["Describe", "Image present?", "Can examine?"]
    answer_labels = ["Sentence", "Yes.", "Yes, ... shown."]
    names = [["description", "description_yes", "description_long_yes"],
             ["present_description", "present", "present_long_yes"],
             ["examine_description", "examine", "examine_long_yes"]]
    for i, prop in enumerate(CLASSES):
        for j, letter in enumerate(("q", "k", "v")):
            axis = axes[i, j]
            data = aggregate["metrics"][prop][f"L32_{letter}"]
            values = np.asarray([[data[name]["balanced_accuracy"] for name in row] for row in names])
            axis.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
            axis.set_title(f"{prop.capitalize()} | Layer 32 {letter.upper()}")
            axis.set_xticks(range(3), answer_labels, rotation=15, ha="right", fontsize=9)
            axis.set_yticks(range(3), question_labels, fontsize=9)
            for a in range(3):
                for b in range(3):
                    axis.text(b, a, f"{values[a,b]:.2f}", ha="center", va="center",
                              color="white" if values[a,b] > .65 else "black", fontsize=13)
    figure.suptitle("Public QA controls | balanced accuracy | fixed prototypes, direct comparison", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .95))
    path = Path("docs/experiments/figures/visual-concept-probe/qa-controls.png")
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report(args.root.resolve())
    except BaseException:
        write_json(args.root / "qa-controls/analysis-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
