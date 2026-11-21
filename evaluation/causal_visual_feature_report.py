"""Audit and summarize the public causal suffix check without loading a model."""

import argparse
import csv
import io
import math
from pathlib import Path
import shutil
import traceback

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.causal_visual_feature_check import VARIANTS, gradient_comparison


def summarize(values):
    return {"comparisons": len(values), "bit_equal": sum(v["bit_equal"] for v in values),
            "max_absolute_error": max(v["max_absolute_error"] for v in values),
            "min_relative_l2": min(v["relative_l2"] for v in values),
            "max_relative_l2": max(v["relative_l2"] for v in values)}


def report(root):
    out = root / "causal"
    full = read_json(out / "report.json")
    plan = read_json(out / "plan.json")
    complete = read_json(out / "complete.json")
    if complete["report_sha256"] != file_hash(out / "report.json"):
        raise ValueError("Report changed after completion")
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Model or public input changed")
    for path, sha in read_json(root / "source-causal-manifest.json")["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen source changed")
    rows = full["rows"]
    if len(rows) != len(plan["public_cases"]) * len(VARIANTS):
        raise ValueError("Missing comparisons")
    hashes, checked_gradients, checked_means = {}, 0, 0
    for index, case in enumerate(plan["public_cases"]):
        baseline = read_tensors(out / "vectors" / f"{index:02d}-baseline.safetensors")
        for variant in VARIANTS:
            row_path = out / "rows" / f"{index:02d}-{variant}.json"
            vector_path = out / "vectors" / f"{index:02d}-{variant}.safetensors"
            row = read_json(row_path)
            if rows[index * len(VARIANTS) + VARIANTS.index(variant)] != row or row["case"] != case["name"]:
                raise ValueError("Report and saved rows differ")
            values = read_tensors(vector_path)
            for layer, feature in row["features"].items():
                equal_hash = feature["sha256"] == feature["reference_sha256"]
                if equal_hash != feature["bit_equal"]:
                    raise ValueError("Feature equality and hashes disagree")
                error = float((values[f"mean_{layer}"] - baseline[f"mean_{layer}"]).abs().max())
                if error != feature["mean_max_absolute_error"]:
                    raise ValueError("Saved pooled feature differs")
                checked_means += 1
            for layer, comparison in row.get("gradients", {}).items():
                actual = gradient_comparison(values[f"gradient_{layer}"], baseline[f"gradient_{layer}"])
                for key, value in actual.items():
                    if isinstance(value, float):
                        if not math.isclose(value, comparison[key], rel_tol=1e-9, abs_tol=1e-10):
                            raise ValueError("Gradient metric recomputation differs")
                    elif value != comparison[key]:
                        raise ValueError("Gradient metric recomputation differs")
                checked_gradients += 1
            for path in (row_path, vector_path):
                hashes[str(path.relative_to(root))] = file_hash(path)
    features, gradients = {}, {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        features[variant] = summarize([v for row in selected for v in row["features"].values()])
        features[variant]["by_layer"] = {
            layer: summarize([row["features"][layer] for row in selected])
            for layer in selected[0]["features"]}
        if "gradients" in selected[0]:
            gradients[variant] = {}
            for layer in selected[0]["gradients"]:
                values = [row["gradients"][layer] for row in selected]
                gradients[variant][layer] = {
                    **summarize(values),
                    "min_rowspace_distance": min(v["rowspace_distance"] for v in values),
                    "max_rowspace_distance": max(v["rowspace_distance"] for v in values),
                    "ranks": sorted(set(v["rank"] for v in values)),
                    "reference_ranks": sorted(set(v["reference_rank"] for v in values))}
    hooks = [v for row in rows for v in row["lora_A_input_vs_q_proj_input"].values()]
    images = [v for row in rows for v in row.get("different_image_control", {}).values()]
    aggregate = {"public_images": full["public_images"], "feature_layers": full["feature_layers"],
                 "gradient_layers": full["gradient_layers"], "feature_comparisons": features,
                 "gradient_comparisons": gradients, "actual_lora_A_input_control": summarize(hooks),
                 "different_image_control": summarize(images),
                 "sequence_lengths": {variant: sorted(set(row["sequence_length"] for row in rows
                                                           if row["variant"] == variant)) for variant in VARIANTS},
                 "uses_private_reference": False, "runtime": complete}
    write_json(out / "aggregate.json", aggregate)
    write_json(out / "integrity-audit.json", {"status": "pass", "checked_pooled_features": checked_means,
               "recomputed_gradient_comparisons": checked_gradients, "row_and_vector_hashes": hashes,
               "plan_sha256": file_hash(out / "plan.json"),
               "source_manifest_sha256": file_hash(root / "source-causal-manifest.json"),
               "report_sha256": file_hash(out / "report.json"), "analysis_source_sha256": file_hash(Path(__file__))})
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Variant", "Comparisons", "Bit equal", "Max absolute error", "Max relative L2"])
    for variant, row in features.items():
        writer.writerow([variant, row["comparisons"], row["bit_equal"], row["max_absolute_error"], row["max_relative_l2"]])
    (out / "summary.csv").write_text(stream.getvalue())
    destination = Path("docs/experiments/data/visual-concept-probe")
    for source, target in [(out / "aggregate.json", "causal-feature-aggregate.json"),
                           (out / "integrity-audit.json", "causal-feature-audit.json"),
                           (out / "summary.csv", "causal-feature-summary.csv")]:
        shutil.copy2(source, destination / target)
    source_copy = out / "analysis-source.py"
    shutil.copy2(Path(__file__), source_copy)
    write_json(destination / "causal-feature-manifest.json", {
        "base_commit": read_json(root / "source-causal-manifest.json")["base_commit"],
        "uncommitted": True, "uses_private_reference": False,
        "artifacts": {str(path.resolve().relative_to(Path.cwd())): file_hash(path) for path in [destination / "causal-feature-aggregate.json",
                      destination / "causal-feature-audit.json", destination / "causal-feature-summary.csv", source_copy]}})
    print({"status": "pass", "pooled_features": checked_means, "gradient_comparisons": checked_gradients})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report(args.root.resolve())
    except BaseException:
        write_json(args.root / "causal/analysis-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
