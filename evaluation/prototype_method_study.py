"""CPU mechanism study: bank sampling, public selection and public offsets."""

import argparse
import csv
import io
from pathlib import Path
import socket
import os
import time
import traceback

import numpy as np
import torch

from attacks.analytic.dager_subspace import row_basis
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation
from evaluation.crossmodel_property import module_key, validate_observation
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.prototype_method_math import (
    METHODS,
    SIZES,
    bank_geometry,
    build_banks,
    calibrated_scores,
    calibrate_offset,
    margin,
    projected_distances,
    select_module,
    stability_bounds,
)


INPUT = Path("outputs/diagnostics/crossmodel_property_20261104_v3")
PUBLIC = Path("outputs/diagnostics/image_property_next_20261028/styles")
CLASSES = ["head", "non-head"]
ALGORITHMS = ("fixed_q32", "public_selected_raw", "q32_calibrated", "public_selected_calibrated")


def prepare(root, source):
    if (root / "plan.json").exists():
        raise FileExistsError("Method plan already frozen")
    if read_json(source / "finalized.json")["status"] != "complete":
        raise ValueError("Only finalized cross-model inputs allowed")
    rows = read_json(source / "public-bank.json")
    fit = [r for r in rows if r["split"] == "fit"]
    calibration = [
        r for r in rows if r["split"] == "dev" and int(r["group"].rsplit("-", 1)[1]) < 12
    ]
    audit = [r for r in rows if r["split"] == "dev" and int(r["group"].rsplit("-", 1)[1]) >= 12]
    partitions = [set(r["group"] for r in group) for group in (fit, calibration, audit)]
    if (len(fit), len(calibration), len(audit)) != (48, 24, 24) or any(
        partitions[i] & partitions[j] for i in range(3) for j in range(i)
    ):
        raise ValueError("Content-family split differs from plan")
    fit_labels = read_json(PUBLIC / "fit_labels.json")
    weights, banks = build_banks(fit, fit_labels)
    write_tensors(root / "bank-weights.safetensors", {"weights": weights})
    write_json(root / "banks.json", banks)
    files = [
        source / "finalized.json",
        source / "plan.json",
        source / "public-bank.json",
        PUBLIC / "fit_labels.json",
        PUBLIC / "evaluation_labels.json",
        root / "bank-weights.safetensors",
        root / "banks.json",
        source / "private/slake-targets.json",
        source / "private/rad-targets.json",
    ]
    for family in ("llava", "qwen35"):
        out = source / family
        files += [
            out / name
            for name in (
                "runtime.json",
                "capture-complete.json",
                "upload-list.json",
                "predictions.json",
                "predictions-complete.json",
            )
        ]
        files.append(out / "private/upload-map.json")
        files += [
            out / "public" / name
            for name in (
                "visual-means.safetensors",
                "prototypes.safetensors",
                "gradient-list.json",
                "predictions.json",
                "predictions-complete.json",
            )
        ]
        files += [Path(row["path"]) for row in read_json(out / "public/gradient-list.json")]
    write_json(
        root / "plan.json",
        {
            "input": str(source.resolve()),
            "methods": list(METHODS),
            "primary": "public_selected_calibrated",
            "algorithms": list(ALGORITHMS),
            "seed": 20261105,
            "bank_count": len(banks),
            "real_banks": 61,
            "null_banks": 20,
            "full_bank": next(i for i, b in enumerate(banks) if b["id"] == "real-08-00"),
            "fit_rows": fit,
            "calibration_rows": calibration,
            "audit_rows": audit,
            "independent_medical_confirmation": False,
            "new_medical_images": 0,
            "gpu_used": False,
            "target_fitting": False,
            "original_score": "distance_nonhead - distance_head",
            "hashes": {str(p.resolve()): file_hash(p) for p in files},
            "execution_source_sha256": file_hash(Path(__file__)),
            "math_source_sha256": file_hash(Path(__file__).with_name("prototype_method_math.py")),
        },
    )


def verify_inputs(plan):
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen input changed")


def build_cache(root, plan):
    source = Path(plan["input"])
    weights = read_tensors(root / "bank-weights.safetensors")["weights"]
    full = plan["full_bank"]
    max_error, bound_error, total, direct_error = 0.0, 0.0, 0, 0.0
    checked_banks = [0, 20, 40, full, 61]
    for family in ("llava", "qwen35"):
        out = source / family
        values = read_tensors(out / "public/visual-means.safetensors")
        geometry = {}
        eps = {}
        for method in METHODS:
            features = torch.stack([values[f"{r['id']}_{method}"] for r in plan["fit_rows"]])
            unit, norm2 = bank_geometry(features, weights)
            geometry[method] = (features, unit, norm2)
            eps[method] = stability_bounds(unit, full)
        cases = [
            {"id": r["id"], "kind": "public", "path": r["path"]}
            for r in read_json(out / "public/gradient-list.json")
        ]
        cases += [{**r, "kind": "medical"} for r in read_json(out / "upload-list.json")]
        runtime = read_json(out / "runtime.json")
        expected = read_json(out / "capture-complete.json")["hashes"]
        previous = {
            r["id"]: r["scores"]
            for r in read_json(out / "public/predictions.json")
            + read_json(out / "predictions.json")
        }
        scores = torch.empty((len(cases), len(weights), len(METHODS), 2), dtype=torch.float64)
        ranks = torch.zeros((len(cases), len(METHODS)), dtype=torch.int64)
        for i, case in enumerate(cases):
            if case["kind"] == "public":
                gradients = read_tensors(case["path"])
            else:
                folder = Path(case["observation"])
                meta = read_json(folder / "observation.json")
                if (
                    file_hash(folder / "observation.json")
                    != expected[str(folder / "observation.json")]
                    or meta["update_sha256"] != expected[str(folder / "update.safetensors")]
                ):
                    raise ValueError("Original observation metadata differs")
                observation = load_observation(folder, "cpu")
                validate_observation(observation, runtime["fingerprint"])
                gradients = {m: observation.tensors[module_key(family, m)] for m in METHODS}
            for j, method in enumerate(METHODS):
                features, unit, norm2 = geometry[method]
                basis, _ = row_basis(gradients[method].double(), 1e-6, 1e-10)
                scores[i, :, j] = projected_distances(features @ basis.T, weights, norm2)
                selected_units = unit.reshape(len(weights), 2, -1)[checked_banks]
                direct = (selected_units - (selected_units @ basis.T) @ basis).norm(dim=-1)
                direct_difference = float((direct - scores[i, checked_banks, j]).abs().max())
                if direct_difference > 1e-12:
                    raise ValueError("Compressed and direct projection distances differ")
                direct_error = max(direct_error, direct_difference)
                ranks[i, j] = len(basis)
                error = float(
                    np.max(
                        np.abs(
                            np.asarray(previous[case["id"]][method]) - scores[i, full, j].numpy()
                        )
                    )
                )
                if error > 1e-12:
                    raise ValueError("Full-bank original score not reproduced")
                changes = torch.abs(
                    (scores[i, :, j, 1] - scores[i, :, j, 0])
                    - (scores[i, full, j, 1] - scores[i, full, j, 0])
                )
                violation = float((changes - eps[method]).max())
                if violation > 1e-12:
                    raise ValueError("Prototype distance stability inequality violated")
                max_error, bound_error = max(max_error, error), max(bound_error, violation)
                total += 1
            if case["kind"] == "medical":
                del observation
            del gradients
            if i % 16 == 0:
                write_json(
                    root / "progress.json",
                    {"stage": "cache", "model": family, "completed": i + 1, "total": len(cases)},
                )
                print(f"{family} cache {i + 1}/{len(cases)}", flush=True)
        write_tensors(
            root / f"{family}-scores.safetensors",
            {"scores": scores, "ranks": ranks, **{f"epsilon_{m}": v for m, v in eps.items()}},
        )
        write_json(root / f"{family}-cases.json", cases)
    write_json(
        root / "cache-complete.json",
        {
            "full_bank_replay_max_error": max_error,
            "max_stability_bound_violation": bound_error,
            "gradient_module_cases": total,
            "direct_projection_banks_per_case": checked_banks,
            "direct_projection_distances_checked": total * len(checked_banks),
            "direct_projection_max_error": direct_error,
            "hashes": {
                str(root / f"{f}-{suffix}"): file_hash(root / f"{f}-{suffix}")
                for f in ("llava", "qwen35")
                for suffix in ("scores.safetensors", "cases.json")
            },
        },
    )


def fit_and_commit(root, plan):
    all_labels = read_json(PUBLIC / "evaluation_labels.json")
    cal_rows = plan["calibration_rows"]
    cal_y = [all_labels[r["id"]] == "head" for r in cal_rows]
    styles = [r["style"] for r in cal_rows]
    rules, predictions = {}, {}
    full = plan["full_bank"]
    for family in ("llava", "qwen35"):
        cases = read_json(root / f"{family}-cases.json")
        scores = read_tensors(root / f"{family}-scores.safetensors")["scores"].numpy()
        index = {r["id"]: i for i, r in enumerate(cases)}
        cal = {
            m: scores[[index[r["id"]] for r in cal_rows], full, j].tolist()
            for j, m in enumerate(METHODS)
        }
        selection = select_module(cal, cal_y, styles)
        offsets = {m: calibrate_offset(margin(cal[m]), cal_y, styles) for m in METHODS}
        selected = selection["selected"]
        definitions = {
            "fixed_q32": ("Q32", 0.0),
            "public_selected_raw": (selected, 0.0),
            "q32_calibrated": ("Q32", offsets["Q32"]["threshold"]),
            "public_selected_calibrated": (selected, offsets[selected]["threshold"]),
        }
        rules[family] = {
            "selection": selection,
            "offsets": offsets,
            "algorithms": {k: {"method": m, "threshold": t} for k, (m, t) in definitions.items()},
            "public_audit_used": False,
            "medical_data_used": False,
        }
        predictions[family] = [
            {
                "id": case["id"],
                "kind": case["kind"],
                "scores": {
                    name: calibrated_scores(scores[i, full, METHODS.index(method)], threshold)
                    for name, (method, threshold) in definitions.items()
                },
            }
            for i, case in enumerate(cases)
        ]
    write_json(root / "rules.json", rules)
    write_json(root / "predictions.json", predictions)
    write_json(
        root / "predictions-complete.json",
        {
            "sha256": file_hash(root / "predictions.json"),
            "rules_sha256": file_hash(root / "rules.json"),
            "cache_sha256": file_hash(root / "cache-complete.json"),
            "public_audit_labels_joined": False,
            "medical_labels_joined": False,
            "all_81_bank_scores_committed": True,
        },
    )


def subset_metrics(labels, scores):
    return {
        **classification_stats(labels, scores, CLASSES),
        **(paired_bootstrap(labels, scores, CLASSES) or {}),
    }


def summarize_libraries(labels, values, metadata, epsilon, full):
    # Library repetitions are not new patients. Report ranges, not population CIs over seeds.
    baseline = values[:, full]
    out = {}
    for key in [f"real-{n}" for n in SIZES] + ["null-8"]:
        kind, count = key.split("-")
        selected = [
            i
            for i, b in enumerate(metadata)
            if b["kind"] == kind and b["families_per_class"] == int(count)
        ]
        rows = []
        for b in selected:
            v = values[:, b]
            metrics = classification_stats(labels, v.tolist(), CLASSES)
            flip = float(np.mean(np.argmin(v, axis=1) != np.argmin(baseline, axis=1)))
            certified = float(np.mean(np.abs(margin(baseline)) > float(epsilon[b])))
            rows.append(
                {
                    "bank": metadata[b]["id"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "auroc": metrics["class_auroc"]["head"],
                    "flip_rate": flip,
                    "stability_bound_coverage": certified,
                }
            )
        out[key] = {
            "images": len(labels),
            "libraries": len(selected),
            "images_per_class": int(count) * 3,
            "unique_draws": len({str(metadata[b]["families"]) for b in selected}),
            "metrics": {
                metric: {
                    "mean": float(np.mean([r[metric] for r in rows])),
                    "min": float(np.min([r[metric] for r in rows])),
                    "max": float(np.max([r[metric] for r in rows])),
                }
                for metric in (
                    "balanced_accuracy",
                    "auroc",
                    "flip_rate",
                    "stability_bound_coverage",
                )
            },
            "rows": rows,
        }
    return out


def evaluate(root, plan):
    commit = read_json(root / "predictions-complete.json")
    if (
        file_hash(root / "predictions.json") != commit["sha256"]
        or file_hash(root / "rules.json") != commit["rules_sha256"]
    ):
        raise ValueError("Frozen predictions or public rules changed")
    source = Path(plan["input"])
    public_labels = read_json(PUBLIC / "evaluation_labels.json")
    predictions = read_json(root / "predictions.json")
    bank_meta = read_json(root / "banks.json")
    metrics, stability, comparisons = {}, {}, {}
    csv_rows = []
    for family in ("llava", "qwen35"):
        cases = read_json(root / f"{family}-cases.json")
        index = {case["id"]: i for i, case in enumerate(cases)}
        tensors = read_tensors(root / f"{family}-scores.safetensors")
        score_values = tensors["scores"].numpy()
        committed = {p["id"]: p for p in predictions[family]}
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
            targets = {r["run"]: r for r in read_json(source / "private" / f"{ds}-targets.json")}
            for qa in ("natural", "fixed"):
                ids = [
                    i
                    for i, r in enumerate(cases)
                    if r["kind"] == "medical" and r["dataset"] == ds and r["qa"] == qa
                ]
                labels = [targets[mapping[cases[i]["id"]]["run"]]["label"] for i in ids]
                groups[f"{ds}_{qa}"] = (ids, labels)
        metrics[family], stability[family], comparisons[family] = {}, {}, {}
        for group, (ids, labels) in groups.items():
            algorithms = {
                name: [committed[cases[i]["id"]]["scores"][name] for i in ids]
                for name in ALGORITHMS
            }
            metrics[family][group] = {
                name: subset_metrics(labels, value) for name, value in algorithms.items()
            }
            comparisons[family][group] = {}
            for name, value in algorithms.items():
                same_module = (
                    "public_selected_raw" if name == "public_selected_calibrated" else "fixed_q32"
                )
                comparisons[family][group][name] = {
                    "vs_fixed_q32": paired_bootstrap(
                        labels, value, CLASSES, algorithms["fixed_q32"]
                    ),
                    "vs_uncalibrated_module": paired_bootstrap(
                        labels, value, CLASSES, algorithms[same_module]
                    ),
                }
                before = metrics[family][group][same_module]["class_auroc"]["head"]
                after = metrics[family][group][name]["class_auroc"]["head"]
                if name in ("q32_calibrated", "public_selected_calibrated") and before != after:
                    raise ValueError("Offset calibration changed ranking")
                m = metrics[family][group][name]
                csv_rows.append(
                    [
                        family,
                        group,
                        name,
                        m["n"],
                        m["correct"],
                        m["balanced_accuracy"],
                        m["recall"]["head"],
                        m["recall"]["non-head"],
                        m["class_auroc"]["head"],
                    ]
                )
            stability[family][group] = {
                method: summarize_libraries(
                    labels,
                    score_values[np.array(ids), :, j],
                    bank_meta,
                    tensors[f"epsilon_{method}"],
                    plan["full_bank"],
                )
                for j, method in enumerate(METHODS)
            }
    write_json(
        root / "aggregate.json",
        {
            "primary": "public_selected_calibrated",
            "metrics": metrics,
            "stability": stability,
            "comparisons": comparisons,
            "rules": read_json(root / "rules.json"),
            "banks": {k: plan[k] for k in ("bank_count", "real_banks", "null_banks", "full_bank")},
            "scope": "Repeated existing images and QA; public-only fitting, not an independent confirmation",
            "unique_medical_images": 40,
            "new_medical_images": 0,
            "gpu_used": False,
            "uncommitted": True,
        },
    )
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(
        [
            "Model",
            "Group",
            "Algorithm",
            "Images",
            "Correct",
            "Balanced accuracy",
            "Head recall",
            "Non-head recall",
            "Head AUROC",
        ]
    )
    writer.writerows(csv_rows)
    (root / "summary.csv").write_text(stream.getvalue())


def run(root):
    if (root / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(
        root / "started.json", {"pid": os.getpid(), "host": socket.gethostname(), "device": "cpu"}
    )
    started = time.monotonic()
    plan = read_json(root / "plan.json")
    if (
        file_hash(Path(__file__)) != plan["execution_source_sha256"]
        or file_hash(Path(__file__).with_name("prototype_method_math.py"))
        != plan["math_source_sha256"]
    ):
        raise ValueError("Frozen method-study source changed")
    verify_inputs(plan)
    build_cache(root, plan)
    fit_and_commit(root, plan)
    evaluate(root, plan)
    write_json(
        root / "complete.json",
        {
            "status": "complete",
            "seconds": time.monotonic() - started,
            "runtime_source": str(Path(__file__).resolve()),
            "aggregate_sha256": file_hash(root / "aggregate.json"),
            "gpu_used": False,
        },
    )
    print("Method study complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=INPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.stage == "prepare":
            prepare(args.root.resolve(), args.input.resolve())
        else:
            run(args.root.resolve())
    except BaseException:
        write_json(args.root / f"{args.stage}-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
