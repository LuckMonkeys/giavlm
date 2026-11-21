"""Boundary diagnosis: paired QA, energy decomposition, declared public medical bank."""

import argparse
from collections import Counter
from dataclasses import asdict, replace
import gc
import hashlib
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from attacks.analytic.dager_subspace import row_basis
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation
from evaluation.crossmodel_property import (
    ANSWER,
    QUESTION,
    METHODS,
    capture_fixed_model,
    hook_features,
    module_key,
    validate_observation,
)
from evaluation.image_property_data import DATA, annotation_join, pixel_hash
from evaluation.image_property_followup_data import OLD, region
from evaluation.prototype_boundary_math import decompose, fit_rule, geometry
from evaluation.prototype_method_math import calibrate_offset


SOURCE = Path("outputs/diagnostics/crossmodel_property_20261104_v3")
PREVIOUS = Path("outputs/diagnostics/prototype_method_20261105")


def inspect(root):
    from core.data import read_manifest

    if (root / "inspection.json").exists():
        raise FileExistsError("No resampling of the proposed auxiliary bank")
    local, joined = annotation_join(read_manifest(DATA), OLD / "metadata_source")
    # Evaluator-only exclusions. No target paths or identities enter the public bank.
    excluded_ids = {k for k, g in local.items() if g[0]["split"] == "eval"}
    excluded_pixels = {pixel_hash(Image.open(local[k][0]["image"])) for k in excluded_ids}
    exclusions = list((OLD / "private").glob("rank-*.json"))
    exclusions += [SOURCE / "private/rad-targets.json", SOURCE / "private/slake-targets.json"]
    for folder in (
        "image_property_followup_gpu5_20261016",
        "synthetic_ct_property_gpu5_20261016",
        "synthetic_head_n20_gpu5_20261017",
        "multilayer_property_20261101",
    ):
        path = Path("outputs/diagnostics") / folder / "private/targets.json"
        if path.exists():
            exclusions.append(path)
    for path in exclusions:
        for row in read_json(path):
            original = row.get("row", row)
            key = original["image_id"]
            excluded_ids.add(key)
            image = original.get("image")
            if image is None and key in local:
                image = local[key][0]["image"]
            if image:
                excluded_pixels.add(pixel_hash(Image.open(image)))
    pools = {(c, m): [] for c in ("head", "non-head") for m in ("CT", "MRI")}
    for key, group in local.items():
        info = joined[key]
        body = region(info["locations"])
        if (
            key in excluded_ids
            or group[0]["split"] != "train"
            or len(info["matches"]) != 1
            or body is None
            or info["label"] not in ("CT", "MRI")
        ):
            continue
        label = "head" if body == "head" else "non-head"
        digest = pixel_hash(Image.open(group[0]["image"]))
        if digest not in excluded_pixels:
            pools[label, info["label"]].append((key, group[0]["image"], digest, body))
    proposed, used = [], set()
    availability = {}
    for (label, modality), rows in pools.items():
        rows.sort(key=lambda r: hashlib.sha256(f"20261105:pub-med:{r[0]}".encode()).hexdigest())
        availability[f"{label}_{modality}"] = len(rows)
        unique = []
        for row in rows:
            if row[2] not in used:
                unique.append(row)
                used.add(row[2])
        if len(unique) < 8:
            raise ValueError("Insufficient isolated training images for declared CT/MRI balance")
        for i, (key, image, digest, body) in enumerate(unique[:8]):
            split = "fit" if i < 4 else "calibration" if i < 6 else "audit"
            proposed.append(
                {
                    "id": f"medical-{len(proposed):02d}",
                    "label": label,
                    "modality": modality,
                    "region": body,
                    "split": split,
                    "image_id": key,
                    "file": image,
                    "pixel_sha256": digest,
                }
            )
    write_json(root / "auxiliary-proposed.json", proposed)
    write_json(
        root / "inspection.json",
        {
            "eligible": availability,
            "n": len(proposed),
            "target_pixel_overlap": 0,
            "target_id_overlap": 0,
            "excluded_sources": {str(p): file_hash(p) for p in exclusions},
            "patient_independence": "Unknown; no patient identifiers",
            "selection": "train only; deterministic public ID hash; CT/MRI balanced in every split and class",
        },
    )
    for page in range(2):
        sheet = Image.new("RGB", (1400, 1280), "white")
        draw = ImageDraw.Draw(sheet)
        for j, row in enumerate(proposed[page * 16 : (page + 1) * 16]):
            x, y = (j % 4) * 350, (j // 4) * 320
            draw.text(
                (x + 6, y + 3),
                f"{row['id']} {row['label']} {row['modality']} {row['region']} {row['split']}",
                fill="black",
            )
            sheet.paste(
                ImageOps.contain(Image.open(row["file"]).convert("RGB"), (338, 285)),
                (x + 6, y + 30),
            )
        sheet.save(root / f"auxiliary-inspection-{page}.jpg", quality=93)
    print({"eligible": availability, "review_images": len(proposed)}, flush=True)


def prepare(root):
    if (root / "plan.json").exists():
        raise FileExistsError("Frozen boundary plan exists")
    review = read_json(root / "visual-review.json")
    medical_files = []
    if review.get("bank_status") == "insufficient_after_visual_review":
        write_json(root / "medical-bank-stopped.json", review)
    else:
        proposed = read_json(root / "auxiliary-reviewed.json")
        if not review.get("all_final_32_accepted"):
            raise ValueError("Auxiliary bank must be visually reviewed before freezing")
        if Counter((r["label"], r["modality"], r["split"]) for r in proposed) != {
            (c, m, split): n
            for c in ("head", "non-head")
            for m in ("CT", "MRI")
            for split, n in (("fit", 4), ("calibration", 2), ("audit", 2))
        }:
            raise ValueError("Reviewed bank changed class/modality/split quotas")
        pixels = {
            r["id"]: torch.from_numpy(np.array(Image.open(r["file"]).convert("RGB"), copy=True))
            for r in proposed
        }
        if any(pixel_hash(Image.open(r["file"])) != r["pixel_sha256"] for r in proposed):
            raise ValueError("Proposed pixels changed during review")
        write_tensors(root / "public-medical-images.safetensors", pixels)
        medical = [
            {k: r[k] for k in ("id", "label", "modality", "split", "pixel_sha256")}
            for r in proposed
        ]
        write_json(root / "public-medical-bank.json", medical)
        medical_files = [
            root / n
            for n in (
                "public-medical-bank.json",
                "public-medical-images.safetensors",
                "auxiliary-reviewed.json",
            )
        ]
    previous = read_json(PREVIOUS / "plan.json")
    public = read_json(SOURCE / "public-bank.json")
    split_by_id = {
        r["id"]: split
        for split in ("fit", "calibration", "audit")
        for r in previous[f"{split}_rows"]
    }
    generic = [{**r, "split": split_by_id[r["id"]]} for r in public if r["style"] == "photo"]
    if Counter(r["split"] for r in generic) != {"fit": 16, "calibration": 8, "audit": 8}:
        raise ValueError("Generic photo split mismatch")
    write_json(root / "public-generic-bank.json", generic)
    files = [
        PREVIOUS / "finalized.json",
        PREVIOUS / "plan.json",
        PREVIOUS / "rules.json",
        SOURCE / "plan.json",
        root / "public-generic-bank.json",
        root / "inspection.json",
        root / "visual-review.json",
    ] + medical_files
    for family in ("llava", "qwen35"):
        files += [
            PREVIOUS / f"{family}-scores.safetensors",
            PREVIOUS / f"{family}-cases.json",
            SOURCE / family / "runtime.json",
            SOURCE / family / "public/visual-means.safetensors",
            SOURCE / family / "public/prototypes.safetensors",
        ]
        files += [
            Path(read_json(SOURCE / "plan.json")["model_paths"][family]) / n
            for n in ("model.json", "model.safetensors")
        ]
    write_json(
        root / "plan.json",
        {
            "source": str(SOURCE.resolve()),
            "previous": str(PREVIOUS.resolve()),
            "primary": "Q24",
            "secondary": "Q32",
            "methods": list(METHODS),
            "new_target_images": 0,
            "new_public_medical_images": 32 if medical_files else 0,
            "medical_bank_status": review.get("bank_status", "ready"),
            "relative_energy_floor": 1e-12,
            "minimum_free_gib": 35,
            "domain_prior": "Public medical bank explicitly declares medical imaging auxiliary knowledge",
            "model_paths": read_json(SOURCE / "plan.json")["model_paths"],
            "hashes": {str(p.resolve()): file_hash(p) for p in files},
            "source_sha256": file_hash(Path(__file__)),
            "math_sha256": file_hash(Path(__file__).with_name("prototype_boundary_math.py")),
        },
    )


def verify(plan):
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen boundary input changed")
    if file_hash(Path(__file__)) != plan["source_sha256"]:
        raise ValueError("Boundary worker source differs from freeze")


def load_gradients(case, family):
    if case["kind"] == "public":
        return read_tensors(case["path"])
    obs = load_observation(case["observation"], "cpu")
    validate_observation(obs, read_json(SOURCE / family / "runtime.json")["fingerprint"])
    return {m: obs.tensors[module_key(family, m)] for m in METHODS}


def score_distances(proto, gradient):
    basis, spectrum = row_basis(gradient.double(), 1e-6, 1e-10)
    projected = proto @ basis.T
    distances = (proto - projected @ basis).norm(dim=1)
    extra = geometry(proto, basis)
    extra.update(
        rank=len(basis),
        singular_max=float(spectrum.max()),
        singular_min_retained=float(spectrum[: len(basis)].min()) if len(basis) else 0.0,
    )
    return distances.tolist(), extra


def cpu(root):
    if (root / "cpu/started.json").exists():
        raise FileExistsError("No automatic repeat of CPU diagnosis")
    write_json(root / "cpu/started.json", {"pid": os.getpid(), "host": socket.gethostname()})
    started = time.monotonic()
    plan = read_json(root / "plan.json")
    verify(plan)
    old_plan = read_json(PREVIOUS / "plan.json")
    old_rules = read_json(PREVIOUS / "rules.json")
    from evaluation.crossmodel_property import PUBLIC

    public_labels = read_json(PUBLIC / "evaluation_labels.json")
    output = {}
    checked, max_error = 0, 0.0
    for family in ("llava", "qwen35"):
        protos = read_tensors(SOURCE / family / "public/prototypes.safetensors")
        cases = read_json(PREVIOUS / f"{family}-cases.json")
        prior = read_tensors(PREVIOUS / f"{family}-scores.safetensors")["scores"][
            :, old_plan["full_bank"]
        ]
        rows = []
        for i, case in enumerate(cases):
            gradients = load_gradients(case, family)
            values, details = {}, {}
            for j, m in enumerate(METHODS):
                distances, extra = score_distances(protos[m], gradients[m])
                error = float(np.max(np.abs(np.asarray(distances) - prior[i, j].numpy())))
                if error > 1e-12:
                    raise ValueError("Original distance replay differs")
                max_error = max(max_error, error)
                values[m], details[m] = distances, extra
                checked += 1
            rows.append({"id": case["id"], "distances": values, "geometry": details})
            if i % 16 == 0:
                print(f"CPU {family} {i + 1}/{len(cases)}", flush=True)
        index = {r["id"]: i for i, r in enumerate(rows)}
        ids = [index[r["id"]] for r in old_plan["calibration_rows"]]
        labels = [public_labels[r["id"]] == "head" for r in old_plan["calibration_rows"]]
        styles = [r["style"] for r in old_plan["calibration_rows"]]
        rules = {}
        for m in METHODS:
            z = decompose([r["distances"][m] for r in rows])
            rules[m] = {
                "raw_threshold": old_rules[family]["offsets"][m]["threshold"],
                "relative_rule": fit_rule(z["relative"][ids], labels, styles, z["valid"][ids]),
            }
        output[family] = {"rows": rows, "rules": rules}
    write_json(root / "cpu/predictions.json", output)
    write_json(
        root / "cpu/predictions-complete.json",
        {
            "sha256": file_hash(root / "cpu/predictions.json"),
            "all_target_labels_unread": True,
            "distance_checks": checked,
            "max_distance_error": max_error,
            "seconds": time.monotonic() - started,
            "runtime_source": str(Path(__file__).resolve()),
        },
    )
    print("CPU predictions committed", flush=True)


def gpu(root, family):
    from core.fl import capture, restore_model

    out = root / family
    if (out / "started.json").exists():
        raise FileExistsError("No GPU overwrite or automatic retry")
    started = time.monotonic()
    write_json(
        out / "started.json",
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    plan = read_json(root / "plan.json")
    verify(plan)
    if plan["medical_bank_status"] != "ready":
        raise ValueError("Frozen public medical bank did not pass data eligibility gate")
    if (
        torch.cuda.device_count() != 1
        or torch.cuda.mem_get_info()[0] < plan["minimum_free_gib"] * 2**30
    ):
        raise RuntimeError("Selected GPU does not meet frozen memory reserve")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = restore_model(plan["model_paths"][family], "cuda:0")
    adapter.eval()
    adapter.set_training_spec(
        replace(
            adapter.training_spec,
            algorithm="fedsgd",
            local_optimizer="sgd",
            batch_size=1,
            local_steps=1,
            gradient_accumulation_steps=1,
            weight_decay=0.0,
            knowledge="private",
            token_lengths_known=False,
        )
    )
    fingerprint = adapter.fingerprint()
    if fingerprint != read_json(SOURCE / family / "runtime.json")["fingerprint"]:
        raise ValueError("Model snapshot differs from original experiment")
    keys = {m: module_key(family, m) for m in METHODS}
    bank = read_json(root / "public-medical-bank.json")
    pixels = read_tensors(root / "public-medical-images.safetensors")
    features, gradients, ranges = {}, [], {}
    gate_done = False
    for i, row in enumerate(bank):
        image = (
            adapter.prepare_image(Image.fromarray(pixels[row["id"]].numpy()))
            .unsqueeze(0)
            .to(adapter.device, adapter.dtype)
        )
        values, ranges[row["id"]] = hook_features(adapter, family, image, QUESTION, ANSWER, keys)
        features.update({f"{row['id']}_{m}": values[m].mean(0) for m in METHODS})
        if row["split"] != "fit":
            batch = adapter.batch(image, [QUESTION], [ANSWER])
            obs = capture_fixed_model(adapter, batch, fingerprint)
            if not gate_done:
                native = capture(adapter, batch, [], [])
                if native.tensors.keys() != obs.tensors.keys() or not all(
                    torch.equal(native.tensors[k], obs.tensors[k]) for k in obs.tensors
                ):
                    raise ValueError("Original full native capture differs")
                write_json(
                    out / "native-capture-gate.json",
                    {"all_tensors_exact": True, "tensors": len(obs.tensors)},
                )
                del native
                gate_done = True
            path = out / "public-gradients" / f"{row['id']}.safetensors"
            write_tensors(path, {m: obs.tensors[k] for m, k in keys.items()})
            gradients.append(
                {
                    "id": row["id"],
                    "kind": "public",
                    "domain": "medical",
                    "path": str(path),
                    "split": row["split"],
                }
            )
            del obs, batch
        write_json(
            out / "progress.json", {"stage": "public_medical", "completed": i + 1, "total": 32}
        )
        print(f"GPU {family} public {i + 1}/32", flush=True)
    if adapter.fingerprint() != fingerprint:
        raise ValueError("Read-only model changed")
    write_tensors(out / "visual-means.safetensors", features)
    write_json(out / "feature-ranges.json", ranges)
    write_json(out / "public-gradient-list.json", gradients)
    write_json(
        out / "gpu-complete.json",
        {
            "seconds": time.monotonic() - started,
            "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "fingerprint": fingerprint,
            "training": asdict(adapter.training_spec),
            "runtime_source": str(Path(__file__).resolve()),
            "public_images": 32,
            "public_gradients": 16,
        },
    )
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU {family} complete", flush=True)


def bank_scores(root, family):
    out = root / family
    if (out / "bank-predictions-complete.json").exists():
        raise FileExistsError("No reselecting prototypes")
    verify(read_json(root / "plan.json"))
    generic_rows = read_json(root / "public-generic-bank.json")
    medical_rows = read_json(root / "public-medical-bank.json")
    from evaluation.crossmodel_property import PUBLIC

    generic_labels = {
        **read_json(PUBLIC / "fit_labels.json"),
        **read_json(PUBLIC / "evaluation_labels.json"),
    }
    generic_features = read_tensors(SOURCE / family / "public/visual-means.safetensors")
    medical_features = read_tensors(out / "visual-means.safetensors")
    prototypes = {}
    for name, rows, features in (
        ("generic", generic_rows, generic_features),
        ("medical", medical_rows, medical_features),
    ):
        for m in METHODS:
            means = torch.stack(
                [
                    torch.stack(
                        [
                            features[f"{r['id']}_{m}"]
                            for r in rows
                            if r["split"] == "fit"
                            and (generic_labels[r["id"]] if name == "generic" else r["label"])
                            == label
                        ]
                    ).mean(0)
                    for label in ("head", "non-head")
                ]
            )
            prototypes[f"{name}_{m}"] = torch.nn.functional.normalize(means.double(), dim=1)
    write_tensors(out / "bank-prototypes.safetensors", prototypes)
    original = {r["id"]: r for r in read_json(PREVIOUS / f"{family}-cases.json")}
    cases = [
        {**original[r["id"]], "domain": "generic", "split": r["split"]}
        for r in generic_rows
        if r["split"] != "fit"
    ]
    cases += read_json(out / "public-gradient-list.json")
    cases += [r for r in original.values() if r["kind"] == "medical"]
    predictions = []
    for i, case in enumerate(cases):
        gs = load_gradients(case, family)
        scores = {}
        for m in METHODS:
            basis, _ = row_basis(gs[m].double(), 1e-6, 1e-10)
            for name in ("generic", "medical"):
                mu = prototypes[f"{name}_{m}"]
                scores[f"{name}_{m}"] = (mu - (mu @ basis.T) @ basis).norm(dim=1).tolist()
        predictions.append({"id": case["id"], "distances": scores})
        if i % 16 == 0:
            print(f"Bank score {family} {i + 1}/{len(cases)}", flush=True)
    rules = {}
    index = {r["id"]: r for r in predictions}
    for name, rows in (("generic", generic_rows), ("medical", medical_rows)):
        cal = [r for r in rows if r["split"] == "calibration"]
        labels = [
            (generic_labels[r["id"]] if name == "generic" else r["label"]) == "head" for r in cal
        ]
        for m in METHODS:
            values = [index[r["id"]]["distances"][f"{name}_{m}"] for r in cal]
            z = decompose(values)
            rules[f"{name}_{m}"] = calibrate_offset(z["margin"], labels, ["fixed"] * len(cal))
    write_json(out / "bank-cases.json", cases)
    write_json(out / "bank-predictions.json", predictions)
    write_json(out / "bank-rules.json", rules)
    write_json(
        out / "bank-predictions-complete.json",
        {
            "sha256": file_hash(out / "bank-predictions.json"),
            "rules_sha256": file_hash(out / "bank-rules.json"),
            "target_labels_read": False,
            "cases_sha256": file_hash(out / "bank-cases.json"),
            "prototype_sha256": file_hash(out / "bank-prototypes.safetensors"),
        },
    )
    print(f"Bank scores {family} committed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "prepare", "cpu", "gpu", "banks"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("llava", "qwen35"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    try:
        if args.stage in ("gpu", "banks"):
            {"gpu": gpu, "banks": bank_scores}[args.stage](root, args.family)
        else:
            {"inspect": inspect, "prepare": prepare, "cpu": cpu}[args.stage](root)
    except BaseException:
        write_json(
            root / (args.family or args.stage) / "failed.json",
            {"traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
