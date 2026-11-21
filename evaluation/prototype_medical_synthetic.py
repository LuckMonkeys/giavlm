"""Predeclared synthetic radiology bank for the prototype boundary control."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, write_json, write_tensors
from evaluation import prototype_boundary_study as boundary
from evaluation.image_property_data import pixel_hash


PARENT = Path("outputs/diagnostics/prototype_boundary_20261105")
MODEL = Path("/home/zx/public/model-hub/huggingface/Qwen/Qwen-Image-2.1-Turbo")


def cases():
    head = [
        "axial brain slice at the lateral ventricles",
        "axial brain slice at the basal ganglia",
        "coronal brain slice through the frontal lobes",
        "paramedian sagittal brain slice showing cerebral tissue",
        "axial brain slice through the centrum semiovale",
        "coronal brain slice through the posterior cerebral hemispheres",
        "axial brain slice through the cerebellum and posterior fossa",
        "midline sagittal brain slice showing the corpus callosum and cerebellum",
    ]
    nonhead = [
        "axial abdominal slice through the kidneys and liver",
        "axial thoracic slice through both lungs",
        "coronal abdominal slice through the liver and kidneys",
        "coronal thoracic slice through the heart and lungs",
        "axial upper abdominal slice through the liver and spleen",
        "axial thoracic slice through the mediastinum and lungs",
        "sagittal abdominal slice through lumbar spine and abdominal organs",
        "sagittal thoracic slice through the sternum and heart",
    ]
    rows = []
    for label, contents in (("head", head), ("non-head", nonhead)):
        for modality in ("CT", "MRI"):
            for i, content in enumerate(contents):
                split = "fit" if i < 4 else "calibration" if i < 6 else "audit"
                texture = (
                    "computed tomography, recognizable bright cortical bone, grayscale soft tissue"
                    if modality == "CT"
                    else "magnetic resonance imaging, grayscale soft tissue contrast, no CT-style bright cortical bone"
                )
                rows.append(
                    {
                        "id": f"medical-{len(rows):02d}",
                        "label": label,
                        "modality": modality,
                        "group": f"{label}-content-{i:02d}",
                        "split": split,
                        "style": modality,
                        "seed": 2026110500 + len(rows),
                        "prompt": f"A realistic radiology image in the visual style of {texture}. A single {content}. "
                        "Only internal anatomy in a flat medical scan, centered and large in a square frame on a plain black background. "
                        "No external face, no portrait, no standing person, no neck-only scan. No labels, no text, no arrows, no watermark, no collage.",
                    }
                )
    return rows


def prepare_generation(root, gpu_uuid):
    if (root / "generation/plan.json").exists():
        raise FileExistsError("Generation plan already frozen")
    if read_json(PARENT / "finalized.json")["status"] != "complete":
        raise ValueError("Prior boundary phase not finalized")
    write_json(
        root / "generation/plan.json",
        {
            "model": str(MODEL),
            "model_index_sha256": file_hash(MODEL / "model_index.json"),
            "sample_sigmas": read_json(MODEL / "model_index.json")["sample_sigmas"],
            "gpu_uuid": gpu_uuid,
            "cases": cases(),
            "dtype": "bfloat16",
            "width": 1024,
            "height": 1024,
            "true_cfg_scale": 1.0,
            "attempts_per_case": 1,
            "target_specific_prompt_details": False,
            "selection": "All fixed prompts and seeds; no target images, QA or gradients used in generation",
            "condition": "Wholly synthetic medical-appearance bank, not a repaired real-public-image bank",
            "clinical_validity": "CT/MRI denote prompt appearance groups; no clinical reconstruction claim",
            "source_sha256": file_hash(Path(__file__)),
        },
    )


def prepare(root):
    if (root / "plan.json").exists():
        raise FileExistsError("Synthetic medical scoring plan already frozen")
    generated = read_json(root / "generation/complete.json")
    if generated["plan_sha256"] != file_hash(root / "generation/plan.json"):
        raise ValueError("Frozen generation plan changed")
    review = read_json(root / "visual-review.json")
    rows = generated["images"]
    expected = cases()
    if not review.get("all_32_accepted") or len(rows) != 32:
        raise ValueError("Every frozen synthetic image must pass the public-only review")
    if any(
        any(a[k] != b[k] for k in ("id", "label", "modality", "group", "split", "seed", "prompt"))
        for a, b in zip(rows, expected, strict=True)
    ):
        raise ValueError("Generated bank differs from frozen prompts or partitions")
    pixels = {}
    for row in rows:
        if file_hash(row["file"]) != row["sha256"]:
            raise ValueError("Generated image changed")
        image = Image.open(row["file"]).convert("RGB")
        pixels[row["id"]] = torch.from_numpy(np.array(image, copy=True))
        row["pixel_sha256"] = pixel_hash(image)
    if len({r["pixel_sha256"] for r in rows}) != 32:
        raise ValueError("Duplicate synthetic images")
    # Evaluator-owned overlap gate; the scoring worker receives no target identity.
    from core.data import read_manifest
    from evaluation.image_property_data import DATA
    from evaluation.image_property_followup_data import OLD

    local = {r["image_id"]: r for r in read_manifest(DATA)}
    target_files = {r["image"] for r in local.values() if r["split"] == "eval"}
    for path in (OLD / "private").glob("rank-*.json"):
        for r in read_json(path):
            target_files.add(local[r["image_id"]]["image"])
    for dataset in ("slake", "rad"):
        for r in read_json(boundary.SOURCE / "private" / f"{dataset}-targets.json"):
            target_files.add(r["row"]["image"])
    targets = {pixel_hash(Image.open(p)) for p in target_files}
    if targets & {r["pixel_sha256"] for r in rows}:
        raise ValueError("Synthetic auxiliary bank overlaps a target by pixels")
    write_json(
        root / "overlap-audit.json",
        {
            "unique_generated_pixels": 32,
            "target_images_checked": len(target_files),
            "target_pixel_overlap": 0,
            "target_specific_generation_inputs": False,
        },
    )
    write_tensors(root / "public-medical-images.safetensors", pixels)
    bank = [
        {k: r[k] for k in ("id", "label", "modality", "group", "split", "pixel_sha256")}
        for r in rows
    ]
    write_json(root / "public-medical-bank.json", bank)
    shutil.copy2(PARENT / "public-generic-bank.json", root / "public-generic-bank.json")
    parent = read_json(PARENT / "plan.json")
    files = [
        root / n
        for n in (
            "generation/plan.json",
            "generation/complete.json",
            "visual-review.json",
            "public-medical-images.safetensors",
            "public-medical-bank.json",
            "public-generic-bank.json",
            "overlap-audit.json",
        )
    ]
    files += [Path(p) for p in parent["hashes"] if not p.startswith(str(PARENT.resolve()))]
    files += [PARENT / "finalized.json"]
    for family in ("llava", "qwen35"):
        files.append(boundary.SOURCE / family / "capture-complete.json")
    write_json(
        root / "plan.json",
        {
            **{k: v for k, v in parent.items() if k != "hashes"},
            "medical_bank_status": "ready",
            "new_public_medical_images": 32,
            "condition": "Generic synthetic photo versus medical synthetic; equal image counts",
            "domain_prior": "Explicit medical appearance and CT/MRI prompts; not a domain-unknown attack",
            "family_caveat": "CT and MRI share content family within each split; synthetic fit has four anatomy families per class versus eight generic-photo families",
            "hashes": {str(p.resolve()): file_hash(p) for p in set(files)},
            "source_sha256": file_hash(Path(boundary.__file__)),
            "synthesis_source_sha256": file_hash(Path(__file__)),
        },
    )
    print(
        {
            "synthetic_medical": dict(Counter(r["split"] for r in bank)),
            "target_images": 40,
            "primary": "Q24",
            "secondary": "Q32",
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generation-plan", "prepare"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-uuid")
    args = parser.parse_args()
    try:
        if args.stage == "generation-plan":
            prepare_generation(args.root.resolve(), args.gpu_uuid)
        else:
            prepare(args.root.resolve())
    except BaseException:
        write_json(args.root / "setup-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
