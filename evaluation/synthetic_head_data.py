"""Evaluator-owned sampling, annotation audit and frozen synthetic head bank."""

import argparse
from collections import Counter
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from core.artifacts import file_hash, read_json, write_json, write_tensors
from core.data import read_manifest
from evaluation.image_property_data import DATA, annotation_join, pixel_hash
from evaluation.image_property_followup_data import CONFIG, MODEL, OLD, region


ROOT = Path("outputs/diagnostics/synthetic_head_n20_gpu5_20261017")
SEED = 20261017
QUOTAS = [("CT", "head", 5), ("MRI", "head", 5), ("CT", "abdomen", 3),
          ("MRI", "abdomen", 3), ("X-Ray", "chest", 4)]
HEAD_WORDS = re.compile(r"head|brain|skull|cran|cerebr|cerebell|ventric|pituitary|thalam", re.I)


def inspect():
    if (ROOT / "private/sampling.json").exists():
        raise FileExistsError("Sampling already saved")
    local, joined = annotation_join(read_manifest(DATA), OLD / "metadata_source")
    rng = random.Random(SEED)
    sampling, rows = [], []
    for modality, body, count in QUOTAS:
        keys = sorted(k for k, info in joined.items()
                      if local[k][0]["split"] == "eval" and len(info["matches"]) == 1
                      and info["label"] == modality and region(info["locations"]) == body)
        rng.shuffle(keys)
        if len(keys) < count:
            raise ValueError("Insufficient target stratum")
        sampling.append({"modality": modality, "region": body, "count": count, "order": keys})
        # One reserve for visual mislabeling; never select by attack outcomes.
        for j, key in enumerate(keys[:count + 1]):
            qa_rng = random.Random(f"{SEED}:{key}")
            row = qa_rng.choice(sorted(local[key], key=lambda r: r["sample_id"]))
            rows.append({"inspection_index": len(rows), "modality": modality, "region": body,
                         "reserve": j >= count, "label": "head" if body == "head" else "non-head",
                         "annotation": joined[key], "row": row,
                         "head_terms_present": bool(HEAD_WORDS.search(row["question"] + " " + row["target"]))})
    write_json(ROOT / "private/sampling.json", {"seed": SEED, "strata": sampling,
               "qa_selection": "Uniform deterministic one original QA per image; no text-content filter",
               "manifest_sha256": file_hash(DATA), "inspection": rows})
    for start in range(0, len(rows), 15):
        chunk = rows[start:start + 15]
        sheet = Image.new("RGB", (1500, 340 * ((len(chunk) + 4) // 5)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, item in enumerate(chunk):
            x, y = j % 5 * 300, j // 5 * 340
            with Image.open(item["row"]["image"]) as image:
                sheet.paste(ImageOps.contain(image.convert("RGB"), (294, 300)), (x + 3, y + 34))
            draw.text((x + 5, y + 8), f"{item['inspection_index']} {item['modality']} {item['region']} "
                      + ("RESERVE" if item["reserve"] else ""), fill="black")
        sheet.save(ROOT / "private" / f"inspection-{start // 15}.jpg", quality=95)
    print("Sampling saved; review anonymous inspection sheets before freezing.")


def freeze():
    if (ROOT / "plan.json").exists():
        raise FileExistsError("Plan already frozen")
    review = read_json(ROOT / "private/visual_review.json")
    sampling = read_json(ROOT / "private/sampling.json")
    by_index = {r["inspection_index"]: r for r in sampling["inspection"]}
    targets = [{**by_index[i], "run": j} for j, i in enumerate(review["selected_indices"])]
    if Counter((r["modality"], r["region"]) for r in targets) != {
            (m, r): n for m, r, n in QUOTAS}:
        raise ValueError("Reviewed targets violate frozen strata")
    target_hashes = {pixel_hash(Image.open(r["row"]["image"])) for r in targets}
    if len(target_hashes) != 20 or len({r["row"]["image_id"] for r in targets}) != 20:
        raise ValueError("Target duplication")
    write_json(ROOT / "private/targets.json", targets)
    generation = read_json(ROOT / "generated/complete.json")
    repair = read_json(ROOT / "generated_repair/complete.json")
    initial_review = read_json(ROOT / "generated/visual_review.json")
    public_review = read_json(ROOT / "public/visual_review.json")
    if (generation["status"] != "complete" or repair["status"] != "complete"
            or public_review["retained"] != 24):
        raise ValueError("Candidate bank not reviewed")
    bank, raw, generated_hashes = [], {}, {}
    candidates = [("generated", generation["images"][i]) for i in initial_review["retained_original_indices"]]
    candidates += [("generated_repair", item) for item in repair["images"]]
    for directory, item in candidates:
        path = ROOT / directory / item["file"]
        if file_hash(path) != item["sha256"]:
            raise ValueError("Candidate changed")
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel = pixel_hash(image)
            if pixel in target_hashes:
                raise ValueError("Candidate overlaps target")
            identifier = f"candidate-{len(bank):02d}"
            raw[identifier] = torch.from_numpy(np.array(image, copy=True))
        bank.append({"id": identifier, "label": item["property_label"], "modality_prompt": item["modality"],
                     "region_prompt": item["region"], "seed": item["seed"], "pixel_sha256": pixel,
                     "generated_file": f"{directory}/{item['file']}"})
        generated_hashes[str(path)] = file_hash(path)
    if Counter(r["label"] for r in bank) != {"head": 12, "non-head": 12}:
        raise ValueError("Unbalanced synthetic bank")
    if len({r["pixel_sha256"] for r in bank}) != 24:
        raise ValueError("Duplicate candidates")
    write_json(ROOT / "public/bank.json", bank)
    write_tensors(ROOT / "public/bank.safetensors", raw)
    previous = {r["image_id"] for p in (OLD / "private").glob("rank-*.json") for r in read_json(p)}
    for name in ["image_property_followup_gpu5_20261016", "synthetic_ct_property_gpu5_20261016"]:
        previous |= {r["row"]["image_id"] for r in read_json(Path("outputs/diagnostics") / name / "private/targets.json")}
    write_json(ROOT / "data_audit.json", {
        "targets": 20, "target_counts": dict(Counter(r["label"] for r in targets)),
        "strata": [{"modality": m, "region": r, "count": n} for m, r, n in QUOTAS],
        "head_terms_present": dict(Counter(r["label"] for r in targets if r["head_terms_present"])),
        "unique_original_qa": len({(r["row"]["question"], r["row"]["target"]) for r in targets}),
        "prior_pilot_image_overlap": sum(r["row"]["image_id"] in previous for r in targets),
        "unique_target_pixels": len(target_hashes), "unique_bank_pixels": 24,
        "bank_target_overlap": 0, "target_split": "eval", "seed": SEED,
        "sampling": "Stratified random images, one random unmodified original QA; visual exclusions only before inference",
        "prevalence": "Balanced diagnostic case-control cohort, not natural prevalence",
        "patient_independence": "Unknown; image deduplication does not establish patient independence",
        "xray_head_targets": 0, "uses_private_reference_for_evaluation": True,
    })
    files = [MODEL / "model.json", MODEL / "model.safetensors", CONFIG, DATA,
             ROOT / "private/targets.json", ROOT / "private/sampling.json", ROOT / "private/visual_review.json",
             ROOT / "public/bank.json", ROOT / "public/bank.safetensors", ROOT / "public/visual_review.json",
             ROOT / "generated/plan.json", ROOT / "generated/complete.json",
             ROOT / "generated/visual_review.json", ROOT / "generated_repair/plan.json",
             ROOT / "generated_repair/complete.json",
             Path(__file__), Path("evaluation/synthetic_head_generation.py"),
             Path("evaluation/synthetic_head_generation_repair.py"),
             Path("evaluation/synthetic_head_property.py"), Path("evaluation/image_property_study.py"),
             Path("evaluation/image_property_scoring.py"), Path("evaluation/dager_rank_study.py"),
             Path("attacks/analytic/dager_adapted.py")]
    hashes = {str(p): file_hash(p) for p in files}
    hashes.update(generated_hashes)
    write_json(ROOT / "plan.json", {
        "schema_version": 1, "classes": ["head", "non-head"], "targets": 20,
        "model": str(MODEL), "config": str(CONFIG), "rank": 32, "round": 10, "physical_gpu": 5,
        "lm_prior_weight": 0.01, "dager_image_source": "random", "token_lengths_known": True,
        "primary_score": "centered", "secondary_scores": ["global", "equal_tensor", "centered_equal_tensor"],
        "top_m": 3, "class_aggregation": "Mean lowest three losses among twelve prototypes per class",
        "text_only_prompt": "USER: Infer whether the image depicts the head from the question and answer. Choose head or non-head.\nQuestion: {question}\nAnswer: {answer}\nASSISTANT:",
        "text_only_calibration": "None; fixed prior to scoring",
        "text_conditions": ["reconstructed", "oracle"], "run_both_regardless_of_accuracy": True,
        "primary_endpoint": "Reconstructed-text centered balanced accuracy; all 20 targets",
        "subgroups": ["modality", "head_terms_present", "joint_text_em"],
        "wrong_gradient": "All nineteen other uploads with own text fixed; also same-modality and opposite-property subsets",
        "reference_policy": "Capture evaluator alone reads targets; public inference receives only bank, observations and reconstructed tokens; evaluator reads labels after both phases commit",
        "target_manifest_sha256": file_hash(ROOT / "private/targets.json"), "frozen_hashes": hashes,
        "uncommitted": True, "scope": "20-image mixed-modality head property diagnostic; not natural prevalence or patient-independent validation",
    })
    print(read_json(ROOT / "data_audit.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inspect", "freeze"])
    args = parser.parse_args()
    try:
        {"inspect": inspect, "freeze": freeze}[args.stage]()
    except BaseException as error:
        write_json(ROOT / f"{args.stage}_failed.json", {"type": type(error).__name__, "reason": str(error)})
        raise
