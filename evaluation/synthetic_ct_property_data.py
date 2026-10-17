"""Freeze evaluator-reviewed CT targets and a target-independent synthetic bank."""

from collections import Counter
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, write_json, write_tensors
from evaluation.image_property_data import pixel_hash
from evaluation.image_property_followup_data import CONFIG, MODEL, OLD
from evaluation.synthetic_ct_property import ROOT


LOCATION = re.compile(
    r"brain|head|skull|cran|cerebr|cerebell|ventric|neck|face|abdomen|abdominal|"
    r"liver|hepatic|kidney|renal|spleen|splenic|stomach|gastric|pancrea|bowel|"
    r"intestin|lung|chest|thorax|heart|cardiac|pelvi|bladder", re.I)
MODALITY = re.compile(
    r"\bct\b|\bmri\b|x[ -]?ray|modality|computed tomography|magnetic resonance|"
    r"type of image|kind of image|imaging", re.I)


def prepare():
    if (ROOT / "plan.json").exists():
        raise FileExistsError("Plan already frozen")
    initial = read_json(ROOT / "private/inspection_candidates.json")
    replacements = read_json(ROOT / "private/replacement_candidates.json")
    # Selected by location inspection and availability of an original modality QA;
    # neither observed gradients nor any attack scores have been computed yet.
    selected = [initial[i] for i in [0, 1, 2, 3, 5, 7, 8, 10]] + [replacements[i] for i in [1, 2]]
    targets = []
    for item in selected:
        choices = [r for r in item["rows"] if not LOCATION.search(r["question"] + " " + r["target"])
                   and MODALITY.search(r["question"])]
        if not choices:
            raise ValueError("Missing original location-neutral modality QA")
        choices.sort(key=lambda r: (r["target"].strip().lower().rstrip(".") not in
                                   ["ct", "computed tomography"], len(r["question"]), r["sample_id"]))
        targets.append({"run": len(targets), "label": item["region"], "row": choices[0],
                        "annotation": item["annotation"]})
    if Counter(r["label"] for r in targets) != {"head": 5, "abdomen": 5}:
        raise ValueError("Unbalanced targets")
    hashes = {pixel_hash(Image.open(r["row"]["image"])) for r in targets}
    if len(hashes) != 10:
        raise ValueError("Duplicate target pixels")
    write_json(ROOT / "private/targets.json", targets)
    write_json(ROOT / "private/label_review.json", {
        "reviewer": "assistant visual inspection plus original SLAKE metadata",
        "selected_initial_indices": [0, 1, 2, 3, 5, 7, 8, 10], "selected_replacement_indices": [1, 2],
        "excluded_initial": {"4": "predominantly neck", "9": "predominantly chest",
                             "6": "no original location-neutral modality QA"},
        "excluded_replacement": {"0": "no original location-neutral modality QA"},
        "clinical_validation": False, "selection_before_gradient_capture": True,
        "patient_independence": "unknown; several head slices look similar",
    })
    generation = read_json(ROOT / "generated/complete.json")
    if generation["status"] != "complete" or len(generation["images"]) != 10:
        raise ValueError("Generation incomplete")
    bank, raw, bank_hashes = [], {}, {}
    for row in generation["images"]:
        path = ROOT / "generated" / row["file"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Generated image changed")
        with Image.open(path) as image:
            image = image.convert("RGB")
            pixel = pixel_hash(image)
            if pixel in hashes:
                raise ValueError("Generated and target pixels overlap")
            name = f"candidate-{len(bank):02d}"
            raw[name] = torch.from_numpy(np.array(image, copy=True))
        bank.append({"id": name, "label": row["case"], "seed": row["seed"],
                     "pixel_sha256": pixel, "generated_file": row["file"]})
        bank_hashes[str(path)] = file_hash(path)
    if Counter(r["label"] for r in bank) != {"head": 5, "abdomen": 5}:
        raise ValueError("Unbalanced bank")
    write_json(ROOT / "public/bank.json", bank)
    write_tensors(ROOT / "public/bank.safetensors", raw)
    write_json(ROOT / "public/visual_review.json", {
        "reviewer": "assistant contact-sheet inspection before gradient scoring",
        "retained": 10, "rejected": 0, "property": "head vs abdomen CT-style axial appearance",
        "assessment": "All five head and five abdomen outputs have the requested broad location and CT-like grayscale appearance. No obvious text or collage.",
        "limitation": "Fine anatomy is artificial, especially one head output with an anomalous central bright structure; location label only, not clinical validity. Same-prompt diversity limited.",
        "target_specific_prompt_details": False,
    })
    old_ids = {r["image_id"] for p in (OLD / "private").glob("rank-*.json") for r in read_json(p)}
    prior_root = Path("outputs/diagnostics/image_property_followup_gpu5_20261016")
    old_ids |= {r["row"]["image_id"] for r in read_json(prior_root / "private/targets.json")}
    write_json(ROOT / "data_audit.json", {
        "target_counts": dict(Counter(r["label"] for r in targets)),
        "bank_counts": dict(Counter(r["label"] for r in bank)), "unique_target_pixels": len(hashes),
        "unique_bank_pixels": len({r["pixel_sha256"] for r in bank}),
        "bank_target_pixel_overlap": 0,
        "unique_original_qa": len({(r["row"]["question"], r["row"]["target"]) for r in targets}),
        "explicit_location_mentions": 0, "target_modality": "CT", "target_split": "eval",
        "prior_pilot_image_overlap": sum(r["row"]["image_id"] in old_ids for r in targets),
        "selection": "Metadata and visual location inspection, then original modality QA eligibility; no gradient-based sample selection",
        "same_patient_exclusion": "unavailable; independent image IDs do not establish independent patients",
    })
    files = [MODEL / "model.json", MODEL / "model.safetensors", CONFIG,
             ROOT / "private/targets.json", ROOT / "public/bank.json", ROOT / "public/bank.safetensors",
             ROOT / "public/visual_review.json", ROOT / "generated/plan.json", ROOT / "generated/case_provider.json",
             Path("evaluation/synthetic_ct_property.py"), Path(__file__),
             Path("evaluation/image_property_study.py"), Path("evaluation/image_property_scoring.py"),
             Path("evaluation/dager_rank_study.py"), Path("attacks/analytic/dager_adapted.py")]
    frozen = {str(p): file_hash(p) for p in files}
    frozen.update(bank_hashes)
    write_json(ROOT / "plan.json", {
        "schema_version": 1, "classes": ["head", "abdomen"], "targets": 10,
        "model": str(MODEL), "config": str(CONFIG), "rank": 32, "round": 10,
        "physical_gpu": 5, "lm_prior_weight": 0.01, "dager_image_source": "random",
        "primary_score": "centered", "secondary_scores": ["global", "equal_tensor", "centered_equal_tensor"],
        "top_m": 3, "class_aggregation": "mean of lowest three candidate losses among five per class",
        "text_only_prompt": "USER: Infer the body region from the question and answer. Choose head or abdomen.\nQuestion: {question}\nAnswer: {answer}\nASSISTANT:",
        "text_only_calibration": "none; fixed prompt, no tuning on these targets",
        "wrong_gradient": "all nine other observed updates per target, text and bank fixed; additionally separate same and opposite property classes at evaluation",
        "text_conditions": ["oracle", "reconstructed"], "run_both_regardless_of_oracle_accuracy": True,
        "target_manifest_sha256": file_hash(ROOT / "private/targets.json"), "frozen_hashes": frozen,
        "reference_policy": "Private evaluator captures updates and labels; inference takes only bank, Observation and reconstructed tokens. Property labels opened only after both prediction phases commit.",
        "scope": "CT-only head/abdomen pilot, not a joint modality-and-region detector", "uncommitted": True,
    })
    print(read_json(ROOT / "data_audit.json"), flush=True)


if __name__ == "__main__":
    try:
        prepare()
    except BaseException as error:
        write_json(ROOT / "prepare_failed.json", {"type": type(error).__name__, "reason": str(error)})
        raise
