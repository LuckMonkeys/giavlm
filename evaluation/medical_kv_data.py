"""Evaluator-only new SLAKE cohort review; freeze public module rule first."""

import argparse
from collections import Counter
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageOps

from core.artifacts import file_hash, read_json, write_json
from core.data import read_manifest
from evaluation.image_property_data import DATA, annotation_join, pixel_hash
from evaluation.image_property_followup_data import OLD, region
from evaluation.multilayer_property_data import TERMS


SEED = 2026110107
QUOTAS = [("MRI", "head", 8), ("CT", "abdomen", 5), ("X-Ray", "chest", 7)]


def select_public_module(public):
    candidates = {}
    for module in public["head"]:
        layer, letter = module.split("_")
        candidates[module] = {"worst": min(public[prop][module]["old_six_qa_worst_balanced_accuracy"] for prop in ("head", "abdomen")),
                              "mean": sum(public[prop][module]["old_six_qa_mean_balanced_accuracy"] for prop in ("head", "abdomen")) / 2}
    winner = max(candidates, key=lambda name: (candidates[name]["worst"], candidates[name]["mean"],
                 -int(name.split("_")[0][1:]), -("q", "k", "v").index(name.split("_")[1])))
    return {"primary": winner, "candidates": candidates,
            "selection": "Max worst BA across both public properties and all six old QA, then equal-property mean BA, shallower layer, Q/K/V order",
            "threshold": "Original raw two-prototype distance comparison; no threshold fitting",
            "target_labels_used": False}


def inspect(root):
    out = root / "medical-kv"
    if (out / "private/sampling.json").exists():
        raise FileExistsError("No resampling after scores")
    source = Path("docs/experiments/data/visual-concept-probe/qa-controls-aggregate.json")
    public_rule = select_public_module(read_json(source)["robustness"])
    public_rule["source_sha256"] = file_hash(source)
    write_json(out / "public-rule.json", public_rule)
    local, joined = annotation_join(read_manifest(DATA), OLD / "metadata_source")
    excluded, old_pixels, sources = set(), set(), {}
    for name in ("image_property_followup_gpu5_20261016", "synthetic_ct_property_gpu5_20261016",
                 "synthetic_head_n20_gpu5_20261017", "multilayer_property_20261101"):
        path = Path("outputs/diagnostics") / name / "private/targets.json"
        for row in read_json(path):
            excluded.add(row["row"]["image_id"])
            old_pixels.add(pixel_hash(Image.open(row["row"]["image"])))
        sources[str(path)] = file_hash(path)
    for path in (OLD / "private").glob("rank-*.json"):
        for row in read_json(path):
            key = row["image_id"]
            excluded.add(key)
            old_pixels.add(pixel_hash(Image.open(local[key][0]["image"])))
        sources[str(path)] = file_hash(path)
    rng = random.Random(SEED)
    rows, strata = [], []
    for modality, body, count in QUOTAS:
        eligible = sorted(key for key, info in joined.items() if key not in excluded and len(info["matches"]) == 1
                          and local[key][0]["split"] == "eval" and info["label"] == modality and region(info["locations"]) == body)
        rng.shuffle(eligible)
        if len(eligible) < count:
            raise ValueError("Insufficient new images in frozen modality/region stratum")
        strata.append({"modality": modality, "region": body, "count": count, "available": len(eligible)})
        for offset, key in enumerate(eligible[:count + 1]):
            item = random.Random(f"{SEED}:{key}").choice(sorted(local[key], key=lambda row: row["sample_id"]))
            image = Image.open(item["image"])
            pixel = pixel_hash(image)
            if pixel in old_pixels:
                raise ValueError("New sample duplicates a previously tested image by pixels")
            text = item["question"] + " " + item["target"]
            answer = item["target"].strip().lower().rstrip(".!? ")
            rows.append({"inspection_index": len(rows), "region": body, "modality": modality,
                         "reserve": offset >= count, "row": item, "pixel_sha256": pixel,
                         "literal_terms": {prop: bool(pattern.search(text)) for prop, pattern in TERMS.items()},
                         "answer_type": "yes_no" if answer in ("yes", "no") else "other",
                         "annotation": joined[key]})
    write_json(out / "private/sampling.json", {"seed": SEED, "strata": strata, "candidates": rows,
               "excluded_ids": sorted(excluded), "exclusion_sources": sources, "manifest_sha256": file_hash(DATA),
               "selection": "One deterministic random unmodified original English QA per new eval image; no QA-type filtering",
               "limitation": "Remaining eval has no unused MRI abdomen. New Head is MRI, Abdomen CT, Chest X-Ray; modality and anatomy remain confounded."})
    for start in range(0, len(rows), 12):
        sheet = Image.new("RGB", (1200, 330 * ((min(12, len(rows) - start) + 3) // 4)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(rows[start:start + 12]):
            x, y = index % 4 * 300, index // 4 * 330
            image = Image.open(item["row"]["image"]).convert("RGB")
            sheet.paste(ImageOps.contain(image, (290, 280)), (x + 5, y + 42))
            draw.text((x + 5, y + 8), f"{item['inspection_index']} {item['modality']} {item['region']}"
                      + (" RESERVE" if item["reserve"] else ""), fill="black")
        sheet.save(out / "private" / f"inspection-{start//12}.jpg", quality=95)
    write_json(out / "sampling-audit.json", {"stage": "awaiting private visual review", "requested_new_images": 20,
               "excluded_previous_property_images": len(excluded), "strata": strata, "public_rule": public_rule})
    print({"primary": public_rule["primary"], "new_images": 20, "review_candidates": len(rows), "strata": strata})


def freeze(root):
    out = root / "medical-kv"
    if (out / "private/targets.json").exists():
        raise FileExistsError("Reviewed target cohort already frozen")
    sampling = read_json(out / "private/sampling.json")
    review = read_json(out / "private/visual-review.json")
    by_id = {row["inspection_index"]: row for row in sampling["candidates"]}
    chosen = [dict(by_id[index]) for index in review["selected_indices"]]
    if Counter((row["modality"], row["region"]) for row in chosen) != {(m, b): n for m, b, n in QUOTAS}:
        raise ValueError("Visual review does not match frozen selection quotas")
    for row in chosen:
        correction = review.get("region_overrides", {}).get(str(row["inspection_index"]))
        if correction:
            row["annotation_region"] = row["region"]
            row["region"] = correction["region"]
            row["visual_review_reason"] = correction["reason"]
    if len({row["pixel_sha256"] for row in chosen}) != 20:
        raise ValueError("Pixel duplicates in new cohort")
    random.Random(SEED + 1).shuffle(chosen)
    targets = [{**row, "run": index} for index, row in enumerate(chosen)]
    write_json(out / "private/targets.json", targets)
    write_json(out / "cohort-audit.json", {"n": len(targets), "split": "eval",
        "region_counts": dict(Counter(row["region"] for row in targets)),
        "modality_counts": dict(Counter(row["modality"] for row in targets)),
        "answer_type_counts": dict(Counter(row["answer_type"] for row in targets)),
        "literal_terms_counts": {prop: sum(row["literal_terms"][prop] for row in targets) for prop in TERMS},
        "previous_image_id_overlap": 0, "previous_pixel_overlap": 0,
        "previous_property_image_groups_excluded": len(sampling["excluded_ids"]),
        "patient_independence": "Unknown; only image ID and decoded pixel isolation verified",
        "annotation_quotas": [{"modality": m, "region": body, "n": n} for m, body, n in QUOTAS],
        "pre_gradient_visual_overrides": len(review.get("region_overrides", {})),
        "attribute_definition": "Primary visible anatomical region; lower chest images with small upper-abdomen tissue are non-abdomen by dominant field of view, not a claim that no abdominal tissue is present",
        "confound": sampling["limitation"], "source": "Original SLAKE unique annotation join and private pre-inference visual review",
        "targets_sha256": file_hash(out / "private/targets.json"), "review_sha256": file_hash(out / "private/visual-review.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        (inspect if args.stage == "inspect" else freeze)(args.root.resolve())
    except BaseException:
        import traceback
        write_json(args.root / "medical-kv" / f"{args.stage}-failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
