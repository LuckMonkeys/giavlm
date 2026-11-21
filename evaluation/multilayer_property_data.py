"""Evaluator-only fresh region cohort and target-independent abdomen generation plan."""

import argparse
from collections import Counter
from pathlib import Path
import random
import re

from PIL import Image, ImageDraw, ImageOps

from core.artifacts import file_hash, read_json, write_json
from core.data import read_manifest
from evaluation.image_property_data import DATA, annotation_join, pixel_hash
from evaluation.image_property_followup_data import OLD, region


QUOTAS = [("CT", "head", 1), ("MRI", "head", 4), ("CT", "abdomen", 10),
          ("MRI", "chest", 1), ("X-Ray", "chest", 4)]
SEED = 20261101
TERMS = {"head": re.compile(r"head|brain|skull|cran|cerebr|cerebell|ventric|pituitary|thalam", re.I),
         "abdomen": re.compile(r"abdom|liver|hepat|kidney|renal|spleen|splen|stomach|gastr|"
                               r"pancrea|bowel|intestin|gallbladder", re.I)}


def prepare_targets(root):
    """Select by original metadata before any new gradient or property score exists."""
    destination = root / "private/sampling.json"
    if destination.exists():
        raise FileExistsError("No resampling after inspecting scores")
    local, joined = annotation_join(read_manifest(DATA), OLD / "metadata_source")
    excluded, sources = set(), {}
    for name in ["image_property_followup_gpu5_20261016", "synthetic_ct_property_gpu5_20261016",
                 "synthetic_head_n20_gpu5_20261017"]:
        path = Path("outputs/diagnostics") / name / "private/targets.json"
        excluded.update(row["row"]["image_id"] for row in read_json(path))
        sources[str(path)] = file_hash(path)
    for path in (OLD / "private").glob("rank-*.json"):
        excluded.update(row["image_id"] for row in read_json(path))
        sources[str(path)] = file_hash(path)
    rng = random.Random(SEED)
    candidates, strata = [], []
    for modality, body, count in QUOTAS:
        eligible = sorted(key for key, info in joined.items() if key not in excluded
                          and local[key][0]["split"] == "eval" and len(info["matches"]) == 1
                          and info["label"] == modality and region(info["locations"]) == body)
        rng.shuffle(eligible)
        if len(eligible) < count:
            raise ValueError(f"Insufficient fresh images for {modality}, {body}")
        strata.append({"modality": modality, "region": body, "count": count, "available": len(eligible)})
        for index, key in enumerate(eligible[:count + 2]):
            row = random.Random(f"{SEED}:{key}").choice(sorted(local[key], key=lambda value: value["sample_id"]))
            text = row["question"] + " " + row["target"]
            candidates.append({"inspection_index": len(candidates), "modality": modality, "region": body,
                               "reserve": index >= count, "row": row, "annotation": joined[key],
                               "pixel_sha256": pixel_hash(Image.open(row["image"])),
                               "literal_terms": {name: bool(pattern.search(text)) for name, pattern in TERMS.items()}})
    write_json(destination, {"seed": SEED, "strata": strata, "candidates": candidates,
                             "excluded_ids": sorted(excluded), "exclusion_sources": sources,
                             "selection": "Metadata-stratified random fresh eval images; random unmodified original QA",
                             "manifest_sha256": file_hash(DATA)})
    for start in range(0, len(candidates), 15):
        chunk = candidates[start:start + 15]
        sheet = Image.new("RGB", (1500, 340 * ((len(chunk) + 4) // 5)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(chunk):
            x, y = index % 5 * 300, index // 5 * 340
            image = Image.open(item["row"]["image"]).convert("RGB")
            sheet.paste(ImageOps.contain(image, (290, 295)), (x + 5, y + 36))
            draw.text((x + 4, y + 8), f"{item['inspection_index']} {item['modality']} {item['region']}"
                      + (" RESERVE" if item["reserve"] else ""), fill="black")
        sheet.save(root / "private" / f"inspection-{start // 15}.jpg", quality=95)
    write_json(root / "sampling-status.json", {"stage": "awaiting visual review", "desired_targets": 20,
               "excluded_prior_image_groups": len(excluded), "strata": strata,
               "confound": "All fresh abdomen positives are CT; head mostly MRI and chest mostly X-Ray. Image ID isolation does not establish patient independence."})


def freeze_targets(root):
    sampling = read_json(root / "private/sampling.json")
    review = read_json(root / "private/visual_review.json")
    if (root / "private/targets.json").exists():
        raise FileExistsError("No overwrite of reviewed cohort")
    by_index = {item["inspection_index"]: item for item in sampling["candidates"]}
    chosen = []
    for index in review["selected_indices"]:
        item = dict(by_index[index])
        if str(index) in review.get("region_overrides", {}):
            correction = review["region_overrides"][str(index)]
            item["annotation_region"] = item["region"]
            item["region"] = correction["region"]
            item["visual_correction_reason"] = correction["reason"]
        chosen.append(item)
    if Counter(item["region"] for item in chosen) != {"head": 5, "abdomen": 10, "chest": 5}:
        raise ValueError("Visual review changed the predeclared region totals")
    if len({item["pixel_sha256"] for item in chosen}) != 20:
        raise ValueError("New cohort contains pixel duplicates")
    rng = random.Random(SEED + 1)
    rng.shuffle(chosen)
    targets = [{**item, "run": index} for index, item in enumerate(chosen)]
    write_json(root / "private/targets.json", targets)
    write_json(root / "cohort-audit.json", {
        "n": 20, "region_counts": dict(Counter(item["region"] for item in targets)),
        "metadata_sampling_strata": sampling["strata"],
        "final_strata": [{"modality": modality, "region": body, "count": count}
                         for (modality, body), count in sorted(Counter(
                             (item["modality"], item["region"]) for item in targets).items())],
        "prior_image_id_overlap": 0,
        "unique_pixels": 20, "excluded_prior_image_groups": len(sampling["excluded_ids"]),
        "label_source": "Original unique SLAKE location metadata and pre-inference visual review",
        "patient_independence": "Unknown; image grouping only",
        "target_split": "eval", "natural_qa": "One original random QA per image, no text-content filter",
        "text_literal_counts": {name: sum(item["literal_terms"][name] for item in targets) for name in TERMS},
        "confound": "Head positives are MRI only, abdomen mostly CT, chest X-Ray only; cross-modality accuracy cannot isolate region from modality. Both region and modality subgroup reporting required.",
        "targets_sha256": file_hash(root / "private/targets.json"),
        "review_sha256": file_hash(root / "private/visual_review.json")})


def abdomen_cases():
    styles = {"photo": "A realistic studio photograph.",
              "line": "A clean monochrome contour line drawing with simple flat gray shading.",
              "render": "An unmistakably stylized matte gray clay sculpture, monochrome, no photographic skin texture."}
    garments = ["a plain fitted cotton T-shirt", "a plain fitted knitted top", "a plain fitted sports jersey",
                "a plain fitted long-sleeve shirt", "a plain fitted polo shirt", "a plain fitted cotton sweater",
                "a plain fitted crew-neck top", "a plain fitted athletic top"]
    rows = []
    for index in range(16):
        side = ["front", "left three-quarter", "right three-quarter", "side"][index % 4]
        person = "an adult man" if index % 2 == 0 else "an adult woman"
        for style, description in styles.items():
            rows.append({"id": f"abdomen-{index:02d}-{style}", "group": f"abdomen-{index:02d}",
                         "label": "abdomen", "style": style, "split": "fit" if index < 8 else "dev",
                         "seed": SEED * 100 + index,
                         "prompt": f"{description} A tightly cropped {side} view of the abdominal waist region of {person}, wearing {garments[index % 8]} and opaque trousers. The frame extends only from the lower ribs to the upper hips; the covered belly and waist occupy most of the image. Arms are outside the frame. The head, face, neck, upper chest, hands and legs are outside the frame. Fully clothed, modest presentation. Plain light gray background, balanced soft lighting. One subject, no text, no labels, no collage."})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["inspect", "freeze"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    (prepare_targets if args.stage == "inspect" else freeze_targets)(args.root)


if __name__ == "__main__":
    main()
