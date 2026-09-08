"""Private-side ingestion. This module is never imported by the attack runner."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from core.artifacts import file_hash, write_json


def assignment(image_id: str, seed: int, clients: int):
    key = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()
    bucket = int.from_bytes(key[:4], "big") % 100
    split = "tune" if bucket < 10 else "eval" if bucket < 30 else "train"
    return split, int.from_bytes(key[4:8], "big") % clients


def write_manifest(path, records, seed=42, clients=10):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    for row in records:
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {row['sample_id']}")
        seen.add(row["sample_id"])
        row["split"], row["client"] = assignment(row["image_id"], seed, clients)
        row["image"] = str(Path(row["image"]).resolve())
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    write_json(path.with_suffix(".meta.json"), {
        "schema_version": 1, "seed": seed, "clients": clients, "samples": len(records),
        "sha256": file_hash(path), "split_unit": "canonical_image_id",
        "image_groups": len({r["image_id"] for r in records}),
        "splits": {s: sum(r["split"] == s for r in records) for s in ["train", "tune", "eval"]}})


def synthetic(output, count=64, seed=42, clients=10):
    output = Path(output)
    manifest = output / "samples.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    rng = np.random.default_rng(seed)
    colors = {"red": (220, 35, 45), "green": (30, 210, 70), "blue": (30, 65, 230)}
    for i in range(count):
        color = list(colors)[i % 3]
        shape = "square" if i % 2 == 0 else "circle"
        image = Image.new("RGB", (32, 32), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        x, y = rng.integers(2, 12, size=2).tolist()
        if shape == "square":
            draw.rectangle((x, y, x + 14, y + 14), fill=colors[color])
        else:
            draw.ellipse((x, y, x + 14, y + 14), fill=colors[color])
        image_path = output / "images" / f"{i:06d}.png"
        image_path.parent.mkdir(exist_ok=True)
        image.save(image_path)
        common = {"image_id": f"synthetic:{i}", "image": str(image_path),
                  "source": "synthetic-fixture-v1"}
        for task in ["vqa", "caption"]:
            target = color if task == "vqa" else f"a {color} {shape}"
            records.append({**common, "sample_id": f"synthetic:{i}:{task}", "task": task,
                            "question": "what color is the object" if task == "vqa" else "",
                            "target": target, "references": [target]})
    write_manifest(manifest, records, seed, clients)
    return manifest


def prepare_coco(captions, image_root, output, questions=None, annotations=None, seed=42, clients=10):
    coco = json.loads(Path(captions).read_text())
    images = {r["id"]: str(Path(image_root) / r["file_name"]) for r in coco["images"]}
    captions_by_image = {}
    for row in coco["annotations"]:
        captions_by_image.setdefault(row["image_id"], []).append(row["caption"])
    records = []
    for row in coco["annotations"]:
        image_id = row["image_id"]
        records.append({"sample_id": f"caption:{row['id']}", "image_id": f"coco:{image_id}",
                        "task": "caption", "image": images[image_id], "question": "",
                        "target": row["caption"], "references": captions_by_image[image_id],
                        "source": "coco-captions"})
    if bool(questions) != bool(annotations):
        raise ValueError("VQAv2 requires both questions and annotations")
    if questions:
        qa = {r["question_id"]: r for r in json.loads(Path(questions).read_text())["questions"]}
        for row in json.loads(Path(annotations).read_text())["annotations"]:
            question = qa[row["question_id"]]
            image_id = row["image_id"]
            if image_id not in images:
                raise ValueError(f"VQA image {image_id} missing in supplied COCO image metadata")
            records.append({"sample_id": f"vqa:{row['question_id']}", "image_id": f"coco:{image_id}",
                            "task": "vqa", "image": images[image_id],
                            "question": question["question"], "target": row["multiple_choice_answer"],
                            "references": [a["answer"] for a in row["answers"]], "source": "vqav2"})
    missing = [p for p in set(images.values()) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[0]}")
    write_manifest(output, records, seed, clients)
    return Path(output)


def read_manifest(path, task=None, split=None, client=None, unique_images=False):
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    groups = {}
    for row in records:
        value = (row["split"], row["client"])
        if row["image_id"] in groups and groups[row["image_id"]] != value:
            raise ValueError("Image group crosses a split or client boundary")
        groups[row["image_id"]] = value
    records = [r for r in records if (task is None or r["task"] == task)
               and (split is None or r["split"] == split) and (client is None or r["client"] == client)]
    records.sort(key=lambda r: r["sample_id"])
    if unique_images:
        seen, unique = set(), []
        for row in records:
            if row["image_id"] not in seen:
                seen.add(row["image_id"])
                unique.append(row)
        records = unique
    return records


def load_batch(rows, adapter):
    images = []
    for row in rows:
        with Image.open(row["image"]) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            # Fixed square view prevents private aspect ratios leaking via public grid metadata.
            image = ImageOps.fit(source, (adapter.spec.image_size, adapter.spec.image_size),
                                 method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
            images.append(torch.from_numpy(np.array(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255)
    return adapter.batch(torch.stack(images), [r["question"] for r in rows], [r["target"] for r in rows])
