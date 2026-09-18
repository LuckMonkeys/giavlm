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


MEDICAL_VQA_SOURCES = {
    "vqa_rad": "flaviagiammarino/vqa-rad",
    "slake": "mdwiratathya/SLAKE-vqa-english",
}


def _canonical_image_id(image):
    """Content hash of the decoded pixels.

    Neither corpus ships an image identifier, but both repeat one image across
    several questions. Hashing the decoded bytes makes those rows share an
    image_id, so `assignment` keeps them inside one split and client and
    `read_manifest` does not reject the manifest.
    """
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.size[0]}x{image.size[1]}:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def prepare_medical_vqa(output, sources=("vqa_rad",), seed=42, clients=10, limit=None,
                        cache_dir=None):
    """Normalize medical VQA corpora into the benchmark manifest schema.

    Upstream train/validation/test splits are provenance only; the benchmark
    re-splits deterministically by canonical image id. Each row carries a single
    reference answer, so leave-one-out VQA consensus scoring degenerates to exact
    match on this data.
    """
    from datasets import load_dataset

    unknown = set(sources) - set(MEDICAL_VQA_SOURCES)
    if unknown:
        raise ValueError(f"Unknown medical VQA source(s): {sorted(unknown)}")
    output = Path(output)
    manifest = output / "samples.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    records, written = [], {}
    for slug in sources:
        dataset = load_dataset(MEDICAL_VQA_SOURCES[slug], cache_dir=cache_dir)
        for source_split in sorted(dataset):
            rows = dataset[source_split]
            for index in range(len(rows) if limit is None else min(limit, len(rows))):
                row = rows[index]
                image = row["image"].convert("RGB")
                key = _canonical_image_id(image)
                if key not in written:
                    path = image_dir / f"{key[:16]}.png"
                    image.save(path)
                    written[key] = path
                answer = str(row["answer"]).strip()
                records.append({"sample_id": f"{slug}:{source_split}:{index}",
                                "image_id": f"medvqa:{key[:16]}", "task": "vqa",
                                "image": str(written[key]), "question": str(row["question"]).strip(),
                                "target": answer, "references": [answer],
                                "source": slug, "source_split": source_split})
    if not records:
        raise ValueError("No medical VQA records were produced")
    write_manifest(manifest, records, seed, clients)
    return manifest


CANARY_GIVEN = ["Alden", "Brisco", "Corwen", "Delphy", "Ellary", "Fenwick",
                "Garrick", "Hesper", "Ilbert", "Jarrah", "Kelvyn", "Lorwen"]
CANARY_FAMILY = ["Fairbrook", "Ganthorpe", "Halvorsen", "Ingersoll", "Jessamy",
                 "Kirkwald", "Lammerton", "Mowbray", "Northcott", "Osgarth"]


def canary_entities(index, rng):
    """Synthetic patient identifiers that do not occur in any source corpus.

    The surnames are invented, so a recovered entity is evidence of leakage from
    this run rather than of the attacker's language prior over clinical text.
    """
    name = f"{CANARY_GIVEN[rng.integers(len(CANARY_GIVEN))]} {CANARY_FAMILY[rng.integers(len(CANARY_FAMILY))]}"
    mrn = f"MRN{int(rng.integers(1000000, 9999999))}"
    dob = "{:04d}-{:02d}-{:02d}".format(int(rng.integers(1930, 2010)),
                                        int(rng.integers(1, 13)), int(rng.integers(1, 29)))
    return [name, mrn, dob]


def inject_canaries(manifest, output, canary_seed=42, rate=1.0, field="question"):
    """Rewrite a prepared manifest with synthetic PII in the question or target.

    The partition seed and client count are inherited from the source manifest so
    every sample keeps its split and client. Choosing them freely here would move
    samples between partitions and silently break any comparison against the
    un-injected baseline, which is the whole point of running both.

    Injection happens before tokenization, so an entity may still be truncated
    away by the model's question/target budget. The evaluator therefore scores
    recall only over the entities that survived into the trained tokens.
    """
    if field not in {"question", "target"}:
        raise ValueError("Canaries are injected into the question or the target")
    if not 0 < rate <= 1:
        raise ValueError("Canary rate must be in (0, 1]")
    manifest = Path(manifest)
    meta_path = manifest.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}; canary injection inherits its partitioning")
    meta = json.loads(meta_path.read_text())
    seed, clients = meta["seed"], meta["clients"]
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError(f"No records in {manifest}")
    output = Path(output)
    destination = output / "samples.jsonl"
    if destination.exists():
        raise FileExistsError(destination)
    original = {row["sample_id"]: (row["split"], row["client"]) for row in rows}
    rng = np.random.default_rng(canary_seed)
    records = []
    for index, row in enumerate(sorted(rows, key=lambda r: r["sample_id"])):
        row = dict(row)
        for key in ["split", "client"]:
            row.pop(key, None)
        if rng.random() < rate:
            entities = canary_entities(index, rng)
            row["canary"] = entities
            prefix = f"patient {entities[0]} {entities[1]} dob {entities[2]}"
            row[field] = f"{prefix} {row[field]}".strip()
            if field == "target":
                row["references"] = [row["target"]]
        else:
            row["canary"] = []
        records.append(row)
    write_manifest(destination, records, seed, clients)
    moved = {r["sample_id"] for r in records if original[r["sample_id"]] != (r["split"], r["client"])}
    if moved:
        raise RuntimeError(f"Canary injection moved {len(moved)} samples between partitions")
    return destination


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
            # The adapter owns the model-native fixed view and keeps its geometry public.
            images.append(adapter.prepare_image(source))
    return adapter.batch(torch.stack(images), [r["question"] for r in rows], [r["target"] for r in rows])
