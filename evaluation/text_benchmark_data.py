"""Private-side data census, frozen sampling, and scheduled VQAv2 image fetches.

Image materialization follows the exact training RNG schedule on the complete
metadata pool; it never restricts that pool or depends on attack results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
from pathlib import Path
import traceback
import urllib.request

import numpy as np
from PIL import Image

from core.artifacts import file_hash, read_json, write_json
from core.config import TrainingSpec
from core.data import read_manifest, write_manifest


LENGTH_BINS = ((1, 16), (17, 32), (33, 64))
TOKENIZERS = {
    "llava": "/home/zx/public/model-hub/huggingface/llava-hf/llava-1.5-7b-hf",
    "qwen35": "/home/zx/public/model-hub/huggingface/Qwen/Qwen3.5-4B"}


def tokenizer_fingerprints():
    filenames = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.jinja")
    return {name: {str(Path(root) / filename): file_hash(Path(root) / filename)
                   for filename in filenames if (Path(root) / filename).is_file()}
            for name, root in TOKENIZERS.items()}


def prepare_vqav2_metadata(root):
    """Full official train/val metadata; images are materialized separately."""
    manifest = root / "samples.jsonl"
    if manifest.exists():
        raise FileExistsError(manifest)
    records = []
    for split in ("train2014", "val2014"):
        questions = {r["question_id"]: r for r in read_json(
            root / "source" / f"v2_OpenEnded_mscoco_{split}_questions.json")["questions"]}
        annotations = read_json(root / "source" / f"v2_mscoco_{split}_annotations.json")["annotations"]
        if len(questions) != len(annotations):
            raise ValueError("VQAv2 questions and annotations differ in size")
        for annotation in annotations:
            question = questions[annotation["question_id"]]
            image_id = annotation["image_id"]
            if question["image_id"] != image_id:
                raise ValueError("VQAv2 question/annotation image mismatch")
            records.append({
                "sample_id": f"vqa:{annotation['question_id']}", "image_id": f"coco:{image_id}",
                "task": "vqa", "image": str(root / "images" / split / f"COCO_{split}_{image_id:012d}.jpg"),
                "question": question["question"], "target": annotation["multiple_choice_answer"],
                "references": [a["answer"] for a in annotation["answers"]],
                "source": "vqav2", "source_split": split})
    write_manifest(manifest, records, seed=42, clients=1)
    metadata = read_json(manifest.with_suffix(".meta.json"))
    metadata.update(image_materialization="exact_training_schedule_and_frozen_evaluation_only",
                    sampling_pool="all_official_train_and_val_annotations",
                    source_files=read_json(root / "source" / "sources.json"))
    write_json(manifest.with_suffix(".meta.json"), metadata)
    return manifest


def training_schedule(rows, training):
    clients = {}
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        if row["split"] == "train":
            clients.setdefault(row["client"], []).append(row)
    result = []
    for round_id in range(1, training.rounds + 1):
        rng = np.random.default_rng(training.seed + round_id)
        selected = rng.choice(sorted(clients), training.clients_per_round, replace=False)
        sampled = []
        for client in selected:
            pool = clients[client]
            indices = rng.choice(len(pool), training.sample_count, replace=len(pool) < training.sample_count)
            sampled.extend(pool[i] for i in indices)
        result.append({"round": round_id, "sample_ids": [r["sample_id"] for r in sampled]})
    return result


def select_evaluation(rows, lengths, per_bin=10, seed=42):
    """Long bins first; no shared image or identical QA across selected rows."""
    selected, summaries, images, questions = [], [], set(), set()
    for low, high in reversed(LENGTH_BINS):
        pool = [r for r in rows if r["split"] == "eval"
                and low <= sum(lengths[r["sample_id"]]) <= high]
        pool.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['sample_id']}".encode()).digest())
        chosen = []
        for row in pool:
            qa = (row["question"], row["target"])
            if row["image_id"] in images or qa in questions:
                continue
            chosen.append(row["sample_id"])
            images.add(row["image_id"])
            questions.add(qa)
            selected.append({"sample_id": row["sample_id"], "length_bin": f"{low}-{high}"})
            if len(chosen) == per_bin:
                break
        summaries.append({"length_bin": f"{low}-{high}", "eligible_rows": len(pool),
                          "eligible_images": len({r["image_id"] for r in pool}),
                          "selected_n": len(chosen), "requested_n": per_bin})
    return selected, list(reversed(summaries))


def content_lengths(rows, tokenizer):
    texts = sorted({r[field] for r in rows for field in ("question", "target")})
    lengths = {}
    for start in range(0, len(texts), 4096):
        batch = texts[start:start + 4096]
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        lengths.update(zip(batch, map(len, encoded), strict=True))
    return {r["sample_id"]: [lengths[r["question"]], lengths[r["target"]]] for r in rows}


def materialize_images(rows, output):
    paths = {r["image"]: r for r in rows}
    def fetch(item):
        filename, row = item
        path = Path(filename)
        url = f"http://images.cocodataset.org/{row['source_split']}/{path.name}"
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".jpg.part")
                with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                with Image.open(temporary) as image:
                    image.verify()
                temporary.rename(path)
            with Image.open(path) as image:
                image.verify()
            return {"image": str(path), "sha256": file_hash(path), "url": url}
        except Exception:
            write_json(output.parent / (path.stem + ".failed.json"),
                       {"status": "failed", "url": url, "traceback": traceback.format_exc()})
            raise
    with ThreadPoolExecutor(max_workers=4) as pool:
        fetched = list(pool.map(fetch, sorted(paths.items())))
    write_json(output, {"policy": "scheduled_images_from_complete_pool", "images": fetched})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare-vqav2", action="store_true")
    parser.add_argument("--materialize-vqav2", action="store_true")
    parser.add_argument("--images-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.images_only:
        frozen = read_json(args.output / "vqav2" / "frozen.json")
        if file_hash(frozen["manifest"]) != frozen["manifest_sha256"]:
            raise ValueError("Frozen VQAv2 metadata changed")
        used = {r["sample_id"] for r in frozen["selected"]}
        used.update(sid for r in frozen["training_schedule"] for sid in r["sample_ids"])
        rows = read_manifest(frozen["manifest"], task="vqa")
        materialize_images([r for r in rows if r["sample_id"] in used],
                           args.output / "vqav2" / "materialized-images.json")
        return
    if args.prepare_vqav2:
        prepare_vqav2_metadata(Path("data/vqav2"))
    from transformers import AutoTokenizer
    tokenizers = {name: AutoTokenizer.from_pretrained(path, local_files_only=True)
                  for name, path in TOKENIZERS.items()}
    training = TrainingSpec(algorithm="fedavg", local_optimizer="adamw", batch_size=2,
                            gradient_accumulation_steps=4, local_steps=10, rounds=10,
                            clients=1, clients_per_round=1, seed=42, lr=.0001,
                            lora_rank=32, lora_alpha=64)
    for dataset in ("slake", "vqa_rad", "vqav2"):
        destination = args.output / dataset / "frozen.json"
        if destination.exists():
            raise FileExistsError(destination)
        manifest = Path("data") / dataset / "samples.jsonl"
        rows = read_manifest(manifest, task="vqa")
        lengths = {name: content_lengths(rows, tok) for name, tok in tokenizers.items()}
        if any(q > 63 or a > 63 for values in lengths.values() for q, a in values.values()):
            raise ValueError(f"{dataset} exceeds Q64/A64 capacity; audit before freezing")
        selected, bins = select_evaluation(rows, lengths["llava"])
        schedule = training_schedule(rows, training)
        selected_ids = {r["sample_id"] for r in selected}
        used_ids = selected_ids | {sid for r in schedule for sid in r["sample_ids"]}
        frozen = {
            "schema_version": 1, "private_side_only": True, "dataset": dataset,
            "manifest": str(manifest.resolve()), "manifest_sha256": file_hash(manifest),
            "reference_tokenizer": "llava", "tokenizers": TOKENIZERS,
            "tokenizer_files": tokenizer_fingerprints(),
            "capacity": {"question_length": 64, "target_length": 64},
            "training": asdict(training), "training_schedule": schedule,
            "selected": [{**r, "native_lengths": {name: values[r["sample_id"]]
                                                     for name, values in lengths.items()}}
                         for r in selected],
            "bins": bins, "sampling_pool_n": len(rows),
            "split_images": {split: len({r["image_id"] for r in rows if r["split"] == split})
                             for split in ("train", "tune", "eval")},
            "max_lengths": {name: {"question": max(q for q, a in values.values()),
                                   "target": max(a for q, a in values.values())}
                            for name, values in lengths.items()}}
        write_json(destination, frozen)
        print(dataset, bins, flush=True)
        if dataset == "vqav2" and args.materialize_vqav2:
            materialize_images([r for r in rows if r["sample_id"] in used_ids],
                               args.output / dataset / "materialized-images.json")


if __name__ == "__main__":
    main()
