"""Private-reference image-ranking diagnostic under unknown or partially known text.

The attacker-visible score is always gradient matching loss. Private image/text are
used only to create the victim update and, after scores are committed, to evaluate
candidate ordering. Text guesses come from the public spec and are fixed across all
image candidates within a replicate.
"""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from scipy.stats import rankdata, spearmanr
import torch

from core.artifacts import file_hash, read_json, source_fingerprint, write_json
from core.config import KNOWLEDGE_FIELDS, digest, load_config
from core.data import read_manifest
from core.types import Batch, Observation
from evaluation.discriminability_metrics import (bootstrap, candidate_statistics, candidates,
                                                  image_scores, replay, update_scores)


SCHEMA = 1
KNOWLEDGE = ("text_known", "question_known", "target_known", "private")
AGGREGATES = ("mean_rank", "mean_loss", "median_loss", "min_loss")


def load_spec(path):
    spec = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    allowed = {"schema_version", "manifest", "seed", "pilot_count", "development_count",
               "validation_count", "protocol", "conditions", "knowledge_conditions",
               "candidate_ids", "text_guesses", "resources"}
    if not isinstance(spec, dict) or set(spec) - allowed or spec.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported text-unknown diagnostic schema or unknown fields")
    required = allowed - {"seed", "pilot_count", "development_count", "validation_count"}
    if not required <= set(spec):
        raise ValueError(f"Missing text-unknown fields: {sorted(required - set(spec))}")
    for key, default in [("seed", 42), ("pilot_count", 3), ("development_count", 20),
                         ("validation_count", 20)]:
        spec.setdefault(key, default)
    for key in ["pilot_count", "development_count", "validation_count"]:
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if spec["development_count"] < spec["pilot_count"]:
        raise ValueError("Development must contain every pilot image")
    if spec["knowledge_conditions"] != list(KNOWLEDGE):
        raise ValueError(f"Knowledge conditions must be exactly {list(KNOWLEDGE)}")
    if (not isinstance(spec["candidate_ids"], list) or "truth" not in spec["candidate_ids"] or
            len(set(spec["candidate_ids"])) != len(spec["candidate_ids"])):
        raise ValueError("candidate_ids must be unique and include truth")
    guesses = spec["text_guesses"]
    if set(guesses) != {"per_family", "template_pairs", "question_words", "target_words",
                       "question_word_count", "target_word_count"}:
        raise ValueError("Invalid text_guesses fields")
    if type(guesses["per_family"]) is not int or guesses["per_family"] < 1:
        raise ValueError("per_family must be positive")
    if len(guesses["template_pairs"]) < guesses["per_family"]:
        raise ValueError("Insufficient public template pairs")
    if any(set(pair) != {"question", "target"} or not pair["question"] or not pair["target"]
           for pair in guesses["template_pairs"]):
        raise ValueError("Each template pair needs nonempty question and target")
    for key in ["question_words", "target_words"]:
        if not isinstance(guesses[key], list) or len(guesses[key]) < 2:
            raise ValueError(f"{key} needs at least two public words")
    resources = spec["resources"]
    if set(resources) != {"gpu_id", "min_free_mib"} or resources["gpu_id"] != 4:
        raise ValueError("This diagnostic is fixed to physical GPU 4")
    if not isinstance(resources["min_free_mib"], (int, float)) or resources["min_free_mib"] <= 0:
        raise ValueError("Invalid GPU memory requirement")
    names = []
    for condition in spec["conditions"]:
        if set(condition) - {"name", "strategy", "round", "snapshot"}:
            raise ValueError("Unknown model-condition field")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", condition["name"]):
            raise ValueError("Invalid model-condition name")
        if condition["round"] and not condition.get("snapshot"):
            raise ValueError("Trained model condition requires a snapshot")
        names.append(condition["name"])
        cfg = configuration(spec, condition, "cpu")
        if cfg.model.device_map or cfg.model.max_memory:
            raise ValueError("Multi-device placement is forbidden")
    if not names or len(set(names)) != len(names):
        raise ValueError("Model-condition names must be unique")
    return spec


def configuration(spec, condition, device):
    overrides = [f"{section}.{key}={json.dumps(value)}"
                 for section, values in spec["protocol"].items() for key, value in values.items()]
    overrides += [f"model.device={device}", "training.knowledge=private",
                  f"training.fine_tuning_strategy={condition['strategy']}",
                  f"training.server_round={condition['round']}"]
    cfg = load_config(overrides=overrides)
    train = cfg.training
    if (train.algorithm != "fedsgd" or train.sample_count != 1 or train.task != "vqa" or
            train.token_lengths_known):
        raise ValueError("Diagnostic requires VQA, single-sample single-step FedSGD without lengths")
    return cfg


def ordered(rows, seed):
    return sorted(rows, key=lambda row: digest([seed, row["image_id"]]))


def select_samples(spec):
    tune = read_manifest(spec["manifest"], "vqa", "tune", unique_images=True)
    evaluation = read_manifest(spec["manifest"], "vqa", "eval", unique_images=True)
    pilot = tune[:spec["pilot_count"]]
    used = {row["image_id"] for row in pilot}
    development = pilot + ordered([row for row in tune if row["image_id"] not in used],
                                  spec["seed"])[:spec["development_count"] - len(pilot)]
    validation = ordered(evaluation, spec["seed"])[:spec["validation_count"]]
    for name, rows, count in [("pilot", pilot, spec["pilot_count"]),
                              ("development", development, spec["development_count"]),
                              ("validation", validation, spec["validation_count"])]:
        if len(rows) != count:
            raise ValueError(f"Insufficient unique images for {name}")
    return {"pilot": pilot, "development": development, "validation": validation}


def input_signature(spec):
    rows = read_manifest(spec["manifest"], "vqa", unique_images=True)
    snapshots = {}
    for condition in spec["conditions"]:
        if condition.get("snapshot"):
            root = Path(condition["snapshot"])
            snapshots[condition["name"]] = {
                name: file_hash(root / name) for name in ["model.json", "model.safetensors"]}
    return digest({"spec": spec, "source": source_fingerprint(),
                   "manifest": file_hash(spec["manifest"]), "snapshots": snapshots,
                   "images": [(row["image_id"], file_hash(row["image"])) for row in rows]})


def prepare(spec, root, resume):
    signature = input_signature(spec)
    pointer = root / "study.json"
    if pointer.exists():
        if not resume:
            raise FileExistsError("Study exists; use --resume")
        study = read_json(pointer)
        if study["signature"] != signature:
            raise ValueError("Study source/protocol/data/snapshot fingerprint changed")
        return study
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Study output is nonempty")
    selection = select_samples(spec)
    write_json(root / "private" / "selection.json", selection)
    study = {"schema_version": SCHEMA, "uses_private_reference": True,
             "signature": signature, "source_sha256": source_fingerprint(), "spec": spec,
             "selection_sha256": file_hash(root / "private" / "selection.json")}
    write_json(pointer, study)
    return study


def check_study(spec, root):
    study = read_json(root / "study.json")
    if study["schema_version"] != SCHEMA or study["signature"] != input_signature(spec):
        raise ValueError("Study source/protocol/data/snapshot fingerprint changed")
    path = root / "private" / "selection.json"
    if file_hash(path) != study["selection_sha256"]:
        raise ValueError("Frozen selection changed")
    return study, read_json(path)


def gpu_inventory():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total",
         "--format=csv,noheader,nounits"], text=True)
    result = {}
    for line in raw.strip().splitlines():
        index, uuid, free, total = [value.strip() for value in line.split(",")]
        result[index] = {"uuid": uuid, "free_mib": int(free), "total_mib": int(total)}
    return result


def worker_binding(device, tiny=False, inventory=None):
    if device == "cpu":
        if not tiny:
            raise ValueError("CPU execution is restricted to the tiny test model")
        return None
    if device != "cuda:0":
        raise ValueError("Worker must use its single visible device cuda:0")
    if os.environ.get("GIAVLM_TEXT_UNKNOWN_GPU") != "4":
        raise ValueError("Worker must declare physical GPU 4")
    devices = inventory if inventory is not None else gpu_inventory()
    uuid = devices["4"]["uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ValueError("Worker must expose exactly the UUID of physical GPU 4")
    return {"physical_gpu": 4, "uuid": uuid, "pid": os.getpid()}


def make_adapter(spec, condition, device):
    from core.fl import restore_model
    from core.vlm_wrapper import build_model
    cfg = configuration(spec, condition, device)
    if condition.get("snapshot"):
        adapter = restore_model(condition["snapshot"], device)
        if asdict(adapter.spec) != asdict(cfg.model):
            raise ValueError("Snapshot model configuration differs from study")
        old = adapter.training_spec
        if (old.fine_tuning_strategy, old.server_round, old.lora_rank, old.lora_alpha) != (
                condition["strategy"], condition["round"], cfg.training.lora_rank,
                cfg.training.lora_alpha):
            raise ValueError("Snapshot strategy, round or LoRA parameterization differs")
        adapter.set_training_spec(cfg.training)
    else:
        adapter = build_model(cfg.model, cfg.training)
    adapter.eval()
    return adapter


def image_view(adapter, row):
    with Image.open(row["image"]) as source:
        return adapter.prepare_image(ImageOps.exif_transpose(source).convert("RGB")).float()[None]


def batch_for(adapter, images, question, target):
    encoded = adapter.batch(images, [question], [target])
    return Batch(images, encoded.questions, encoded.targets)


def observation_for(adapter, batch, update, knowledge):
    training = replace(adapter.training_spec, knowledge=knowledge)
    questions, targets = adapter.decode(batch.questions), adapter.decode(batch.targets)
    observation = Observation(
        model=replace(adapter.spec), training=training,
        tensors={name: value.detach().clone() for name, value in update.items()},
        model_fingerprint=adapter.fingerprint(),
        public_questions=list(questions) if training.question_public else [],
        public_targets=list(targets) if training.target_public else [],
        public_question_ids=batch.questions.detach().cpu().tolist() if training.question_public else [],
        public_target_ids=batch.targets.detach().cpu().tolist() if training.target_public else [])
    observation.validate()
    return observation


def _random_phrase(words, count, rng, question=False):
    value = " ".join(rng.choice(words) for _ in range(count))
    return value.capitalize() + ("?" if question else "")


def text_guesses(spec, observation, knowledge):
    options = spec["text_guesses"]
    count = options["per_family"]
    if knowledge == "text_known":
        return [{"id": "oracle-00", "family": "oracle",
                 "question": observation.public_questions[0],
                 "target": observation.public_targets[0]}]
    guesses = []
    for index, pair in enumerate(options["template_pairs"][:count]):
        question = (observation.public_questions[0] if "question" in KNOWLEDGE_FIELDS[knowledge]
                    else pair["question"])
        target = (observation.public_targets[0] if "target" in KNOWLEDGE_FIELDS[knowledge]
                  else pair["target"])
        guesses.append({"id": f"template-{index:02d}", "family": "template",
                        "question": question, "target": target})
    rng = random.Random(int(digest([spec["seed"], knowledge, "random-words"]), 16))
    for index in range(count):
        question = (observation.public_questions[0] if "question" in KNOWLEDGE_FIELDS[knowledge]
                    else _random_phrase(options["question_words"],
                                        options["question_word_count"], rng, True))
        target = (observation.public_targets[0] if "target" in KNOWLEDGE_FIELDS[knowledge]
                  else _random_phrase(options["target_words"],
                                      options["target_word_count"], rng))
        guesses.append({"id": f"random-{index:02d}", "family": "random_words",
                        "question": question, "target": target})
    return guesses


def candidate_bank(spec, truth):
    available = {identifier: (metadata, image) for identifier, metadata, image in candidates(truth)}
    missing = set(spec["candidate_ids"]) - set(available)
    if missing:
        raise ValueError(f"Unknown candidate IDs: {sorted(missing)}")
    return [(identifier, available[identifier][0], available[identifier][1])
            for identifier in spec["candidate_ids"]]


def checked_record(path, signature, resume):
    if not path.exists():
        return False
    if not resume:
        raise FileExistsError("Diagnostic record exists; use --resume")
    saved = read_json(path)
    if saved.get("signature") != signature or saved.get("status") != "completed":
        raise ValueError("Record fingerprint or completion state differs")
    return True


def score_condition(args, spec, study, selection, condition):
    device = args.device or "cuda:0"
    tiny = configuration(spec, condition, device).model.family == "tiny"
    binding = worker_binding(device, tiny)
    if binding:
        inventory = gpu_inventory()["4"]
        if inventory["free_mib"] < spec["resources"]["min_free_mib"]:
            raise RuntimeError("GPU 4 does not meet the declared free-memory requirement")
    adapter = make_adapter(spec, condition, device)
    started = time.monotonic()
    if binding:
        torch.cuda.reset_peak_memory_stats()
    output = Path(args.output) / "records" / args.cohort / condition["name"]
    new_records = 0
    for image_index, row in enumerate(selection[args.cohort]):
        image_name = f"image-{image_index:03d}"
        directory = output / image_name
        truth = image_view(adapter, row)
        true_batch = batch_for(adapter, truth, row["question"], row["target"])
        target_update = replay(adapter, truth, true_batch)
        bank = candidate_bank(spec, truth)
        image_metrics = {identifier: image_scores(truth, image)
                         for identifier, _, image in bank}
        index = {"status": "completed", "signature": study["signature"],
                 "uses_private_reference": True,
                 "candidate_ids": [identifier for identifier, _, _ in bank],
                 "knowledge": {}}
        for knowledge in spec["knowledge_conditions"]:
            observation = observation_for(adapter, true_batch, target_update, knowledge)
            guesses = text_guesses(spec, observation, knowledge)
            index["knowledge"][knowledge] = [
                {"id": guess["id"], "family": guess["family"]} for guess in guesses]
            for guess in guesses:
                for candidate_id, metadata, image in bank:
                    path = directory / "scores" / knowledge / guess["id"] / f"{candidate_id}.json"
                    signature = digest([study["signature"], condition["name"], image_name,
                                        knowledge, guess["id"], candidate_id])
                    if checked_record(path, signature, args.resume):
                        continue
                    batch = batch_for(adapter, image, guess["question"], guess["target"])
                    predicted = replay(adapter, image, batch)
                    scores = update_scores(predicted, target_update)
                    write_json(path, {"status": "completed", "signature": signature,
                                      "uses_private_reference": True, "knowledge": knowledge,
                                      "guess_id": guess["id"], "guess_family": guess["family"],
                                      "candidate_id": candidate_id, "family": metadata["family"],
                                      "losses": scores["losses"],
                                      "image": image_metrics[candidate_id]})
                    new_records += 1
                    del predicted
        write_json(directory / "candidate_index.json", index)
        del target_update, true_batch, truth
        print(json.dumps({"stage": "score", "condition": condition["name"],
                          "image": image_name, "status": "completed"}), flush=True)
    marker = {"status": "completed", "signature": study["signature"],
              "uses_private_reference": True, "condition": condition["name"],
              "images": len(selection[args.cohort]), "new_records": new_records,
              "seconds": time.monotonic() - started, "binding": binding,
              "peak_allocated_mib": (torch.cuda.max_memory_allocated() / 2**20 if binding else 0),
              "peak_reserved_mib": (torch.cuda.max_memory_reserved() / 2**20 if binding else 0)}
    write_json(output / "score-complete.json", marker)


def _families(guess_metadata):
    families = sorted({guess["family"] for guess in guess_metadata})
    return {"all": [guess["id"] for guess in guess_metadata],
            **{family: [guess["id"] for guess in guess_metadata if guess["family"] == family]
               for family in families}}


def _image_summary(directory, index, knowledge):
    candidate_ids = index["candidate_ids"]
    guess_metadata = index["knowledge"][knowledge]
    by_guess = {}
    for guess in guess_metadata:
        rows = [read_json(directory / "scores" / knowledge / guess["id"] / f"{candidate}.json")
                for candidate in candidate_ids]
        by_guess[guess["id"]] = rows
    result = {}
    for family, guess_ids in _families(guess_metadata).items():
        conditional = []
        for guess_id in guess_ids:
            rows = [row for row in by_guess[guess_id] if row["family"] != "truth"]
            conditional.append(candidate_statistics(rows, "cosine"))
        candidates_without_truth = [candidate for candidate in candidate_ids
                                    if candidate != "truth"]
        values = {candidate: {name: [] for name in AGGREGATES} for candidate in candidate_ids}
        percentile_ranks = {}
        for guess_id in guess_ids:
            losses = np.asarray([next(row for row in by_guess[guess_id]
                                      if row["candidate_id"] == candidate)["losses"]["cosine"]
                                 for candidate in candidate_ids])
            ranks = (rankdata(losses, method="average") - 1) / max(1, len(losses) - 1)
            percentile_ranks[guess_id] = ranks
            for candidate, loss, rank in zip(candidate_ids, losses, ranks):
                values[candidate]["mean_rank"].append(float(rank))
                for name in ["mean_loss", "median_loss", "min_loss"]:
                    values[candidate][name].append(float(loss))
        rows = []
        for candidate in candidates_without_truth:
            reference = next(row for row in by_guess[guess_ids[0]]
                             if row["candidate_id"] == candidate)
            losses = {"mean_rank": float(np.mean(values[candidate]["mean_rank"])),
                      "mean_loss": float(np.mean(values[candidate]["mean_loss"])),
                      "median_loss": float(np.median(values[candidate]["median_loss"])),
                      "min_loss": float(np.min(values[candidate]["min_loss"]))}
            rows.append({"losses": losses, "image": reference["image"]})
        truth_index = candidate_ids.index("truth")
        truth_rank = float(np.mean([percentile_ranks[guess][truth_index] for guess in guess_ids]))
        stability = []
        for left_index, left in enumerate(guess_ids):
            for right in guess_ids[left_index + 1:]:
                stability.append(float(spearmanr(percentile_ranks[left],
                                                percentile_ranks[right]).statistic))
        result[family] = {
            "guess_count": len(guess_ids),
            "conditional": {
                "spearman_mse": float(np.mean([item["spearman"]["mse"]["value"]
                                                for item in conditional])),
                "near_win": float(np.mean([item["near_win"]["value"]
                                            for item in conditional])),
            },
            "aggregate": {name: candidate_statistics(rows, name) for name in AGGREGATES},
            "truth_mean_percentile_rank": truth_rank,
            "rank_stability": float(np.mean(stability)) if stability else 1.0,
        }
    return result


def report(args, spec, study, condition):
    root = Path(args.output)
    directory = root / "records" / args.cohort / condition["name"]
    marker = read_json(directory / "score-complete.json")
    if marker["status"] != "completed" or marker["signature"] != study["signature"]:
        raise ValueError("Report requires a completed matching score stage")
    images = {}
    for index_path in sorted(directory.glob("image-*/candidate_index.json")):
        index = read_json(index_path)
        if index["signature"] != study["signature"]:
            raise ValueError("Candidate index fingerprint differs")
        images[index_path.parent.name] = {
            knowledge: _image_summary(index_path.parent, index, knowledge)
            for knowledge in spec["knowledge_conditions"]}
    if len(images) != marker["images"]:
        raise ValueError("Report is missing image records")
    summary = {}
    for knowledge in spec["knowledge_conditions"]:
        families = sorted({family for image in images.values()
                           for family in image[knowledge]})
        summary[knowledge] = {}
        for family in families:
            records = [image[knowledge][family] for image in images.values()
                       if family in image[knowledge]]
            summary[knowledge][family] = {
                "conditional_spearman_mse": bootstrap(
                    [record["conditional"]["spearman_mse"] for record in records]),
                "conditional_near_win": bootstrap(
                    [record["conditional"]["near_win"] for record in records]),
                "truth_mean_percentile_rank": bootstrap(
                    [record["truth_mean_percentile_rank"] for record in records]),
                "rank_stability": bootstrap([record["rank_stability"] for record in records]),
                "aggregate": {name: {
                    "spearman_mse": bootstrap([
                        record["aggregate"][name]["spearman"]["mse"]["value"]
                        for record in records]),
                    "near_win": bootstrap([
                        record["aggregate"][name]["near_win"]["value"]
                        for record in records]),
                    "bottom_1pct_far_fraction": bootstrap([
                        record["aggregate"][name]["bottom_fraction"]["0.01"]["far_fraction"]
                        for record in records]),
                } for name in AGGREGATES},
            }
    result = {"schema_version": 1, "uses_private_reference": True,
              "note": "Finite public-text-guess diagnostic; not an attack benchmark.",
              "signature": study["signature"], "cohort": args.cohort,
              "model_condition": condition["name"], "images": images, "summary": summary,
              "resources": marker}
    output = root / "reports" / args.cohort / condition["name"]
    write_json(output / "report.json", result)
    _write_markdown_and_plot(output, result)
    return result


def _write_markdown_and_plot(output, result):
    lines = ["# Image discriminability under unknown text", "",
             "Private-reference diagnostic; text guesses are public and fixed across image candidates.", "",
             f"Cohort: {result['cohort']}; model condition: {result['model_condition']}.", "",
             "| Knowledge | Guess family | Conditional rho | Aggregate-rank rho | "
             "Aggregate near-win | Truth percentile | Rank stability |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for knowledge, families in result["summary"].items():
        for family, record in families.items():
            aggregate = record["aggregate"]["mean_rank"]
            lines.append(
                f"| {knowledge} | {family} | {record['conditional_spearman_mse']['mean']:.4f} | "
                f"{aggregate['spearman_mse']['mean']:.4f} | {aggregate['near_win']['mean']:.4f} | "
                f"{record['truth_mean_percentile_rank']['mean']:.4f} | "
                f"{record['rank_stability']['mean']:.4f} |")
    lines += ["", "Lower truth percentile is better; near-win above 0.5 indicates useful image ordering.",
              "Bootstrap units are images. Full per-image and aggregation results are in report.json.", ""]
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels, rho, near = [], [], []
    for knowledge, families in result["summary"].items():
        preferred = "all" if "all" in families else next(iter(families))
        record = families[preferred]["aggregate"]["mean_rank"]
        labels.append(knowledge)
        rho.append(record["spearman_mse"]["mean"])
        near.append(record["near_win"]["mean"])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, rho, color="#1976D2")
    axes[0].axhline(0, color="gray", linestyle="--")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Spearman(rank score, image MSE)",
                title="Aggregated image ranking")
    axes[1].bar(x, near, color="#278A5B")
    axes[1].axhline(.5, color="gray", linestyle="--")
    axes[1].set(xticks=x, xticklabels=labels, ylim=(0, 1), ylabel="Near-win probability",
                title="Lower-error candidate wins")
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"unknown_text_ranking.{suffix}", dpi=180)
    plt.close(fig)


def freeze(root, study, spec):
    hashes = {}
    for condition in spec["conditions"]:
        marker = root / "records" / "development" / condition["name"] / "score-complete.json"
        report_path = root / "reports" / "development" / condition["name"] / "report.json"
        if not marker.exists() or not report_path.exists():
            raise FileNotFoundError("Freeze requires every development score and report")
        hashes[condition["name"]] = {"marker": file_hash(marker), "report": file_hash(report_path)}
    write_json(root / "freeze.json", {"status": "completed", "signature": study["signature"],
                                      "development": hashes})


def execute(args):
    root = Path(args.output)
    try:
        spec = load_spec(args.spec)
        if args.stage == "prepare":
            study = prepare(spec, root, args.resume)
            print(json.dumps({"stage": "prepare", "status": "completed",
                              "signature": study["signature"]}), flush=True)
            return
        study, selection = check_study(spec, root)
        if args.cohort == "validation":
            frozen = read_json(root / "freeze.json")
            if frozen["signature"] != study["signature"] or frozen["status"] != "completed":
                raise ValueError("Validation requires the matching frozen development analysis")
        if args.stage == "freeze":
            freeze(root, study, spec)
            print(json.dumps({"stage": "freeze", "status": "completed"}), flush=True)
            return
        conditions = [condition for condition in spec["conditions"]
                      if not args.model_condition or condition["name"] == args.model_condition]
        if not conditions:
            raise ValueError("Unknown model condition")
        for condition in conditions:
            if args.stage == "score":
                score_condition(args, spec, study, selection, condition)
            elif args.stage == "report":
                report(args, spec, study, condition)
                print(json.dumps({"stage": "report", "condition": condition["name"],
                                  "status": "completed"}), flush=True)
            else:
                raise ValueError("Unsupported stage")
    except BaseException as error:
        write_json(root / "failures" / f"{int(time.time_ns())}.json",
                   {"status": "failed", "error_type": type(error).__name__,
                    "stage": args.stage, "cohort": args.cohort})
        raise
