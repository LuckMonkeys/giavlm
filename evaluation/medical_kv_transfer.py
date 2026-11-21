"""Frozen public QKV rule: anonymous natural medical uploads, then evaluator labels."""

import argparse
import gc
import os
from pathlib import Path
import random
import time
import traceback
import uuid

import torch
from PIL import Image

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.deep_property_probe import distance_to_gradient
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.multilayer_property_gpu import CLASSES


METHODS = ("L24_q", "L24_k", "L24_v", "L32_q", "L32_k", "L32_v")


def score(tensors, prototypes):
    """The actual classifier receives no QA, private image, labels or token counts."""
    result = {}
    for prop in CLASSES:
        result[prop] = {}
        for method in METHODS:
            layer, letter = method.split("_")
            key = f"backend.language_model.model.layers.{int(layer[1:])-1}.self_attn.{letter}_proj.lora_A.default.weight"
            result[prop][method] = distance_to_gradient(
                prototypes[prop][f"visual_{layer[1:]}"] , tensors[key])
    return result


def prepare(root):
    out = root / "medical-kv"
    destination = root / "medical-kv-plan-template.json"
    if destination.exists():
        raise FileExistsError("Medical transfer plan already frozen")
    parent = read_json(root / "probe-plan.json")
    cohort = read_json(out / "cohort-audit.json")
    uploads = read_json(root / "fresh/upload-list.json")
    metadata = read_json(Path(uploads[0]["observation"]) / "observation.json")
    files = [Path(parent["model"]) / name for name in ("model.json", "model.safetensors")]
    files += [Path(parent["head_prototypes"]), root / "public/abdomen-prototypes.safetensors",
              out / "public-rule.json", out / "cohort-audit.json", root / "fresh/upload-list.json",
              root / "fresh/predictions.json", root / "fresh/predictions-complete.json"]
    files += [Path(row["observation"]) / name for row in uploads for name in ("observation.json", "update.safetensors")]
    write_json(destination, {"model": parent["model"], "gpu_uuid": None,
        "head_prototypes": parent["head_prototypes"], "model_fingerprint": metadata["model_fingerprint"],
        "training": metadata["training"], "primary": read_json(out / "public-rule.json")["primary"],
        "methods": list(METHODS), "old_images": len(uploads), "new_images": cohort["n"],
        "new_targets_sha256": cohort["targets_sha256"], "qa": "Original natural VQA only, unmodified; no reconstruction or fixed QA",
        "knowledge": "private; no target token lengths disclosed", "selection": "Public-only frozen module rule before medical scoring",
        "scoring": "Original raw two-prototype subspace distances, argmin; no threshold or target normalization fit",
        "scope": "Historical 20-image diagnosis plus new 20-image ID/pixel-disjoint medical transfer confirmation; modality/region confounding and patient dependence remain",
        "source_sha256": file_hash(Path(__file__)), "hashes": {str(path.resolve()): file_hash(path) for path in files}})


def capture_new(adapter, root, plan):
    from core.config import TrainingSpec
    from core.data import load_batch
    from core.fl import capture, save_observation
    from evaluation.image_property_data import pixel_hash
    out = root / "medical-kv"
    path = out / "private/targets.json"
    if file_hash(path) != plan["new_targets_sha256"]:
        raise ValueError("Reviewed target cohort changed")
    adapter.set_training_spec(TrainingSpec(**plan["training"]))
    if adapter.training_spec.knowledge != "private" or adapter.training_spec.token_lengths_known:
        raise ValueError("Capture must preserve the original private natural-QA protocol")
    targets = read_json(path)
    random.SystemRandom().shuffle(targets)
    uploads, mapping = [], {}
    for index, item in enumerate(targets):
        with Image.open(item["row"]["image"]) as image:
            if pixel_hash(image) != item["pixel_sha256"]:
                raise ValueError("Reviewed target pixels changed")
        batch = load_batch([item["row"]], adapter)
        observation = capture(adapter, batch, [], [])
        identifier = uuid.uuid4().hex
        folder = out / "new/observations" / identifier
        save_observation(folder, observation)
        uploads.append({"id": identifier, "observation": str(folder)})
        mapping[identifier] = item["run"]
        write_json(out / "progress.json", {"stage": "capture_new", "completed": index + 1, "total": len(targets)})
        if index == 0:
            write_json(out / "first-artifact.json", {"update_sha256": file_hash(folder / "update.safetensors"),
                       "knowledge": "private", "token_lengths_known": False})
        print(f"Medical capture {index+1}/{len(targets)}", flush=True)
        del batch, observation
        gc.collect()
    write_json(out / "new/upload-list.json", uploads)
    write_json(out / "private/upload-map.json", mapping)
    write_json(out / "new/capture-complete.json", {"images": len(uploads), "hashes": {
               str(path): file_hash(path) for path in (out / "new/observations").rglob("*") if path.is_file()}})


def predict(uploads, prototypes, fingerprint):
    from core.fl import load_observation
    results = []
    for item in uploads:
        observation = load_observation(item["observation"], "cpu")
        if (observation.model_fingerprint != fingerprint or observation.training.knowledge != "private"
                or observation.training.token_lengths_known or observation.sample_count != 1
                or observation.public_images is not None or observation.public_questions or observation.public_targets
                or observation.public_question_ids or observation.public_target_ids):
            raise ValueError("Only matching anonymous private Observations are allowed")
        results.append({"id": item["id"], "scores": score(observation.tensors, prototypes)})
        del observation
    return results


def commit_predictions(root, plan):
    out = root / "medical-kv"
    prototypes = {"head": read_tensors(plan["head_prototypes"]),
                  "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    result = {"historical": predict(read_json(root / "fresh/upload-list.json"), prototypes, plan["model_fingerprint"]),
              "new": predict(read_json(out / "new/upload-list.json"), prototypes, plan["model_fingerprint"])}
    baseline = {row["id"]: row for row in read_json(root / "fresh/predictions.json")}
    replay_error = 0.
    for row in result["historical"]:
        for prop in CLASSES:
            for layer in (24, 32):
                error = max(abs(a - b) for a, b in zip(row["scores"][prop][f"L{layer}_q"],
                            baseline[row["id"]]["scores"][prop][f"gradient_{layer}"], strict=True))
                replay_error = max(replay_error, error)
                if error > 1e-12:
                    raise ValueError("Original historical Q distances were not reproduced")
    write_json(out / "predictions.json", result)
    write_json(out / "predictions-complete.json", {"sha256": file_hash(out / "predictions.json"),
               "primary": plan["primary"], "private_labels_read_for_evaluation": False,
               "public_rule_sha256": file_hash(out / "public-rule.json"), "historical_q_replay_max_error": replay_error})


def answer_type(row):
    return "yes_no" if row["row"]["target"].strip().lower().rstrip(".!? ") in ("yes", "no") else "other"


def evaluate(root, plan):
    out = root / "medical-kv"
    if file_hash(out / "predictions.json") != read_json(out / "predictions-complete.json")["sha256"]:
        raise ValueError("Medical prediction commit changed")
    predictions = read_json(out / "predictions.json")
    output, composition = {}, {}
    for cohort, target_path, mapping_path in [
        ("historical", root / "private/targets.json", root / "private/upload-map.json"),
        ("new", out / "private/targets.json", out / "private/upload-map.json")]:
        # Join evaluation-only labels, QA categories and image mapping after all 40 blind predictions are committed.
        targets = {row["run"]: row for row in read_json(target_path)}
        mapping = read_json(mapping_path)
        truth = [targets[mapping[row["id"]]] for row in predictions[cohort]]
        from collections import Counter
        composition[cohort] = {"n": len(truth), "region": dict(Counter(row["region"] for row in truth)),
            "modality": dict(Counter(row["modality"] for row in truth)),
            "answer_type": dict(Counter(answer_type(row) for row in truth))}
        output[cohort] = {}
        for prop, classes in CLASSES.items():
            groups = {"all": list(range(len(truth))),
                "no_literal_property_terms": [i for i, row in enumerate(truth) if not row["literal_terms"][prop]],
                **{name: [i for i, row in enumerate(truth) if row["modality"] == name] for name in ("CT", "MRI", "X-Ray")},
                **{name: [i for i, row in enumerate(truth) if answer_type(row) == name] for name in ("yes_no", "other")}}
            output[cohort][prop] = {}
            for group, ids in groups.items():
                labels = [classes[0] if truth[i]["region"] == prop else classes[1] for i in ids]
                scores = {method: [predictions[cohort][i]["scores"][prop][method] for i in ids] for method in METHODS}
                metrics = {method: {**classification_stats(labels, value, classes),
                    **(paired_bootstrap(labels, value, classes) or {})} for method, value in scores.items()}
                comparisons = {}
                if labels:
                    for method in METHODS:
                        layer, _ = method.split("_")
                        baseline = layer + "_q"
                        comparisons[method] = {"comparator": baseline,
                            "accuracy_gain": metrics[method]["accuracy"] - metrics[baseline]["accuracy"],
                            "balanced_accuracy_gain": metrics[method]["balanced_accuracy"] - metrics[baseline]["balanced_accuracy"]
                                if metrics[method]["balanced_accuracy"] is not None else None,
                            **paired_bootstrap(labels, scores[method], classes, scores[baseline])}
                output[cohort][prop][group] = {"metrics": metrics, "paired_vs_same_layer_q": comparisons}
    write_json(out / "report.json", {"primary": plan["primary"], "properties": output, "composition": composition,
               "scope": plan["scope"], "uses_target_text": False,
               "prediction_sha256": file_hash(out / "predictions.json"), "public_rule": read_json(out / "public-rule.json"),
               "new_cohort_audit": read_json(out / "cohort-audit.json")})
    print({c: {p: {m: v["correct"] for m, v in group["all"]["metrics"].items()} for p, group in props.items()}
           for c, props in output.items()}, flush=True)


def run(root):
    from core.fl import restore_model
    out = root / "medical-kv"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(out / "started.json", {"host": os.uname().nodename, "pid": os.getpid()})
    start = time.monotonic()
    try:
        plan = read_json(out / "plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen input changed")
        if file_hash(Path(__file__)) != plan["source_sha256"]:
            raise ValueError("Frozen transfer source changed")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly the selected GPU")
        if torch.cuda.mem_get_info()[0] < 60 * 2**30:
            raise RuntimeError("At least 60 GiB free required; no changes")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        if adapter.fingerprint() != plan["model_fingerprint"]:
            raise ValueError("Medical capture model differs from public probe model")
        capture_new(adapter, root, plan)
        if adapter.fingerprint() != plan["model_fingerprint"]:
            raise ValueError("Victim model changed")
        peak = torch.cuda.max_memory_allocated() / 1024**2
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        commit_predictions(root, plan)
        evaluate(root, plan)
        write_json(out / "complete.json", {"status": "complete", "seconds": time.monotonic() - start,
                   "peak_memory_mib": peak, "report_sha256": file_hash(out / "report.json"),
                   "runtime_source": str(Path(__file__).resolve())})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc(), "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    try:
        (prepare if args.prepare else run)(args.root.resolve())
    except BaseException:
        write_json(args.root / "medical-kv" / ("prepare-failed.json" if args.prepare else "run-failed.json"),
                   {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
