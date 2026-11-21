"""Matched fixed-QA diagnostic with frozen public templates and paired natural-QA scores."""

import argparse
from dataclasses import replace
import gc
from pathlib import Path
import os
import random
import time
import traceback
import uuid

from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.deep_property_probe import distance_to_gradient
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.multilayer_property import LAYERS, fuse
from evaluation.multilayer_property_gpu import CLASSES


QUESTION = "Describe the image."
ANSWER = "The image is shown."
METHODS = ("single_selected", "single_32", "majority", "mean_raw", "mean_standardized")


def substitute_qa(row, question=QUESTION, answer=ANSWER):
    return {**row, "question": question, "target": answer}


def align_runs(predictions, mapping, runs):
    by_run = {mapping[row["id"]]: row for row in predictions}
    if len(by_run) != len(predictions) or set(by_run) != set(runs):
        raise ValueError("A one-to-one matched image mapping is required")
    return [by_run[run] for run in runs]


def score_upload(tensors, prototypes, rules):
    """Only uploaded tensors and public prototypes enter the actual classifier."""
    result = {}
    for name in CLASSES:
        individual = {}
        for layer in LAYERS:
            key = f"backend.language_model.model.layers.{layer - 1}.self_attn.q_proj.lora_A.default.weight"
            individual[f"gradient_{layer}"] = [distance_to_gradient(prototypes[name][f"visual_{layer}"], tensors[key])]
        result[name] = {method: values[0] for method, values in fuse(individual, rules[name]).items()}
        result[name].update({method: values[0] for method, values in individual.items()})
    return result


def prepare(root):
    from core.config import TrainingSpec
    path = root / "fixed-qa-plan-template.json"
    if path.exists():
        raise FileExistsError("Fixed-QA plan is already frozen")
    parent = read_json(root / "probe-plan.json")
    observation = read_json(Path(read_json(root / "fresh/upload-list.json")[0]["observation"]) / "observation.json")
    training = TrainingSpec(**observation["training"])
    if training.algorithm != "fedsgd" or training.sample_count != 1 or training.knowledge != "private":
        raise ValueError("Expected the original private single-sample FedSGD condition")
    public_files = [root / "head-rule.json", root / "public/abdomen-rule.json",
                    Path(parent["head_prototypes"]), root / "public/abdomen-prototypes.safetensors",
                    Path(parent["model"]) / "model.json", Path(parent["model"]) / "model.safetensors",
                    root / "fresh/predictions.json", root / "fresh/predictions-complete.json",
                    root / "fresh/report.json", root / "fresh/complete.json", root / "cohort-audit.json"]
    write_json(path, {"question": QUESTION, "answer": ANSWER, "model": parent["model"],
               "model_fingerprint": observation["model_fingerprint"], "natural_training": observation["training"],
               "head_prototypes": parent["head_prototypes"], "n": 20, "gpu_uuid": None,
               "primary": "mean_standardized", "single_selected": 24, "secondary": 32,
               "knowledge": "text_known fixed-QA diagnostic; scorer uses uploaded tensors only",
               "comparison": "Same 20 images, model, layers, public prototypes, scales, threshold and single-step update; only client QA changes",
               "interpretation": "QA content, token lengths and supervised task change together. This is not unknown-natural-text benchmark evidence.",
               "hashes": {str(p.resolve()): file_hash(p) for p in public_files}})


def capture_fixed(adapter, root, plan):
    from core.config import TrainingSpec
    from core.data import load_batch
    from core.fl import capture, save_observation
    from evaluation.image_property_data import pixel_hash
    out = root / "fixed-qa"
    audit = read_json(root / "cohort-audit.json")
    if file_hash(root / "private/targets.json") != audit["targets_sha256"]:
        raise ValueError("The reviewed image cohort changed")
    training = replace(TrainingSpec(**plan["natural_training"]), knowledge="text_known")
    adapter.set_training_spec(training)
    targets = read_json(root / "private/targets.json")
    random.SystemRandom().shuffle(targets)
    uploads, mapping = [], {}
    for index, item in enumerate(targets):
        with Image.open(item["row"]["image"]) as image:
            if pixel_hash(image) != item["pixel_sha256"]:
                raise ValueError("Target pixels changed between natural and fixed QA")
        batch = load_batch([substitute_qa(item["row"], plan["question"], plan["answer"])], adapter)
        observation = capture(adapter, batch, [plan["question"]], [plan["answer"]])
        identifier = uuid.uuid4().hex
        destination = out / "observations" / identifier
        save_observation(destination, observation)
        uploads.append({"id": identifier, "observation": str(destination)})
        mapping[identifier] = item["run"]
        write_json(out / "progress.json", {"stage": "capture", "completed": index + 1, "total": len(targets)})
        if index == 0:
            write_json(out / "first-artifact.json", {"update_sha256": file_hash(destination / "update.safetensors"),
                       "knowledge": "text_known", "fixed_public_qa": True})
        print(f"Fixed-QA capture {index + 1}/{len(targets)}", flush=True)
        del batch, observation
        gc.collect()
    write_json(out / "private/upload-map.json", mapping)
    write_json(out / "upload-list.json", uploads)
    write_json(out / "capture-complete.json", {"n": len(uploads), "hashes": {
        str(p): file_hash(p) for p in (out / "observations").rglob("*") if p.is_file()}})


def predict(root, plan):
    from core.fl import load_observation
    out = root / "fixed-qa"
    rules = {"head": read_json(root / "head-rule.json"), "abdomen": read_json(root / "public/abdomen-rule.json")}
    prototypes = {"head": read_tensors(plan["head_prototypes"]),
                  "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    predictions = []
    for item in read_json(out / "upload-list.json"):
        observation = load_observation(item["observation"], "cpu")
        if (observation.model_fingerprint != plan["model_fingerprint"] or observation.sample_count != 1
                or observation.training.knowledge != "text_known" or observation.public_images is not None
                or observation.public_questions != [plan["question"]] or observation.public_targets != [plan["answer"]]):
            raise ValueError("Fixed-task Observation does not match the frozen condition")
        # The declared public QA is checked above but never passed into scoring.
        scores = score_upload(observation.tensors, prototypes, rules)
        predictions.append({"id": item["id"], "scores": scores})
        del observation
    write_json(out / "predictions.json", predictions)
    write_json(out / "predictions-complete.json", {"sha256": file_hash(out / "predictions.json"),
               "labels_read": False, "plan_sha256": file_hash(out / "plan.json")})


def evaluate(root):
    out = root / "fixed-qa"
    if file_hash(out / "predictions.json") != read_json(out / "predictions-complete.json")["sha256"]:
        raise ValueError("Fixed-QA prediction commit changed")
    if file_hash(root / "fresh/predictions.json") != read_json(root / "fresh/predictions-complete.json")["sha256"]:
        raise ValueError("Natural-QA prediction commit changed")
    # All fixed predictions exist before target labels and the matched mappings join.
    truth = sorted(read_json(root / "private/targets.json"), key=lambda row: row["run"])
    runs = [row["run"] for row in truth]
    natural = align_runs(read_json(root / "fresh/predictions.json"), read_json(root / "private/upload-map.json"), runs)
    fixed = align_runs(read_json(out / "predictions.json"), read_json(out / "private/upload-map.json"), runs)
    results = {}
    for prop, classes in CLASSES.items():
        groups = {"all": runs, "original_no_literal_terms": [i for i, row in enumerate(truth) if not row["literal_terms"][prop]],
                  **{m: [i for i, row in enumerate(truth) if row["modality"] == m] for m in ["CT", "MRI", "X-Ray"]}}
        results[prop] = {}
        for name, ids in groups.items():
            labels = [classes[0] if truth[i]["region"] == prop else classes[1] for i in ids]
            scores = {condition: {method: [rows[i]["scores"][prop][method] for i in ids]
                      for method in METHODS} for condition, rows in [("natural", natural), ("fixed", fixed)]}
            metrics = {condition: {method: {**classification_stats(labels, value, classes),
                       **(paired_bootstrap(labels, value, classes) or {})} for method, value in by_method.items()}
                       for condition, by_method in scores.items()}
            gains = {}
            if ids:
                for method in METHODS:
                    a, b = metrics["fixed"][method], metrics["natural"][method]
                    gains[method] = {"accuracy_gain": a["accuracy"] - b["accuracy"],
                        "balanced_accuracy_gain": a["balanced_accuracy"] - b["balanced_accuracy"]
                        if a["balanced_accuracy"] is not None else None,
                        **paired_bootstrap(labels, scores["fixed"][method], classes, scores["natural"][method])}
            results[prop][name] = {"metrics": metrics, "paired_fixed_minus_natural": gains}
    report = {"n": 20, "properties": results, "primary": "mean_standardized", "layers": [8, 16, 24, 32],
              "fixed_question": QUESTION, "fixed_answer": ANSWER, "cohort": read_json(root / "cohort-audit.json"),
              "scope": "Matched diagnostic on the same 20 images; fixed task is known, not a new unknown-QA benchmark",
              "predictions_sha256": file_hash(out / "predictions.json"),
              "natural_predictions_sha256": file_hash(root / "fresh/predictions.json"),
              "source_sha256": file_hash(Path(__file__))}
    write_json(out / "report.json", report)
    write_json("docs/experiments/data/visual-concept-probe/fixed-qa-region-aggregate.json", report)
    print({prop: {method: group["all"]["metrics"]["fixed"][method]["correct"] for method in METHODS}
           for prop, group in results.items()}, flush=True)


def run(root):
    from core.fl import restore_model
    out = root / "fixed-qa"
    if (out / "started.json").exists():
        raise FileExistsError("Fixed-QA stage already attempted; no automatic retry")
    write_json(out / "started.json", {"pid": os.getpid(), "host": os.uname().nodename})
    start = time.monotonic()
    try:
        plan = read_json(out / "plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen paired input changed")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly the selected physical GPU")
        if torch.cuda.mem_get_info()[0] < 60 * 2 ** 30:
            raise RuntimeError("At least 60 GiB free required; no settings changes")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        if adapter.fingerprint() != plan["model_fingerprint"]:
            raise ValueError("Fixed-QA victim differs from natural-QA victim")
        capture_fixed(adapter, root, plan)
        if adapter.fingerprint() != plan["model_fingerprint"]:
            raise ValueError("Victim changed during fixed-QA capture")
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        predict(root, plan)
        evaluate(root)
        write_json(out / "complete.json", {"status": "complete", "seconds": time.monotonic() - start,
                   "peak_memory_mib": peak, "report_sha256": file_hash(out / "report.json")})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.root.resolve())


if __name__ == "__main__":
    main()
