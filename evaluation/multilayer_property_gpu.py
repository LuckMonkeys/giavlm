"""Fresh-region experiment: public templates, private capture, blind prediction, evaluation."""

import argparse
from dataclasses import replace
import gc
import os
from pathlib import Path
import random
import time
import traceback
import uuid

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from evaluation.deep_property_probe import distance_to_gradient
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.multilayer_property import LAYERS, OLD, fuse, margins


MODEL = Path("outputs/diagnostics/dager_rank_gpu5_20261010_v2/models/rank-32/round-0010")
CLASSES = {"head": ["head", "non-head"], "abdomen": ["abdomen", "non-abdomen"]}


def public_negatives(rows):
    """Mix head and limb negatives; never define non-abdomen as head alone."""
    groups = {f"{label}-{index:02d}" for label in ["head", "non-head"]
              for index in [0, 1, 2, 3, 8, 9, 10, 11]}
    selected = [row for row in rows if row["group"] in groups]
    if len(selected) != 48:
        raise ValueError("Expected 48 reviewed public head and limb negatives")
    return selected


def abdomen_family(row):
    """The frozen generator repeats its eight prompt families with another seed."""
    family = int(row["id"].split("-")[1]) % 8
    return f"abdomen-prompt-family-{family:02d}", "fit" if family < 4 else "dev"


def prepare(root):
    if (root / "probe-plan-template.json").exists():
        raise FileExistsError("Public bank already frozen")
    generated = read_json(root / "generation/complete.json")["images"]
    review = read_json(root / "public/review.json")
    retained = set(review["retained_ids"])
    by_group = {}
    for item in generated:
        by_group.setdefault(item["group"], []).append(item)
    positives = []
    for group, cases in by_group.items():
        if any(case["id"] in retained for case in cases):
            if len(cases) != 3 or not all(case["id"] in retained for case in cases):
                raise ValueError("Review must retain or reject the complete three-style content group")
            for case in cases:
                group, split = abdomen_family(case)
                positives.append({**case, "original_group": case["group"], "original_split": case["split"],
                                  "group": group, "split": split})
    minimum = review.get("minimum_families_per_split", 6)
    for split in ["fit", "dev"]:
        if len({row["group"] for row in positives if row["split"] == split}) < minimum:
            raise ValueError("Insufficient reviewed prompt families per split; no regeneration")
    cases = []
    for row in positives:
        if file_hash(row["file"]) != row["sha256"]:
            raise ValueError("Generated image changed")
        cases.append({**row, "label": "abdomen", "reuse_index": None})
    original = read_json(OLD / "styles/public_manifest.json")
    for row in public_negatives(original):
        index = original.index(row)
        cases.append({**row, "label": "non-abdomen", "reuse_index": index,
                      "sha256": file_hash(row["file"])})
    fit_groups = {row["group"] for row in cases if row["split"] == "fit"}
    dev_groups = {row["group"] for row in cases if row["split"] == "dev"}
    if fit_groups & dev_groups:
        raise ValueError("Public fit/dev content overlap")
    write_json(root / "public/abdomen-bank.json", cases)
    public_files = [MODEL / "model.json", MODEL / "model.safetensors", root / "head-rule.json",
                    root / "public/abdomen-bank.json", root / "public/review.json",
                    OLD / "deep-probe/public-prototypes.safetensors"]
    for row in cases:
        public_files.append(Path(row["file"]))
        if row["reuse_index"] is not None:
            public_files.extend([OLD / "styles/features" / f"{row['reuse_index']:03d}.safetensors",
                                 OLD / "styles/predictions" / f"{row['reuse_index']:03d}.json"])
    write_json(root / "probe-plan-template.json", {
        "model": str(MODEL.resolve()), "gpu_uuid": None, "targets": 20,
        "training_question": "Describe the image.", "training_answer": "The image is shown.",
        "layers": list(LAYERS), "properties": CLASSES,
        "head_rule": str((root / "head-rule.json").resolve()),
        "head_prototypes": str((OLD / "deep-probe/public-prototypes.safetensors").resolve()),
        "public_bank": str((root / "public/abdomen-bank.json").resolve()),
        "hashes": {str(path.resolve()): file_hash(path) for path in public_files},
        "primary": "mean_standardized", "parameter_updates": "None; fixed native bf16 model",
        "scope": "Fresh image-ID isolated 20-image pilot; modality confounded and patient independence unknown"})
    print({"reviewed_positive_images": len(positives), "negative_images": 48}, flush=True)


def public_feature(adapter, row, q, a):
    if row["reuse_index"] is not None:
        index = row["reuse_index"]
        values = read_tensors(OLD / "styles/features" / f"{index:03d}.safetensors")
        meta = read_json(OLD / "styles/predictions" / f"{index:03d}.json")
        begin, end = meta["visual_token_start"], meta["visual_token_stop"]
        return {**{f"mean_{layer}": values[f"input_{layer}"][0, begin:end].double().mean(0)
                   for layer in LAYERS},
                **{f"gradient_{layer}": values[f"gradient_{layer}"] for layer in LAYERS}}
    from core.adapters.llava import llava_input_pieces
    from evaluation.visual_concept_probe import inject_logits
    from evaluation.visual_token_recovery import parameter_grads
    image = adapter.prepare_image(Image.open(row["file"]).convert("RGB")).unsqueeze(0)
    image = image.to(adapter.device, adapter.dtype)
    values, handles = {}, []
    with torch.no_grad():
        visual = adapter.visual_embeddings(image)[0].detach()
        lm = adapter.backend.language_model
        for layer in LAYERS:
            def hook(module, args, key=layer):
                values[key] = args[0].detach().cpu()
            handles.append(lm.model.layers[layer - 1].self_attn.q_proj.register_forward_pre_hook(hook))
        try:
            inject_logits(adapter, visual, q, a)
        finally:
            for handle in handles:
                handle.remove()
        pieces = llava_input_pieces(adapter, visual, adapter.embedding()(q), adapter.embedding()(a[:, :-1]))
        begin, end = pieces[0].shape[1], pieces[0].shape[1] + visual.shape[1]
    grads = parameter_grads(adapter, visual, q, a)
    upload = dict(zip(adapter.trainable(), grads, strict=True))
    output = {f"mean_{layer}": values[layer][0, begin:end].double().mean(0) for layer in LAYERS}
    for layer in LAYERS:
        key = f"backend.language_model.model.layers.{layer - 1}.self_attn.q_proj.lora_A.default.weight"
        output[f"gradient_{layer}"] = upload[key].detach().cpu()
    return output


def collect_public(adapter, root, plan):
    rows = read_json(plan["public_bank"])
    q = adapter.encode([plan["training_question"]], adapter.spec.question_length)
    a = adapter.encode([plan["training_answer"]], adapter.spec.target_length)
    for index, row in enumerate(rows):
        output = public_feature(adapter, row, q, a)
        path = root / "public/features" / f"{index:03d}.safetensors"
        write_tensors(path, output)
        write_json(root / "progress.json", {"stage": "public_features", "completed": index + 1, "total": len(rows)})
        if index == 0:
            write_json(root / "first-artifact.json", {"stage": "public_features", "sha256": file_hash(path)})
        print(f"Public features {index + 1}/{len(rows)}", flush=True)
        del output
        gc.collect()
    write_json(root / "public/features-complete.json", {"images": len(rows), "hashes": {
        str(path): file_hash(path) for path in (root / "public/features").glob("*.safetensors")}})


def public_development(root, head_rule):
    rows = read_json(root / "public/abdomen-bank.json")
    features = [read_tensors(root / "public/features" / f"{i:03d}.safetensors") for i in range(len(rows))]
    values = {layer: torch.stack([data[f"mean_{layer}"] for data in features]) for layer in LAYERS}

    def prototypes(ids):
        return {f"visual_{layer}": torch.nn.functional.normalize(torch.stack([
            values[layer][[i for i in ids if rows[i]["label"] == label]].mean(0)
            for label in CLASSES["abdomen"]]), dim=1) for layer in LAYERS}

    folds = {}
    for style in ["photo", "line", "render"]:
        fit = []
        for other in [name for name in ["photo", "line", "render"] if name != style]:
            for label in CLASSES["abdomen"]:
                available = [i for i, row in enumerate(rows) if row["split"] == "fit"
                             and row["style"] == other and row["label"] == label]
                if label == "non-abdomen":
                    available = ([i for i in available if rows[i]["group"].startswith("head-")][:2]
                                 + [i for i in available if rows[i]["group"].startswith("non-head-")][:2])
                fit.extend(available[:4])
        query = [i for i, row in enumerate(rows) if row["split"] == "dev" and row["style"] == style]
        proto = prototypes(fit)
        scores = {f"gradient_{layer}": [distance_to_gradient(proto[f"visual_{layer}"],
                  features[i][f"gradient_{layer}"]) for i in query] for layer in LAYERS}
        folds[style] = {"fit_indices": fit, "query_indices": query, "scores": scores}
    rule = {**head_rule, "rms_scales": {}, "property": "abdomen",
            "selection": "Layer set and single layer inherited unchanged from public head development"}
    for layer in rule["layers"]:
        scale = float(np.sqrt(np.mean(np.concatenate([
            margins(fold["scores"][f"gradient_{layer}"]) for fold in folds.values()]) ** 2)))
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError("Degenerate public abdomen margin scale")
        rule["rms_scales"][str(layer)] = scale
    for fold in folds.values():
        fold["fusion"] = fuse(fold["scores"], rule)
    write_json(root / "public/abdomen-rule.json", rule)
    write_tensors(root / "public/abdomen-prototypes.safetensors", prototypes(
        [i for i, row in enumerate(rows) if row["split"] == "fit"]))
    write_json(root / "public/development-predictions.json", folds)
    write_json(root / "public/development-predictions-complete.json", {
        "sha256": file_hash(root / "public/development-predictions.json")})
    report = {}
    for style, fold in folds.items():
        labels = [rows[i]["label"] for i in fold["query_indices"]]
        report[style] = {name: classification_stats(labels, value, CLASSES["abdomen"])
                         for name, value in fold["fusion"].items()}
    write_json(root / "public/development-report.json", {"folds": report,
               "scope": "Public development only; no target-label selection or head-layer replacement"})
    write_json(root / "public/frozen.json", {"rule_sha256": file_hash(root / "public/abdomen-rule.json"),
               "prototype_sha256": file_hash(root / "public/abdomen-prototypes.safetensors")})


def capture_fresh(adapter, root):
    """Evaluator owns private images and QA; only anonymous Observations leave here."""
    from core.data import load_batch
    from core.fl import capture, save_observation
    from evaluation.public_style_study import controlled_training
    audit = read_json(root / "cohort-audit.json")
    if file_hash(root / "private/targets.json") != audit["targets_sha256"]:
        raise ValueError("Reviewed target cohort changed")
    adapter.set_training_spec(replace(controlled_training(adapter.training_spec), knowledge="private"))
    targets = read_json(root / "private/targets.json")
    random.SystemRandom().shuffle(targets)
    public, mapping = [], {}
    for index, row in enumerate(targets):
        identifier = uuid.uuid4().hex
        batch = load_batch([row["row"]], adapter)
        obs = capture(adapter, batch, [], [])
        path = root / "fresh/observations" / identifier
        save_observation(path, obs)
        public.append({"id": identifier, "observation": str(path)})
        mapping[identifier] = row["run"]
        write_json(root / "progress.json", {"stage": "capture", "completed": index + 1, "total": len(targets)})
        print(f"Fresh capture {index + 1}/{len(targets)}", flush=True)
        del batch, obs
        gc.collect()
    write_json(root / "private/upload-map.json", mapping)
    write_json(root / "fresh/upload-list.json", public)
    write_json(root / "fresh/capture-complete.json", {"observations": len(public),
               "hashes": {str(path): file_hash(path) for path in (root / "fresh/observations").rglob("*") if path.is_file()}})


def predict(root, fingerprint, head_path):
    """Scoring consumes public prototypes/rules and anonymous uploads, never target rows."""
    from core.fl import load_observation
    rules = {"head": read_json(root / "head-rule.json"), "abdomen": read_json(root / "public/abdomen-rule.json")}
    prototypes = {"head": read_tensors(head_path),
                  "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    predictions = []
    for item in read_json(root / "fresh/upload-list.json"):
        obs = load_observation(item["observation"], "cpu")
        if (obs.model_fingerprint != fingerprint or obs.training.knowledge != "private" or obs.sample_count != 1
                or obs.public_images is not None or obs.public_questions or obs.public_targets
                or obs.public_question_ids or obs.public_target_ids):
            raise ValueError("Scoring requires matching private single-sample Observations")
        scores = {}
        for property_name in CLASSES:
            individual = {}
            for layer in LAYERS:
                key = f"backend.language_model.model.layers.{layer - 1}.self_attn.q_proj.lora_A.default.weight"
                individual[f"gradient_{layer}"] = [distance_to_gradient(
                    prototypes[property_name][f"visual_{layer}"], obs.tensors[key])]
            scores[property_name] = {name: value[0] for name, value in fuse(individual, rules[property_name]).items()}
            scores[property_name].update({name: value[0] for name, value in individual.items()})
        predictions.append({"id": item["id"], "scores": scores})
        del obs
    path = root / "fresh/predictions.json"
    write_json(path, predictions)
    write_json(root / "fresh/predictions-complete.json", {"sha256": file_hash(path),
               "target_labels_read": False, "head_rule_sha256": file_hash(root / "head-rule.json"),
               "abdomen_rule_sha256": file_hash(root / "public/abdomen-rule.json")})


def evaluate(root):
    path = root / "fresh/predictions.json"
    if file_hash(path) != read_json(root / "fresh/predictions-complete.json")["sha256"]:
        raise ValueError("Committed predictions changed")
    predictions = read_json(path)
    mapping = read_json(root / "private/upload-map.json")
    targets = {row["run"]: row for row in read_json(root / "private/targets.json")}
    truth = [targets[mapping[row["id"]]] for row in predictions]
    output = {}
    for property_name, classes in CLASSES.items():
        groups = {"all": list(range(len(truth))),
                  "no_literal_property_terms": [i for i, row in enumerate(truth) if not row["literal_terms"][property_name]],
                  **{modality: [i for i, row in enumerate(truth) if row["modality"] == modality]
                     for modality in ["CT", "MRI", "X-Ray"]}}
        output[property_name] = {}
        for group, ids in groups.items():
            labels = [classes[0] if truth[i]["region"] == property_name else classes[1] for i in ids]
            scores = {name: [predictions[i]["scores"][property_name][name] for i in ids]
                      for name in predictions[0]["scores"][property_name]}
            output[property_name][group] = {name: {**classification_stats(labels, values, classes),
                  **(paired_bootstrap(labels, values, classes) or {})} for name, values in scores.items()}
    report = {"n": len(truth), "properties": output, "cohort": read_json(root / "cohort-audit.json"),
              "primary": "mean_standardized", "single_selected_layer": 24, "fusion_layers": [8, 16, 24, 32],
              "scope": "New image-ID isolated natural-QA pilot; modality confounded, patient independence unknown",
              "predictions_sha256": file_hash(path), "source_sha256": file_hash(Path(__file__))}
    write_json(root / "fresh/report.json", report)
    write_json("docs/experiments/data/visual-concept-probe/fresh-region-aggregate.json", report)
    print({name: {method: value["correct"] for method, value in groups["all"].items()}
           for name, groups in output.items()}, flush=True)
    return report


def run(root):
    from core.fl import restore_model
    from evaluation.public_style_study import controlled_training
    if (root / "probe-started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(root / "probe-started.json", {"pid": os.getpid(), "host": os.uname().nodename})
    start = time.monotonic()
    try:
        plan = read_json(root / "probe-plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen public input changed")
        torch.set_num_threads(1)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
            raise ValueError("Expose only the selected physical GPU")
        if torch.cuda.mem_get_info()[0] < 60 * 2 ** 30:
            raise RuntimeError("At least 60 GiB free required; do not change dtype or batch size")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        initial = adapter.fingerprint()
        adapter.set_training_spec(controlled_training(adapter.training_spec))
        collect_public(adapter, root, plan)
        public_development(root, read_json(plan["head_rule"]))
        capture_fresh(adapter, root)
        if adapter.fingerprint() != initial:
            raise ValueError("Victim model changed")
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
        predict(root, initial, plan["head_prototypes"])
        report = evaluate(root)
        write_json(root / "fresh/complete.json", {"status": "complete", "seconds": time.monotonic() - start,
                   "peak_memory_mib": peak, "report_sha256": file_hash(root / "fresh/report.json"), "n": report["n"]})
    except BaseException:
        write_json(root / "probe-failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.root.resolve())


if __name__ == "__main__":
    main()
