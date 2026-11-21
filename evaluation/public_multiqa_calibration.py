"""Public-only QA-diverse score calibration, followed by blind upload scoring."""

import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from evaluation.deep_property_probe import distance_to_gradient
from evaluation.image_property_scoring import classification_stats, paired_bootstrap
from evaluation.multilayer_property import LAYERS, OLD, fuse
from evaluation.multilayer_property_gpu import CLASSES


FEATURE_LAYERS = (8, 16, 24, 32)
QA = [
    {"id": "description", "split": "calibration", "q": "Describe the image.", "a": "The image is shown."},
    {"id": "visible", "split": "calibration", "q": "What is visible in this picture?", "a": "A visual subject is shown."},
    {"id": "present", "split": "calibration", "q": "Is there an image to inspect?", "a": "Yes."},
    {"id": "blank", "split": "calibration", "q": "Is the image completely blank?", "a": "No."},
    {"id": "summary", "split": "audit", "q": "Please summarize what you see.", "a": "There is a visible subject."},
    {"id": "examine", "split": "audit", "q": "Can this picture be examined?", "a": "Yes."},
]


def balanced_weights(labels, groups):
    """Give each class and content family equal mass; QA and style are repeats."""
    labels, groups = np.asarray(labels, dtype=bool), np.asarray(groups)
    if set(labels.tolist()) != {False, True}:
        raise ValueError("Both public classes are required")
    result = np.zeros(len(labels))
    for label in (False, True):
        families = set(groups[labels == label])
        for group in families:
            ids = (labels == label) & (groups == group)
            if len(set(labels[groups == group])) != 1:
                raise ValueError("A public family has inconsistent labels")
            result[ids] = .5 / len(families) / ids.sum()
    return result


def fit_linear(x, labels, groups, ridge=.1):
    """Closed-form balanced ridge on four margins; no target or hyperparameter fit."""
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    weight = balanced_weights(labels, groups)
    if x.shape != (len(labels), len(FEATURE_LAYERS)) or not np.isfinite(x).all():
        raise ValueError("Expected finite public layer margins")
    mean = (x * weight[:, None]).sum(0)
    scale = np.sqrt(((x - mean) ** 2 * weight[:, None]).sum(0)).clip(1e-12)
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    penalty = np.diag([0., *([ridge] * x.shape[1])])
    coefficient = np.linalg.solve(design.T @ (weight[:, None] * design) + penalty,
                                  design.T @ (weight * np.where(labels, 1., -1.)))
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coefficient": coefficient.tolist(),
            "ridge": ridge, "layers": list(FEATURE_LAYERS), "threshold": 0.,
            "weighting": "Equal classes, equal content families within class; QA/style/seed repeats share mass"}


def linear_margin(x, rule):
    values = np.asarray(x, dtype=float)
    return np.column_stack([np.ones(len(values)), (values - rule["mean"]) / rule["scale"]]) @ rule["coefficient"]


def fit_threshold(values, labels, groups):
    values, labels = np.asarray(values, dtype=float), np.asarray(labels, dtype=bool)
    weight = balanced_weights(labels, groups)
    unique = np.unique(values)
    candidates = np.r_[np.nextafter(unique[0], -np.inf),
                       (unique[:-1] + unique[1:]) / 2, np.nextafter(unique[-1], np.inf), 0.]
    threshold = max(candidates, key=lambda t: (float(weight[(values >= t) == labels].sum()), -abs(t), -t))
    return float(threshold)


def calibration_split(group):
    return "fit" if int(group.rsplit("-", 1)[1]) % 2 == 0 else "audit"


def prepare(root):
    destination = root / "multi-qa-plan-template.json"
    if destination.exists():
        raise FileExistsError("No overwrite of the public multi-QA plan")
    parent = read_json(root / "probe-plan.json")
    old = read_json(OLD / "styles/public_manifest.json")
    from evaluation.public_style_generation import generation_cases
    head_labels = {row["id"]: row["label"] for row in generation_cases()}
    rows, by_file = [], {}
    for index, row in enumerate(old):
        if row["split"] != "dev":
            continue
        entry = {"file": str(Path(row["file"]).resolve()), "group": row["group"], "style": row["style"],
                 "calibration_split": calibration_split(row["group"]),
                 "labels": {"head": head_labels[row["id"]]},
                 "baseline": str((OLD / "styles/features" / f"{index:03d}.safetensors").resolve())}
        by_file[entry["file"]] = len(rows)
        rows.append(entry)
    for index, row in enumerate(read_json(root / "public/abdomen-bank.json")):
        if row["split"] != "dev":
            continue
        path = str(Path(row["file"]).resolve())
        if path in by_file:
            entry = rows[by_file[path]]
            if entry["group"] != row["group"]:
                raise ValueError("Shared public file has inconsistent family")
            entry["labels"]["abdomen"] = row["label"]
        else:
            entry = {"file": path, "group": row["group"], "style": row["style"],
                     "calibration_split": calibration_split(row["group"]), "labels": {"abdomen": row["label"]},
                     "baseline": str((root / "public/features" / f"{index:03d}.safetensors").resolve())}
            rows.append(entry)
            by_file[path] = len(rows) - 1
    if len(rows) != 66:
        raise ValueError("Expected 66 distinct public development images")
    for prop, classes in CLASSES.items():
        for split in ("fit", "audit"):
            selected = [row for row in rows if prop in row["labels"] and row["calibration_split"] == split]
            balanced_weights([row["labels"][prop] == classes[0] for row in selected], [row["group"] for row in selected])
    uploads = read_json(root / "fresh/upload-list.json")
    files = [Path(parent["model"]) / name for name in ("model.json", "model.safetensors")]
    files += [Path(parent["head_prototypes"]), root / "public/abdomen-prototypes.safetensors",
              root / "head-rule.json", root / "public/abdomen-rule.json", root / "fresh/upload-list.json",
              root / "fresh/predictions.json", root / "fresh/predictions-complete.json"]
    files += [Path(row[key]) for row in rows for key in ("file", "baseline")]
    files += [Path(item["observation"]) / name for item in uploads for name in ("observation.json", "update.safetensors")]
    out = root / "multi-qa"
    write_json(out / "public-bank.json", rows)
    files.append(out / "public-bank.json")
    write_json(destination, {"model": parent["model"], "gpu_uuid": None, "public_images": len(rows),
               "public_bank": str((out / "public-bank.json").resolve()),
               "head_prototypes": parent["head_prototypes"], "qa": QA, "layers": list(LAYERS),
               "feature_layers": list(FEATURE_LAYERS), "ridge": .1, "target_images": len(uploads),
               "primary": "multi_ridge", "control": "single_ridge", "secondary": "Single-layer threshold calibration at 24 and 32",
               "qa_disclosure": "All public QA are identical across property classes and omit head, abdomen and modality labels",
               "public_audit": "Even public dev content families for calibration, odd for held-out image audit; all styles and seeds of a family stay together. Two QA templates excluded from all fitting.",
               "deployment": "After public audit, refit on all public development families using only the four calibration QA; no rule or hyperparameter selection",
               "boundary": "Original private natural-QA uploads unchanged; no reconstructed text, private images or private labels enter calibration or prediction",
               "scope": "Repeated 20-image diagnostic, not independent confirmation; medical modality/region confounding remains",
               "public_fit_excluded": "Prototypes remain the original fit-image prototypes; score calibration uses original dev images only",
               "source_sha256": file_hash(Path(__file__)),
               "hashes": {str(path.resolve()): file_hash(path) for path in files}})
    print({"public_images": len(rows), "qa": len(QA), "new_gradient_passes": len(rows) * (len(QA) - 1)}, flush=True)


def public_scores(adapter, root, plan):
    from evaluation.visual_concept_probe import inject_logits, native_hard_loss
    out = root / "multi-qa"
    rows = read_json(plan["public_bank"])
    prototypes = {"head": read_tensors(plan["head_prototypes"]),
                  "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    output, hashes = [], {}
    trainable = adapter.trainable()
    keys = [f"backend.language_model.model.layers.{layer-1}.self_attn.q_proj.lora_A.default.weight" for layer in LAYERS]
    encoded = {}
    for qa in plan["qa"]:
        for text, length in [(qa["q"], adapter.spec.question_length), (qa["a"], adapter.spec.target_length)]:
            if len(adapter.tokenizer.encode(text, add_special_tokens=False)) >= length:
                raise ValueError("Public text would truncate")
        encoded[qa["id"]] = (adapter.encode([qa["q"]], adapter.spec.question_length),
                             adapter.encode([qa["a"]], adapter.spec.target_length))
    for index, item in enumerate(rows):
        image = adapter.prepare_image(Image.open(item["file"]).convert("RGB")).unsqueeze(0).to(adapter.device, adapter.dtype)
        with torch.no_grad():
            visual = adapter.visual_embeddings(image)[0].detach()
        for qa in plan["qa"]:
            if qa["id"] == "description":
                stored = read_tensors(item["baseline"])
                gradients = {layer: stored[f"gradient_{layer}"] for layer in LAYERS}
                # Recompute one baseline to establish the partial autograd interface.
                if index == 0:
                    q, a = encoded[qa["id"]]
                    values = torch.autograd.grad(native_hard_loss(inject_logits(adapter, visual, q, a), a, adapter.pad),
                                                 tuple(trainable[key] for key in keys))
                    max_error = max(float((value.detach().cpu() - gradients[layer]).abs().max())
                                    for layer, value in zip(LAYERS, values, strict=True))
                    if max_error != 0:
                        raise ValueError("Native single-QA gradient replay differs from original public baseline")
                    write_json(out / "baseline-replay.json", {"max_absolute_error": max_error, "layers": list(LAYERS)})
            else:
                q, a = encoded[qa["id"]]
                loss = native_hard_loss(inject_logits(adapter, visual, q, a), a, adapter.pad)
                values = torch.autograd.grad(loss, tuple(trainable[key] for key in keys))
                gradients = {layer: value.detach().cpu() for layer, value in zip(LAYERS, values, strict=True)}
            if not all(torch.isfinite(value).all() for value in gradients.values()):
                raise ValueError("Nonfinite public gradient")
            path = out / "gradients" / f"{index:03d}-{qa['id']}.safetensors"
            write_tensors(path, {f"gradient_{layer}": value for layer, value in gradients.items()})
            scores = {prop: {str(layer): distance_to_gradient(prototypes[prop][f"visual_{layer}"], gradients[layer])
                      for layer in LAYERS} for prop in item["labels"]}
            output.append({"image": index, "qa": qa["id"], "scores": scores})
            hashes[str(path)] = file_hash(path)
            write_json(out / "progress.json", {"stage": "public_gradients", "completed": len(output),
                       "total": len(rows) * len(QA), "image": index, "qa": qa["id"]})
            if len(output) == 1:
                write_json(out / "first-artifact.json", {"sha256": file_hash(path), "baseline_replay": "exact"})
            print(f"Public multi-QA {len(output)}/{len(rows) * len(QA)}", flush=True)
    write_json(out / "public-scores.json", output)
    write_json(out / "public-scores-complete.json", {"sha256": file_hash(out / "public-scores.json"), "gradient_hashes": hashes})


def features(scores):
    return np.asarray([scores[str(layer)][1] - scores[str(layer)][0] for layer in FEATURE_LAYERS])


def margin_scores(value):
    return np.column_stack([-np.asarray(value), np.asarray(value)]).tolist()


def original_score_form(scores, method):
    """Old fusion stores a symmetric margin; single-layer distances need conversion."""
    if method == "original_fusion":
        return scores
    if method not in ("original_24", "original_32"):
        raise ValueError("Unknown original baseline representation")
    positive, negative = scores
    margin = negative - positive
    return [-margin, margin]


def fit_rules(records, bank, plan, split=None):
    result = {}
    for prop, classes in CLASSES.items():
        result[prop] = {}
        for condition, qas in [("single", {"description"}),
                               ("multi", {qa["id"] for qa in plan["qa"] if qa["split"] == "calibration"})]:
            ids = [row for row in records if prop in row["scores"] and row["qa"] in qas
                   and (split is None or bank[row["image"]]["calibration_split"] == split)]
            x = np.stack([features(row["scores"][prop]) for row in ids])
            labels = [bank[row["image"]]["labels"][prop] == classes[0] for row in ids]
            groups = [bank[row["image"]]["group"] for row in ids]
            result[prop][condition + "_ridge"] = fit_linear(x, labels, groups, plan["ridge"])
            for layer in (24, 32):
                column = FEATURE_LAYERS.index(layer)
                result[prop][f"{condition}_threshold_{layer}"] = {
                    "layer": layer, "threshold": fit_threshold(x[:, column], labels, groups)}
    return result


def calibrated_scores(scores, rules, original):
    x = features(scores)[None]
    result = {}
    for name, rule in rules.items():
        value = float(linear_margin(x, rule)[0]) if name.endswith("ridge") else (
            scores[str(rule["layer"])][1] - scores[str(rule["layer"])][0] - rule["threshold"])
        result[name] = [-value, value]
    for layer in (24, 32):
        result[f"original_{layer}"] = scores[str(layer)]
    result["original_fusion"] = fuse({f"gradient_{layer}": [scores[str(layer)]] for layer in LAYERS}, original)["mean_standardized"][0]
    return result


def public_audit(root, plan):
    out = root / "multi-qa"
    records, bank = read_json(out / "public-scores.json"), read_json(plan["public_bank"])
    if file_hash(out / "public-scores.json") != read_json(out / "public-scores-complete.json")["sha256"]:
        raise ValueError("Public score commit changed")
    rules = fit_rules(records, bank, plan, "fit")
    originals = {"head": read_json(root / "head-rule.json"), "abdomen": read_json(root / "public/abdomen-rule.json")}
    results = {}
    for prop, classes in CLASSES.items():
        rows = [row for row in records if prop in row["scores"] and bank[row["image"]]["calibration_split"] == "audit"]
        predictions = [{"image": row["image"], "qa": row["qa"],
                        "scores": calibrated_scores(row["scores"][prop], rules[prop], originals[prop])} for row in rows]
        metrics = {}
        for qa in plan["qa"]:
            selected = [row for row in predictions if row["qa"] == qa["id"]]
            labels = [bank[row["image"]]["labels"][prop] for row in selected]
            metrics[qa["id"]] = {name: classification_stats(labels, [row["scores"][name] for row in selected], classes)
                                  for name in predictions[0]["scores"]}
        robustness = {}
        for method in predictions[0]["scores"]:
            ids = sorted({row["image"] for row in predictions})
            by_image = [[float((row["scores"][method][1] - row["scores"][method][0]) / 2)
                         for row in predictions if row["image"] == image] for image in ids]
            robustness[method] = {"held_out_images": len(ids),
                "qa_prediction_flip_images": sum(min(x) < 0 <= max(x) for x in by_image),
                "mean_within_image_margin_std": float(np.mean([np.std(x) for x in by_image])),
                "worst_qa_balanced_accuracy": min(metrics[qa["id"]][method]["balanced_accuracy"] for qa in plan["qa"]),
                "held_out_qa_balanced_accuracy": float(np.mean([metrics[qa["id"]][method]["balanced_accuracy"]
                    for qa in plan["qa"] if qa["split"] == "audit"]))}
        results[prop] = {"by_qa": metrics, "robustness": robustness, "predictions": predictions}
    write_json(out / "public-audit.json", {"rules": rules, "properties": results,
               "note": "Image/content split and QA split both predetermined; image repeats are not independent samples"})
    deployment = fit_rules(records, bank, plan)
    write_json(out / "frozen-rules.json", deployment)
    write_json(out / "rules-complete.json", {"sha256": file_hash(out / "frozen-rules.json"),
               "public_audit_sha256": file_hash(out / "public-audit.json"), "private_labels_read": False,
               "qa_audit_used_for_fitting": False, "hyperparameters_selected": False})


def predict(root, plan):
    from core.fl import load_observation
    out = root / "multi-qa"
    rules = read_json(out / "frozen-rules.json")
    if file_hash(out / "frozen-rules.json") != read_json(out / "rules-complete.json")["sha256"]:
        raise ValueError("Frozen public rules changed")
    originals = {"head": read_json(root / "head-rule.json"), "abdomen": read_json(root / "public/abdomen-rule.json")}
    proto = {"head": read_tensors(plan["head_prototypes"]), "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
    fingerprint = read_json(Path(plan["model"]) / "model.json")["fingerprint"]
    predictions = []
    for item in read_json(root / "fresh/upload-list.json"):
        obs = load_observation(item["observation"], "cpu")
        if (obs.model_fingerprint != fingerprint or obs.training.knowledge != "private" or obs.sample_count != 1
                or obs.public_images is not None or obs.public_questions or obs.public_targets
                or obs.public_question_ids or obs.public_target_ids or obs.training.token_lengths_known):
            raise ValueError("Only original private natural-QA uploads are allowed")
        scores = {}
        for prop in CLASSES:
            raw = {str(layer): distance_to_gradient(proto[prop][f"visual_{layer}"], obs.tensors[
                   f"backend.language_model.model.layers.{layer-1}.self_attn.q_proj.lora_A.default.weight"])
                   for layer in LAYERS}
            scores[prop] = calibrated_scores(raw, rules[prop], originals[prop])
        predictions.append({"id": item["id"], "scores": scores})
    write_json(out / "predictions.json", predictions)
    write_json(out / "predictions-complete.json", {"sha256": file_hash(out / "predictions.json"),
               "rules_sha256": file_hash(out / "frozen-rules.json"), "target_labels_read": False})


def evaluate(root, plan):
    out = root / "multi-qa"
    if file_hash(out / "predictions.json") != read_json(out / "predictions-complete.json")["sha256"]:
        raise ValueError("Target prediction commit changed")
    predictions = read_json(out / "predictions.json")
    # Target truth joins only here, after all blind predictions have been saved.
    mapping = read_json(root / "private/upload-map.json")
    targets = {row["run"]: row for row in read_json(root / "private/targets.json")}
    truth = [targets[mapping[row["id"]]] for row in predictions]
    baseline = {row["id"]: row for row in read_json(root / "fresh/predictions.json")}
    results = {}
    for prop, classes in CLASSES.items():
        for row in predictions:
            for name, original in [("original_24", "single_selected"), ("original_32", "single_32"), ("original_fusion", "mean_standardized")]:
                if not np.array_equal(original_score_form(row["scores"][prop][name], name),
                                      baseline[row["id"]]["scores"][prop][original]):
                    raise ValueError("Original target scores were not reproduced exactly")
        groups = {"all": list(range(len(truth))),
                  "no_literal_property_terms": [i for i, row in enumerate(truth) if not row["literal_terms"][prop]],
                  **{m: [i for i, row in enumerate(truth) if row["modality"] == m] for m in ("CT", "MRI", "X-Ray")}}
        results[prop] = {}
        for name, ids in groups.items():
            labels = [classes[0] if truth[i]["region"] == prop else classes[1] for i in ids]
            scores = {method: [predictions[i]["scores"][prop][method] for i in ids] for method in predictions[0]["scores"][prop]}
            metrics = {method: {**classification_stats(labels, values, classes),
                       **(paired_bootstrap(labels, values, classes) or {})} for method, values in scores.items()}
            paired = {method: {"accuracy_gain": metrics[method]["accuracy"] - metrics["single_ridge"]["accuracy"],
                       **paired_bootstrap(labels, values, classes, scores["single_ridge"])}
                      for method, values in scores.items() if labels and method != "single_ridge"}
            results[prop][name] = {"metrics": metrics, "paired_vs_single_ridge": paired}
    write_json(out / "report.json", {"n": len(truth), "properties": results, "primary": plan["primary"],
               "control": plan["control"], "scope": plan["scope"], "uses_target_text": False,
               "public_images": plan["public_images"], "qa_templates": plan["qa"],
               "predictions_sha256": file_hash(out / "predictions.json"),
               "rules_sha256": file_hash(out / "frozen-rules.json")})
    print({prop: {name: v["correct"] for name, v in group["all"]["metrics"].items()} for prop, group in results.items()}, flush=True)


def run(root):
    from core.fl import restore_model
    from evaluation.public_style_study import controlled_training
    out = root / "multi-qa"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(out / "started.json", {"pid": os.getpid(), "host": os.uname().nodename})
    start = time.monotonic()
    try:
        plan = read_json(out / "plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen input changed")
        if file_hash(Path(__file__)) != plan["source_sha256"]:
            raise ValueError("Calibration source changed")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly the chosen idle GPU")
        if torch.cuda.mem_get_info()[0] < 60 * 2 ** 30:
            raise RuntimeError("At least 60 GiB free required; no automatic changes")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        fingerprint = adapter.fingerprint()
        adapter.set_training_spec(controlled_training(adapter.training_spec))
        public_scores(adapter, root, plan)
        if adapter.fingerprint() != fingerprint:
            raise ValueError("Calibration changed victim weights")
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        del adapter
        torch.cuda.empty_cache()
        public_audit(root, plan)
        predict(root, plan)
        evaluate(root, plan)
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
    try:
        (prepare if args.prepare else run)(args.root.resolve())
    except BaseException:
        write_json(args.root / "multi-qa" / ("prepare-failed.json" if args.prepare else "run-failed.json"),
                   {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
