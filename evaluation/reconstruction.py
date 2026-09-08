"""Ground truth is consumed only here, after reconstruction is committed."""
from collections import Counter, defaultdict
from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from core.artifacts import read_json, read_tensors


from metrics.text import words, edit_distance, pii_exact_match_recall, text_metrics
from metrics.image import image_metrics
from metrics.semantic import OptionalMetrics

def match_pairs(reference_images, prediction_images, reference_texts, prediction_texts, use_text=True):
    n = len(reference_images)
    if n != len(prediction_images) or n != len(reference_texts) or n != len(prediction_texts):
        raise ValueError("Reconstruction cardinality differs from observed sample slots")
    image_cost, text_cost = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            image_cost[i, j] = float((reference_images[i] - prediction_images[j]).square().mean())
            ref, pred = words(reference_texts[i]), words(prediction_texts[j])
            text_cost[i, j] = edit_distance(ref, pred) / max(1, len(ref), len(pred))
    _, assignment = linear_sum_assignment(image_cost + (text_cost if use_text else 0))
    _, image_assignment = linear_sum_assignment(image_cost)
    _, text_assignment = linear_sum_assignment(text_cost)
    agreement = float(np.mean(image_assignment == text_assignment)) if use_text and n > 1 else None
    return assignment.tolist(), agreement


def canary_metrics(row, reference_text, prediction_text):
    """Recall over the synthetic entities that actually reached the trained tokens.

    An entity injected into the manifest can be truncated away by the model's
    question/target budget, and one that was never trained on cannot leak. Scoring
    it would understate recall, so survivors are counted separately and the drop is
    reported rather than hidden.
    """
    declared = row.get("canary") or []
    if not declared:
        return {}
    survived = [value for value in declared if value in reference_text]
    return {"canary_declared": len(declared), "canary_trained": len(survived),
            "canary_recall": pii_exact_match_recall(survived, prediction_text)}


def evaluate(reconstruction_dir, truth_dir, spec, device="cpu"):
    reconstruction_dir, truth_dir = Path(reconstruction_dir), Path(truth_dir)
    result = read_json(reconstruction_dir / "result.json")
    truth = read_json(truth_dir / "truth.json")
    if result["observation_id"] != truth["observation_id"]:
        raise ValueError("Ground truth belongs to a different observation")
    report = {"schema_version": 1, "observation_id": result["observation_id"],
              "status": result["status"], "reason": result.get("reason", ""),
              "condition": result["condition"], "costs": result.get("costs", {}), "samples": []}
    report["evaluation_config"] = asdict(spec)
    if result["status"] != "completed":
        return report
    true_images = read_tensors(truth_dir / "images.safetensors")["images"]
    pred_images = read_tensors(reconstruction_dir / "images.safetensors")["images"]
    if true_images.shape != pred_images.shape or not torch.isfinite(pred_images).all():
        raise ValueError("Invalid reconstructed image tensor")
    if pred_images.min() < 0 or pred_images.max() > 1:
        raise ValueError("Reconstructed RGB images must be in [0,1]")
    rows = truth["samples"]
    t = truth["training"]
    private_q = t["task"] == "vqa" and t["knowledge"] == "private"
    private_y = t["knowledge"] != "text_known"
    reference_texts = [(r["model_question"] + " " if private_q else "") +
                       (r["model_target"] if private_y else "") for r in rows]
    prediction_texts = [(q + " " if private_q else "") + (y if private_y else "")
                        for q, y in zip(result["questions"], result["targets"])]
    assignment, agreement = match_pairs(true_images, pred_images, reference_texts, prediction_texts,
                                        use_text=private_q or private_y)
    optional = OptionalMetrics(spec, device)
    report.update({"metric_status": optional.status, "pair_assignment": assignment,
                   "pair_assignment_agreement": agreement,
                   "alignment": "Hungarian(image_MSE + normalized_private_text_edit); one joint assignment",
                   "reference_policy": "actual trained tokens after truncation, excluding EOS/pad"})
    for i, j in enumerate(assignment):
        metrics = image_metrics(true_images[i], pred_images[j])
        if private_q:
            metrics.update({"question_" + k: v for k, v in text_metrics(
                rows[i]["model_question"], result["questions"][j]).items()})
        if private_y:
            metrics.update({"target_" + k: v for k, v in text_metrics(
                rows[i]["model_target"], result["targets"][j]).items()})
        metrics.update(optional.score(true_images[i], pred_images[j], rows[i]["model_target"],
                                      result["targets"][j], t["task"]))
        metrics.update(canary_metrics(rows[i], reference_texts[i], prediction_texts[j]))
        report["samples"].append({"image_id": rows[i]["image_id"], "sample_id": rows[i]["sample_id"],
                                   "prediction_index": j, "metrics": metrics})
    return report


def summarize(reports, bootstrap=1000, seed=42):
    groups = {}
    for report in reports:
        condition = dict(report["condition"])
        condition.pop("seed", None)
        if "evaluation_config" in report:
            from core.config import digest
            condition["evaluation_protocol_hash"] = digest(report["evaluation_config"])
        key = str(sorted(condition.items()))
        group = groups.setdefault(key, {"condition": condition, "statuses": Counter(),
                                        "images": defaultdict(lambda: defaultdict(list)), "costs": []})
        group["statuses"][report["status"]] += 1
        group["costs"].append(report.get("costs", {}))
        for row in report["samples"]:
            for name, value in row["metrics"].items():
                if isinstance(value, (int, float)) and math.isfinite(value):
                    group["images"][row["image_id"]][name].append(value)
    output = []
    rng = np.random.default_rng(seed)
    for group in groups.values():
        metric_values = defaultdict(list)
        for metrics in group["images"].values():
            for name, values in metrics.items():
                metric_values[name].append(float(np.mean(values)))
        summaries = {}
        for name, values in metric_values.items():
            values = np.array(values)
            boot = [rng.choice(values, len(values), replace=True).mean() for _ in range(bootstrap)]
            summaries[name] = {"mean": float(values.mean()), "image_groups": len(values),
                               "ci95": np.percentile(boot, [2.5, 97.5]).tolist() if boot else None}
        output.append({"condition": group["condition"], "statuses": dict(group["statuses"]),
                       "metrics": summaries, "costs": group["costs"]})
    return {"groups": output, "bootstrap_unit": "image_id; seeds/duplicate annotations averaged first",
            "bootstrap": bootstrap, "seed": seed,
            "note": "Quality is conditional on completed outputs; inspect every status denominator."}
