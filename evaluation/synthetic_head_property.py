"""Fixed mixed-modality head property bank: evaluator capture, public inference, then evaluation."""

from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.config import load_config
from core.data import load_batch
from core.fl import capture, load_observation, restore_model, save_observation, simulate_update
from evaluation.dager_rank_study import GPU5, attack_public
from evaluation.image_property_scoring import METHODS, classification_stats, paired_bootstrap
from evaluation.image_property_study import PublicWorker
from evaluation.token_image_ordering import GradientScorer


ROOT = Path("outputs/diagnostics/synthetic_head_n20_gpu5_20261017")


def capture_targets(adapter, config, plan):
    """Private evaluator stage; no reference data is passed to prediction code."""
    path = ROOT / "private/targets.json"
    if file_hash(path) != plan["target_manifest_sha256"]:
        raise ValueError("Target selection changed")
    adapter.set_training_spec(replace(config.training, knowledge="text_known", server_round=10))
    errors = []
    for item in read_json(path):
        i = item["run"]
        batch = load_batch([item["row"]], adapter)
        question, answer = adapter.decode(batch.questions)[0], adapter.decode(batch.targets)[0]
        obs = capture(adapter, batch, [question], [answer])
        private = replace(obs, training=replace(obs.training, knowledge="private"),
                          public_questions=[], public_targets=[], public_question_ids=[],
                          public_target_ids=[])
        private.validate()
        folder = ROOT / "captures" / f"run-{i:05d}"
        save_observation(folder / "oracle", obs)
        save_observation(folder / "private", private)
        write_tensors(ROOT / "private" / f"reference-{i:05d}.safetensors",
                      {"questions": batch.questions, "targets": batch.targets})
        error = GradientScorer(obs.tensors, adapter.device).score(
            simulate_update(adapter, batch, obs.training))["all"]["relative_l2"]
        if error > 1e-12:
            raise ValueError("Exact capture replay failed")
        if (read_json(folder / "oracle/observation.json")["update_sha256"] !=
                read_json(folder / "private/observation.json")["update_sha256"]):
            raise ValueError("Paired observation mismatch")
        errors.append(error)
        write_json(ROOT / "progress.json", {"stage": "capture", "completed": i + 1, "total": plan["targets"]})
        print(f"Captured {i + 1}/{plan['targets']}; exact replay {error}", flush=True)
    write_json(ROOT / "capture_complete.json", {"observations": plan["targets"], "max_relative_l2": max(errors),
               "hashes": {str(p.relative_to(ROOT)): file_hash(p)
                          for p in (ROOT / "captures").rglob("*") if p.is_file()}})


def predict(adapter, plan, phase, model_fingerprint):
    """Consumes only declared observations, generated bank, and attack outputs."""
    folder = ROOT / phase
    folder.mkdir(exist_ok=False)
    known = phase == "oracle"
    observations = [load_observation(ROOT / "captures" / f"run-{i:05d}" /
                                    ("oracle" if known else "private"), adapter.device)
                    for i in range(plan["targets"])]
    if any(o.model_fingerprint != model_fingerprint for o in observations):
        raise ValueError("Observation model mismatch")
    expected_knowledge = "text_known" if known else "private"
    for obs in observations:
        if (obs.sample_count != 1 or obs.training.knowledge != expected_knowledge
                or not obs.training.token_lengths_known or obs.training.algorithm != "fedsgd"
                or asdict(obs.training) != asdict(observations[0].training)):
            raise ValueError("Unsupported or incompatible observation")
        if not known and any([obs.public_questions, obs.public_targets,
                              obs.public_question_ids, obs.public_target_ids]):
            raise ValueError("Private observation exposes text")
    adapter.set_training_spec(observations[0].training)
    bank = read_json(ROOT / "public/bank.json")
    raw = read_tensors(ROOT / "public/bank.safetensors")
    images = [adapter.prepare_image(Image.fromarray(raw[r["id"]].numpy())).unsqueeze(0)
              .to(adapter.device, adapter.dtype) for r in bank]
    public_plan = {key: plan[key] for key in ["classes", "top_m", "text_only_prompt"]}
    worker = PublicWorker(adapter, observations, bank, images, folder, public_plan)
    for i, obs in enumerate(observations):
        if known:
            if obs.training.knowledge != "text_known":
                raise ValueError("Correct text requires text_known observation")
            text = {"questions": torch.tensor(obs.public_question_ids, device=adapter.device),
                    "targets": torch.tensor(obs.public_target_ids, device=adapter.device)}
        else:
            reconstruction = ROOT / "reconstruction" / f"run-{i:05d}"
            status = read_json(reconstruction / "result.json")["status"]
            if status != "completed":
                # An unsupported/no-signal/budget-limited result is an abstention,
                # not a reason to remove a target or substitute oracle text.
                write_json(folder / "predictions" / f"{i:05d}-{phase}.json", {
                    "run": i, "condition": phase, "reconstruction_status": status,
                    "class_scores": dict.fromkeys(METHODS), "model_only": None, "text_only": None,
                    "shuffled_target_runs": [j for j in range(plan["targets"]) if j != i],
                    "shuffled_class_scores": [dict.fromkeys(METHODS)
                                              for j in range(plan["targets"]) if j != i],
                })
                continue
            text = read_tensors(reconstruction / "text_tokens.safetensors", str(adapter.device))
        worker.predict(i, phase, text)
        write_json(ROOT / "progress.json", {"stage": phase, "completed": i + 1, "total": plan["targets"]})
        if i == 0:
            write_json(folder / "first_checkpoint.json", {"prediction_committed": True,
                       "gradient_evaluations": worker.evaluations})
    write_json(folder / "complete.json", {"predictions": plan["targets"],
               "gradient_evaluations": worker.evaluations, "unique_texts": len(worker.cache),
               "private_reference_labels_read": False,
               "hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in folder.rglob("*.json")}})
    del worker, observations, raw, images
    gc.collect()


def evaluate(plan):
    """Read property labels only after BOTH prediction phases are committed."""
    for phase in ["oracle", "reconstructed"]:
        for path, expected in read_json(ROOT / phase / "complete.json")["hashes"].items():
            if file_hash(ROOT / path) != expected:
                raise ValueError("Committed predictions changed")
    truth = read_json(ROOT / "private/targets.json")
    labels = [r["label"] for r in truth]
    classes = plan["classes"]
    report = {"classes": classes, "primary_method": plan["primary_score"],
              "plan_sha256": file_hash(ROOT / "plan.json"), "phases": {}}
    for phase in ["oracle", "reconstructed"]:
        predictions = [read_json(ROOT / phase / "predictions" / f"{i:05d}-{phase}.json")
                       for i in range(plan["targets"])]
        scores = {m: [p["class_scores"][m] for p in predictions] for m in METHODS}
        scores.update({m: [p[m] for p in predictions] for m in ["model_only", "text_only"]})
        results = {m: {**classification_stats(labels, values, classes),
                       **paired_bootstrap(labels, values, classes)} for m, values in scores.items()}
        wrong = {}
        for method in METHODS:
            all_credit, same_credit, opposite_credit = [], [], []
            for i, pred in enumerate(predictions):
                group = {"all": [], "same": [], "opposite": []}
                for j, values in zip(pred["shuffled_target_runs"], pred["shuffled_class_scores"], strict=True):
                    s = values[method]
                    credit = float(s is not None and classes[int(np.argmin(s))] == labels[i])
                    group["all"].append(credit)
                    group["same" if labels[i] == labels[j] else "opposite"].append(credit)
                all_credit.append(float(np.mean(group["all"])))
                same_credit.append(float(np.mean(group["same"])))
                opposite_credit.append(float(np.mean(group["opposite"])))
            wrong[method] = {"accuracy": float(np.mean(all_credit)),
                             "per_target_accuracy": all_credit,
                             "same_class_accuracy": float(np.mean(same_credit)),
                             "opposite_class_accuracy": float(np.mean(opposite_credit)),
                             "pairs": plan["targets"] * (plan["targets"] - 1), "independent_images": plan["targets"]}
        main = scores[plan["primary_score"]]
        paired = {m: paired_bootstrap(labels, main, classes, scores[m])
                  for m in ["model_only", "text_only"]}
        rows = [{"run": i, "label": labels[i],
                 "primary_prediction": classes[int(np.argmin(main[i]))] if main[i] is not None else None,
                 "primary_scores": main[i],
                 "text_only_prediction": classes[int(np.argmin(scores["text_only"][i]))]
                    if scores["text_only"][i] is not None else None,
                 "model_only_prediction": classes[int(np.argmin(scores["model_only"][i]))]
                    if scores["model_only"][i] is not None else None}
                for i in range(plan["targets"])]
        report["phases"][phase] = {"metrics": results, "wrong_gradient": wrong,
                                    "paired_gain_ci95": paired, "per_target": rows}
    text_rows = []
    for i in range(plan["targets"]):
        reconstruction = ROOT / "reconstruction" / f"run-{i:05d}"
        status = read_json(reconstruction / "result.json")["status"]
        ref = read_tensors(ROOT / "private" / f"reference-{i:05d}.safetensors")
        rec = read_tensors(reconstruction / "text_tokens.safetensors") if status == "completed" else None
        exact = {f: rec is not None and torch.equal(rec[f], ref[f]) for f in ["questions", "targets"]}
        text_rows.append({"run": i, "question_em": exact["questions"],
                          "answer_em": exact["targets"], "joint_em": all(exact.values()),
                          "status": status})
    report["text_recovery"] = {"per_target": text_rows, "counts": {
        k: sum(r[k] for r in text_rows) for k in ["question_em", "answer_em", "joint_em"]}}
    write_json(ROOT / "report.json", report)
    return report


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError("Only authorized GPU 5 may be exposed")
    if (ROOT / "started.json").exists():
        raise FileExistsError("No automatic retry or overwrite")
    torch.set_num_threads(1)
    started = time.monotonic()
    write_json(ROOT / "started.json", {"plan_sha256": file_hash(ROOT / "plan.json")})
    try:
        plan = read_json(ROOT / "plan.json")
        for path, expected in plan["frozen_hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("Frozen input changed")
        adapter = restore_model(plan["model"], "cuda:0")
        fingerprint = adapter.fingerprint()
        versions = {n: p._version for n, p in adapter.named_parameters()}
        config = load_config(plan["config"])
        config = replace(config, training=replace(config.training, server_round=10),
                         attack=replace(config.attack, dager=replace(config.attack.dager, lm_prior_weight=0.01)))
        write_json(ROOT / "reconstruction/config.json", config)
        if plan.get("reuse_verified_capture", False):
            captured = read_json(ROOT / "capture_complete.json")
            if captured["observations"] != plan["targets"] or captured["max_relative_l2"] > 1e-12:
                raise ValueError("Cannot reuse unverified capture")
            for path, expected in captured["hashes"].items():
                if file_hash(ROOT / path) != expected:
                    raise ValueError("Reused capture changed")
            print("Reusing verified captures from the recorded preparation phase", flush=True)
        else:
            capture_targets(adapter, config, plan)
        for i in range(plan["targets"]):
            write_json(ROOT / "progress.json", {"stage": "reconstruct", "completed": i, "total": plan["targets"]})
            status, costs = attack_public(adapter, config, ROOT / "captures" / f"run-{i:05d}" / "private",
                                          ROOT / "reconstruction" / f"run-{i:05d}", fingerprint)
            if status not in {"completed", "no_signal", "no_candidates", "budget_exhausted"}:
                raise RuntimeError(f"Unexpected reconstruction status: {status}")
            print(f"Reconstructed {i + 1}/{plan['targets']}; {status}; {costs['seconds']:.1f} s", flush=True)
            gc.collect()
        write_json(ROOT / "reconstruction/complete.json", {"reconstructions": plan["targets"],
                   "hashes": {str(p.relative_to(ROOT)): file_hash(p)
                              for p in (ROOT / "reconstruction").rglob("*") if p.is_file()}})
        predict(adapter, plan, "reconstructed", fingerprint)
        predict(adapter, plan, "oracle", fingerprint)
        evaluate(plan)
        if adapter.fingerprint() != fingerprint or any(p._version != versions[n] for n, p in adapter.named_parameters()):
            raise ValueError("Victim model changed")
        for path, expected in plan["frozen_hashes"].items():
            if file_hash(path) != expected:
                raise ValueError("Frozen input changed during run")
        write_json(ROOT / "complete.json", {"seconds": time.monotonic() - started,
                   "report_sha256": file_hash(ROOT / "report.json"), "model_unchanged": True,
                   "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
        print("Mixed-modality head property experiment complete", flush=True)
    except BaseException:
        write_json(ROOT / "failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
