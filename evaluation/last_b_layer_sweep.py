"""All 32 down/o LoRA-B layers, same uploads and fixed candidate budgets."""
import argparse
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import shutil
import time
import traceback

import torch

from attacks.analytic.dager_adapted import forbidden_token_ids
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation, restore_model
from evaluation import last_b_token_readout as base
from evaluation.last_b_positions import POSITIONS as CONTROLS
from evaluation.last_b_sweep_math import (
    METHODS, POSITIONS, cached_scores, layer_ranking, method_info, order_and_ranks,
    public_features, selection, supplement, vector_metrics,
)
from evaluation.soft_residual_filter_study import verify_plan


def prepare(output, source, previous):
    base.prepare(output, source)
    plan = read_json(output / "plan.json")
    old = read_json(previous / "plan.json")
    if old["model_fingerprint"] != plan["model_fingerprint"] or old["baselines"] != plan["baselines"]:
        raise ValueError("Historical model or observations differ")
    scores, candidates = {}, {}
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            names = read_json(source / "runs" / job / "captures/random/public/observation.json")["parameter_names"]
            if any(name not in names for name in POSITIONS.values()):
                raise ValueError("A declared layer/module is missing from uploaded parameters")
            scores[job] = file_hash(previous / "scores" / job / "scores.safetensors")
            for image in ("gt", "random"):
                candidates[f"{job}:{image}"] = file_hash(previous / "candidates" / image / job / "candidates.safetensors")
    prior_sha = file_hash(previous / "prior.safetensors")
    if prior_sha != old["public_prior_sha256"]:
        raise ValueError("Historical public prior changed")
    shutil.copyfile(previous / "prior.safetensors", output / "prior.safetensors")
    plan.update(positions=POSITIONS, methods=list(METHODS), controls=CONTROLS,
        previous=str(previous.resolve()), previous_scores=scores, previous_candidates=candidates,
        public_prior_sha256=prior_sha, score_groups=90, position_scores=5760,
        candidate_groups=180, candidate_conditions=35100,
        development_selection="Random mean Noise/Top-k mix Answer recall; worst recall, mean answer coverage, deeper layer, down module tie breaks",
        independent_validation=False, public_word_features_cached=True,
        source_boundary="All 64 word scores and matched/wrong/fixed supplements commit before reference metrics")
    write_json(output / "plan.json", plan)


class ScoreOrders:
    """At most three public score bundles and their context-dependent rankings."""
    def __init__(self, output):
        self.output = output

    @lru_cache(maxsize=3)
    def entry(self, condition, index):
        folder = self.output / "scores" / f"{condition}-{index:02d}"
        return {"scores": read_tensors(folder / "scores.safetensors"), "contexts": {}}

    def get(self, condition, index, flags, forbidden):
        entry = self.entry(condition, index)
        key = (tuple(torch.nonzero(flags).flatten().tolist()), tuple(sorted(forbidden)))
        if key not in entry["contexts"]:
            entry["contexts"][key] = {m: order_and_ranks(s, flags, forbidden) for m, s in entry["scores"].items()}
        return entry["contexts"][key]


def candidates_for(base_order, prior_order, current, wrong, ambiguous):
    chosen = {"baseline": selection(base_order, ambiguous), "lm_prior": selection(prior_order, ambiguous),
              "mix_lm_prior": supplement(base_order, prior_order, ambiguous)}
    for position in POSITIONS:
        chosen[position] = selection(current[position][0], ambiguous)
        chosen["wrong_" + position] = selection(wrong[position][0], ambiguous)
        chosen["mix_" + position] = supplement(base_order, current[position][0], ambiguous)
    if set(chosen) != set(METHODS) or {len(v) for v in chosen.values()} != {109}:
        raise ValueError("All-layer candidate surface or budget changed")
    return chosen


@torch.no_grad()
def score_all(output):
    plan = verify_plan(output)
    if plan["positions"] != POSITIONS or plan["methods"] != list(METHODS):
        raise ValueError("Frozen all-layer sweep changed")
    if file_hash(output / "prior.safetensors") != plan["public_prior_sha256"]:
        raise ValueError("Public prior changed")
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Victim model fingerprint mismatch")
    parameters = dict(adapter.named_parameters())
    versions = {name: p._version for name, p in parameters.items()}
    if any(parameters[name].shape != (4096, 32) for name in POSITIONS.values()):
        raise ValueError("Unexpected B weight coordinate")
    head = adapter.backend.language_model.lm_head.weight.detach().double()
    gamma = adapter.backend.language_model.model.norm.weight.detach().double()
    features, norms = public_features(head, gamma)
    del head, gamma
    forbidden = forbidden_token_ids(adapter)
    count = 0
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            public = Path(plan["source"]) / "runs" / job / "captures/random/public"
            if file_hash(public / "observation.json") != plan["sources"][job + ":random"]["metadata_sha256"]:
                raise ValueError("Observation metadata changed")
            observation = load_observation(public, "cpu")
            if observation.public_images is not None or observation.model_fingerprint != plan["model_fingerprint"]:
                raise ValueError("Observation image/fingerprint boundary violated")
            started, values = time.monotonic(), {}
            for position, name in POSITIONS.items():
                gradient = observation.tensors[name].to(device=features.device, dtype=torch.float64)
                if gradient.shape != parameters[name].shape:
                    raise ValueError("Uploaded B shape mismatch")
                values[position] = cached_scores(gradient, features, norms).cpu()
            path = Path(plan["previous"]) / "scores" / job / "scores.safetensors"
            if file_hash(path) != plan["previous_scores"][job]:
                raise ValueError("Historical position score changed")
            old = read_tensors(path)
            deltas = {m: float((values[m] - old[m]).abs().max()) for m in CONTROLS}
            if max(deltas.values()) > 1e-12:
                raise ValueError(f"Historical readout mismatch: {deltas}")
            folder = output / "scores" / job
            if folder.exists():
                raise FileExistsError("No implicit score retry")
            write_tensors(folder / "scores.safetensors", values)
            write_json(folder / "complete.json", {"status": "completed", "positions": POSITIONS,
                "vocab_rows": len(features), "control_max_deltas": deltas,
                "score_sha256": file_hash(folder / "scores.safetensors"),
                "source_sha256": plan["source_sha256"], "forbidden_ids": sorted(forbidden),
                "seconds": time.monotonic() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated()})
            count += 1
            write_json(output / "progress.json", {"phase": "word_scores", "completed": count, "total": 90})
            print("committed", job, "64 positions; control delta", max(deltas.values()), flush=True)
            del observation, gradient, values
    cache = ScoreOrders(output)
    prior = read_tensors(output / "prior.safetensors")["lm_prior"]
    groups = 0
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            for image in ("gt", "random"):
                original = base.historical(output, plan, condition, index, image)
                flags = original["ambiguous"]
                ambiguous = set(torch.nonzero(flags).flatten().tolist()) - forbidden
                current = cache.get(condition, index, flags, forbidden)
                wrong = cache.get(condition, (index + 1) % 30, flags, forbidden)
                order, _ = order_and_ranks(original["scores"], flags, forbidden)
                prior_order, _ = order_and_ranks(prior, flags, forbidden)
                selected = candidates_for(order, prior_order, current, wrong, ambiguous)
                path = Path(plan["previous"]) / "candidates" / image / job / "candidates.safetensors"
                if file_hash(path) != plan["previous_candidates"][job + ":" + image]:
                    raise ValueError("Historical candidate artifact changed")
                for method, ids in read_tensors(path).items():
                    if selected[method] != set(ids.tolist()):
                        raise ValueError("Historical candidate set mismatch")
                folder = output / "candidates" / image / job
                write_tensors(folder / "candidates.safetensors", {
                    m: torch.tensor(sorted(v), dtype=torch.long) for m, v in selected.items()})
                write_json(folder / "complete.json", {"status": "completed", "count": 109,
                    "historical_candidates_exact": True, "sha256": file_hash(folder / "candidates.safetensors")})
                groups += 1
                write_json(output / "progress.json", {"phase": "candidates", "completed": groups, "total": 180})
    if any(p._version != versions[name] for name, p in parameters.items()):
        raise ValueError("Scoring mutated the public victim")
    write_json(output / "all-scores-complete.json", {"status": "completed", "score_groups": 90,
        "position_scores": 5760, "candidate_groups": 180, "candidate_conditions": 35100,
        "initial_model_immutable": True, "private_reference_read": False, "time": time.time()})
    write_json(output / "progress.json", {"status": "completed", "phase": "scoring_finished"})


def preflight(output, plan):
    completion = read_json(output / "all-scores-complete.json")
    if completion["status"] != "completed" or completion["candidate_conditions"] != 35100:
        raise ValueError("Incomplete all-layer scoring matrix")
    if file_hash(output / "prior.safetensors") != plan["public_prior_sha256"]:
        raise ValueError("Prior commitment changed")
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            folder = output / "scores" / job
            meta = read_json(folder / "complete.json")
            if (file_hash(folder / "scores.safetensors") != meta["score_sha256"] or
                    set(meta["positions"]) != set(POSITIONS) or max(meta["control_max_deltas"].values()) > 1e-12):
                raise ValueError("Word-score commitment changed")
            for image in ("gt", "random"):
                folder = output / "candidates" / image / job
                if file_hash(folder / "candidates.safetensors") != read_json(folder / "complete.json")["sha256"]:
                    raise ValueError("Candidate commitment changed")
                base.historical(output, plan, condition, index, image)
    write_json(output / "evaluation-barrier.json", {"score_groups_verified": 90,
        "candidate_groups_verified": 180, "private_reference_read": False, "time": time.time()})


def evaluate_all(output):
    plan = verify_plan(output)
    preflight(output, plan)
    prior, cache = read_tensors(output / "prior.safetensors")["lm_prior"], ScoreOrders(output)
    rows = []
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            meta = read_json(output / "scores" / job / "complete.json")
            forbidden = set(meta["forbidden_ids"])
            for image in ("gt", "random"):
                original = base.historical(output, plan, condition, index, image)
                flags = original["ambiguous"]
                ambiguous = set(torch.nonzero(flags).flatten().tolist()) - forbidden
                current = cache.get(condition, index, flags, forbidden)
                wrong = cache.get(condition, (index + 1) % 30, flags, forbidden)
                _, base_ranks = order_and_ranks(original["scores"], flags, forbidden)
                _, prior_ranks = order_and_ranks(prior, flags, forbidden)
                candidates = read_tensors(output / "candidates" / image / job / "candidates.safetensors")
                refs = read_tensors(Path(plan["source"]) / "runs" / job / "captures" / image / "private/text_tokens.safetensors")
                truth = {"question": set(refs["questions"].flatten().tolist()) - forbidden,
                         "answer": set(refs["targets"].flatten().tolist()) - forbidden}
                truth["joint"] = truth["question"] | truth["answer"]
                for method in METHODS:
                    ranks = None
                    if method == "baseline":
                        ranks = base_ranks
                    elif method == "lm_prior":
                        ranks = prior_ranks
                    elif method in POSITIONS:
                        ranks = current[method][1]
                    elif method.startswith("wrong_"):
                        ranks = wrong[method[6:]][1]
                    chosen = set(candidates[method].tolist())
                    rows.append({"condition": condition, "image": image, "index": index, "method": method,
                        "length_bin": plan["length_bins_by_index"][f"{index:02d}"], "selected_count": len(chosen),
                        **vector_metrics(ranks, chosen, truth, ambiguous)})
            write_json(output / "evaluation-progress.json", {"condition": condition, "index": index, "rows": len(rows)})
    fields = tuple(f"{part}_{metric}" for part in ("question", "answer", "joint")
                   for metric in ("recall", "ap", "full_coverage")) + ("selected_count",)
    overall = base.aggregate_metrics(rows, ("condition", "image", "method"), fields)
    by_length = base.aggregate_metrics(rows, ("condition", "image", "method", "length_bin"), fields)
    if len(rows) != 35100 or len(overall) != 1170 or len(by_length) != 3510:
        raise ValueError("Incomplete all-layer evaluation matrix")
    for row in overall + by_length:
        row.update(method_info(row["method"]))
    ranking = layer_ranking(overall)
    chosen = [next(r for r in ranking if r["module"] == module) for module in ("down", "o")]
    buckets = defaultdict(dict)
    for row in rows:
        buckets[row["condition"], row["image"], row["method"]][row["index"]] = row
    paired = []
    for condition in base.CONDITIONS:
        for image in ("gt", "random"):
            for position in POSITIONS:
                current, anchor = buckets[condition, image, "mix_" + position], buckets[condition, image, "mix_l32_down"]
                differences = [current[i]["answer_recall"] - anchor[i]["answer_recall"] for i in range(30)]
                paired.append({"condition": condition, "image": image, "position": position,
                    "answer_recall_delta": sum(differences) / 30,
                    "recall_wins": sum(v > 1e-12 for v in differences),
                    "recall_losses": sum(v < -1e-12 for v in differences),
                    "answer_full_gains": sum(current[i]["answer_full_coverage"] and not anchor[i]["answer_full_coverage"] for i in range(30)),
                    "answer_full_losses": sum(not current[i]["answer_full_coverage"] and anchor[i]["answer_full_coverage"] for i in range(30))})
    write_json(output / "summary.json", {"status": "completed", "rows": rows, "aggregate": overall,
        "by_length": by_length, "ranking": ranking, "development_candidates": chosen,
        "paired_changes": paired, "all_scores_before_truth": True, "independent_validation": False,
        "uses_private_reference": False, "contains_private_tokens": False, "source_sha256": plan["source_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}.json"
    if state.exists():
        raise FileExistsError("No implicit retry")
    write_json(state, {"status": "running"})
    try:
        if args.phase == "prepare":
            if args.source is None or args.previous is None:
                raise ValueError("Original and historical artifacts required")
            prepare(args.output, args.source, args.previous)
        elif args.phase == "score":
            score_all(args.output)
        else:
            evaluate_all(args.output)
        write_json(state, {"status": "completed"})
    except BaseException:
        write_json(state, {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
