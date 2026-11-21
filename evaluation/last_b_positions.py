"""Three prespecified LoRA-B positions on identical uploaded observations."""
import argparse
from pathlib import Path
import shutil
import statistics
import time
import traceback

import torch

from attacks.analytic.dager_adapted import forbidden_token_ids
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation, restore_model
from evaluation import last_b_token_readout as base
from evaluation.soft_residual_filter_study import selected_ids, verify_plan


POSITIONS = {
    "l32_down": "backend.language_model.model.layers.31.mlp.down_proj.lora_B.default.weight",
    "l32_o": "backend.language_model.model.layers.31.self_attn.o_proj.lora_B.default.weight",
    "l31_down": "backend.language_model.model.layers.30.mlp.down_proj.lora_B.default.weight",
}
ANCHOR = "l32_down"
PRIMARY = ("baseline", "lm_prior", *POSITIONS, *("wrong_" + m for m in POSITIONS))
MIXES = ("mix_lm_prior", *("mix_" + m for m in POSITIONS))


def prepare(output, source, previous):
    base.prepare(output, source)
    plan = read_json(output / "plan.json")
    old = read_json(previous / "plan.json")
    if old["model_fingerprint"] != plan["model_fingerprint"] or old["baselines"] != plan["baselines"]:
        raise ValueError("Previous readout used different model or baseline captures")
    for condition in base.CONDITIONS:
        for index in range(30):
            public = source / "runs" / f"{condition}-{index:02d}" / "captures/random/public/observation.json"
            names = read_json(public)["parameter_names"]
            if any(name not in names for name in POSITIONS.values()):
                raise ValueError("Prespecified B module missing from original observation")
    prior_meta = read_json(previous / "prior.json")
    if file_hash(previous / "prior.safetensors") != prior_meta["sha256"]:
        raise ValueError("Public prior integrity error")
    shutil.copyfile(previous / "prior.safetensors", output / "prior.safetensors")
    references = {}
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            path = previous / "scores" / job / "scores.safetensors"
            references[job] = file_hash(path)
    plan.update(positions=POSITIONS, primary_methods=list(PRIMARY), mix_methods=list(MIXES),
        previous=str(previous.resolve()), previous_scores=references,
        public_prior_sha256=prior_meta["sha256"], score_groups=90, position_scores=270,
        candidate_groups=180, candidate_conditions=2160,
        comparison="Prespecified positions separately; no cross-position fusion or truth-selected layer")
    write_json(output / "plan.json", plan)


def method_scores(output, condition, index, original):
    job = f"{condition}-{index:02d}"
    current = read_tensors(output / "scores" / job / "scores.safetensors")
    donor = read_tensors(output / "scores" / f"{condition}-{(index + 1) % 30:02d}" / "scores.safetensors")
    prior = read_tensors(output / "prior.safetensors")["lm_prior"]
    return {"baseline": original["scores"], "lm_prior": prior,
            **current, **{"wrong_" + k: v for k, v in donor.items()}}


def candidates_for(scores, flags, forbidden):
    values = {m: selected_ids(scores[m], flags, forbidden) for m in PRIMARY}
    for name in ("lm_prior", *POSITIONS):
        values["mix_" + name] = base.mixed_ids(scores["baseline"], scores[name], flags, forbidden)
    if {len(v) for v in values.values()} != {109}:
        raise ValueError("Unequal candidate budgets")
    return values


@torch.no_grad()
def score_all(output):
    plan = verify_plan(output)
    if plan["positions"] != POSITIONS or plan["primary_methods"] != list(PRIMARY):
        raise ValueError("Prespecified positions changed")
    if file_hash(output / "prior.safetensors") != plan["public_prior_sha256"]:
        raise ValueError("Frozen public prior changed")
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Victim fingerprint differs")
    parameters = dict(adapter.named_parameters())
    versions = {n: p._version for n, p in parameters.items()}
    if any(parameters[n].shape != (4096, 32) for n in POSITIONS.values()):
        raise ValueError("Unexpected public B weight shape")
    head = adapter.backend.language_model.lm_head.weight.detach().double()
    gamma = adapter.backend.language_model.model.norm.weight.detach().double()
    forbidden = forbidden_token_ids(adapter)
    completed = 0
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            public = Path(plan["source"]) / "runs" / job / "captures/random/public"
            if file_hash(public / "observation.json") != plan["sources"][job + ":random"]["metadata_sha256"]:
                raise ValueError("Observation metadata changed")
            observation = load_observation(public, "cpu")
            if observation.public_images is not None or observation.model_fingerprint != plan["model_fingerprint"]:
                raise ValueError("Readout input boundary violated")
            started = time.monotonic()
            scores = {}
            for position, name in POSITIONS.items():
                gradient = observation.tensors[name].to(device=head.device, dtype=torch.float64)
                if gradient.shape != parameters[name].shape:
                    raise ValueError("Uploaded B shape differs")
                scores[position] = base.readout_scores(gradient, head, gamma).cpu()
            prior_path = Path(plan["previous"]) / "scores" / job / "scores.safetensors"
            if file_hash(prior_path) != plan["previous_scores"][job]:
                raise ValueError("Historical anchor score changed")
            old = read_tensors(prior_path)["last_b"]
            delta = float((scores[ANCHOR] - old).abs().max())
            if delta > 1e-12:
                raise ValueError(f"Anchor readout changed: {delta}")
            folder = output / "scores" / job
            if folder.exists():
                raise FileExistsError("No implicit score retry")
            write_tensors(folder / "scores.safetensors", scores)
            write_json(folder / "complete.json", {"status": "completed", "vocab_rows": len(head),
                "positions": POSITIONS, "forbidden_ids": sorted(forbidden), "anchor_max_delta": delta,
                "score_sha256": file_hash(folder / "scores.safetensors"), "source_sha256": plan["source_sha256"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "seconds": time.monotonic() - started})
            completed += 1
            write_json(output / "progress.json", {"status": "running", "completed": completed, "total": 90})
            print("committed", job, "anchor delta", delta, flush=True)
            del observation, gradient
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            for image in ("gt", "random"):
                original = base.historical(output, plan, condition, index, image)
                scores = method_scores(output, condition, index, original)
                chosen = candidates_for(scores, original["ambiguous"], forbidden)
                previous = Path(plan["previous"])
                old_candidates = read_tensors(previous / "candidates" / image / job / "candidates.safetensors")
                old_mix = read_tensors(previous / "fusion-candidates" / image / job / "candidates.safetensors")
                for new_name, old_name in [("baseline", "baseline"), ("lm_prior", "lm_prior"),
                                           (ANCHOR, "last_b"), ("wrong_" + ANCHOR, "wrong_upload")]:
                    if chosen[new_name] != set(old_candidates[old_name].tolist()):
                        raise ValueError("Historical candidate reproduction failed")
                for new_name, old_name in [("mix_" + ANCHOR, "mix_last_b"), ("mix_lm_prior", "mix_lm_prior")]:
                    if chosen[new_name] != set(old_mix[old_name].tolist()):
                        raise ValueError("Historical mixture reproduction failed")
                folder = output / "candidates" / image / job
                write_tensors(folder / "candidates.safetensors", {
                    m: torch.tensor(sorted(v), dtype=torch.long) for m, v in chosen.items()})
                write_json(folder / "complete.json", {"sha256": file_hash(folder / "candidates.safetensors"),
                    "historical_candidates_exact": True, "count": 109})
    if any(p._version != versions[n] for n, p in parameters.items()):
        raise ValueError("Readout mutated model")
    write_json(output / "all-scores-complete.json", {"status": "completed", "groups": 90,
        "position_scores": 270, "candidate_groups": 180, "candidate_conditions": 2160,
        "initial_model_immutable": True, "private_reference_read": False, "time": time.time()})
    write_json(output / "progress.json", {"status": "completed", "completed": 90, "total": 90})


def metrics(scores, chosen, truths, flags, forbidden):
    """Sort once per method, with the same AP definition as the prior phase."""
    ambiguous = set(torch.nonzero(flags).flatten().tolist()) - forbidden
    positions = ({token: i + 1 for i, token in enumerate(base.ranked_ids(scores, flags, forbidden))}
                 if scores is not None else None)
    result = {}
    for part, truth in truths.items():
        informative = truth - ambiguous
        ranks = sorted(positions[t] for t in informative) if positions is not None else []
        result.update({f"{part}_recall": len(chosen & truth) / len(truth),
            f"{part}_full_coverage": truth <= chosen,
            f"{part}_ap": statistics.fmean((i + 1) / r for i, r in enumerate(ranks)) if ranks else None})
    return result


def evaluate_all(output):
    plan = verify_plan(output)
    completion = read_json(output / "all-scores-complete.json")
    if completion["status"] != "completed" or completion["candidate_conditions"] != 2160:
        raise ValueError("Incomplete score matrix")
    if file_hash(output / "prior.safetensors") != plan["public_prior_sha256"]:
        raise ValueError("Prior changed")
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            folder = output / "scores" / job
            meta = read_json(folder / "complete.json")
            if file_hash(folder / "scores.safetensors") != meta["score_sha256"] or meta["anchor_max_delta"] > 1e-12:
                raise ValueError("Score commitment changed")
            for image in ("gt", "random"):
                folder = output / "candidates" / image / job
                if file_hash(folder / "candidates.safetensors") != read_json(folder / "complete.json")["sha256"]:
                    raise ValueError("Candidate commitment changed")
                base.historical(output, plan, condition, index, image)
    write_json(output / "evaluation-barrier.json", {"groups": 90, "candidate_groups": 180,
        "private_reference_read": False, "time": time.time()})
    rows = []
    for condition in base.CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            meta = read_json(output / "scores" / job / "complete.json")
            forbidden = set(meta["forbidden_ids"])
            for image in ("gt", "random"):
                original = base.historical(output, plan, condition, index, image)
                scores = method_scores(output, condition, index, original)
                candidates = read_tensors(output / "candidates" / image / job / "candidates.safetensors")
                refs = read_tensors(Path(plan["source"]) / "runs" / job / "captures" / image / "private/text_tokens.safetensors")
                truth = {"question": set(refs["questions"].flatten().tolist()) - forbidden,
                         "answer": set(refs["targets"].flatten().tolist()) - forbidden}
                truth["joint"] = truth["question"] | truth["answer"]
                for method, values in candidates.items():
                    chosen = set(values.tolist())
                    rows.append({"condition": condition, "image": image, "index": index, "method": method,
                        "length_bin": plan["length_bins_by_index"][f"{index:02d}"], "selected_count": len(chosen),
                        **metrics(scores.get(method), chosen, truth, original["ambiguous"], forbidden)})
    fields = tuple(f"{part}_{metric}" for part in ("question", "answer", "joint")
                   for metric in ("recall", "ap", "full_coverage")) + ("selected_count",)
    if len(rows) != 2160:
        raise ValueError("Incomplete candidate evaluation")
    paired = []
    for condition in base.CONDITIONS:
        for image in ("gt", "random"):
            for prefix in ("", "mix_"):
                anchor = {r["index"]: r for r in rows if r["condition"] == condition and r["image"] == image
                          and r["method"] == prefix + ANCHOR}
                for position in ("l32_o", "l31_down"):
                    current = [r for r in rows if r["condition"] == condition and r["image"] == image
                               and r["method"] == prefix + position]
                    differences = [r["answer_recall"] - anchor[r["index"]]["answer_recall"] for r in current]
                    aps = [r["answer_ap"] - anchor[r["index"]]["answer_ap"] for r in current if r["answer_ap"] is not None]
                    paired.append({"condition": condition, "image": image, "method": prefix + position,
                        "anchor": prefix + ANCHOR, "n": len(current), "answer_recall_delta": statistics.fmean(differences),
                        "answer_ap_delta": statistics.fmean(aps) if aps else None,
                        "recall_wins": sum(v > 1e-12 for v in differences),
                        "recall_losses": sum(v < -1e-12 for v in differences),
                        "answer_full_gains": sum(r["answer_full_coverage"] and not anchor[r["index"]]["answer_full_coverage"] for r in current),
                        "answer_full_losses": sum(not r["answer_full_coverage"] and anchor[r["index"]]["answer_full_coverage"] for r in current)})
    write_json(output / "summary.json", {"status": "completed", "rows": rows,
        "aggregate": base.aggregate_metrics(rows, ("condition", "image", "method"), fields),
        "by_length": base.aggregate_metrics(rows, ("condition", "image", "method", "length_bin"), fields),
        "paired_position_changes": paired, "all_scores_before_truth": True,
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
                raise ValueError("Original uploads and previous readout required")
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
