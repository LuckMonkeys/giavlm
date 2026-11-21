"""Fixed last-MLP LoRA-B answer readout; commit all scores before evaluation."""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from attacks.analytic.dager_adapted import forbidden_token_ids
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation, restore_model
from evaluation.lora_ab_filter_study import aggregate
from evaluation.soft_residual_filter_study import prepare as prepare_uploads, selected_ids, verify_plan


CONDITIONS = ("fedsgd-b1", "noise", "sparsify")
METHODS = ("baseline", "last_b", "lm_prior", "wrong_upload")
MODULE = "backend.language_model.model.layers.31.mlp.down_proj.lora_B.default.weight"
PROMPTS = ("What is shown in the image?", "What type of image is this?",
           "Which body part is shown?", "Is an abnormality visible?", "Describe the image.")
PRIOR_STEPS = 16


@torch.no_grad()
def readout_scores(gradient, head, gamma, chunk=2048):
    """Negative normalized response energy; lower ranks first, no truth input."""
    if gradient.ndim != 2 or head.shape[1] != gradient.shape[0] or gamma.shape != head.shape[1:]:
        raise ValueError("Last-B and output vocabulary dimensions differ")
    if any(x.dtype != torch.float64 for x in (gradient, head, gamma)):
        raise ValueError("Readout analysis requires explicit float64")
    norm = gradient.square().sum()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Zero or nonfinite last-B observation")
    scores = head.new_empty(len(head))
    for start in range(0, len(head), chunk):
        u = head[start:start + chunk] * gamma
        denominator = u.square().sum(-1)
        if (denominator <= 0).any():
            raise ValueError("Zero output token direction")
        scores[start:start + len(u)] = -(u @ gradient).square().sum(-1) / (denominator * norm)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite last-B score")
    return scores


def ranked_ids(scores, ambiguous, forbidden):
    values, flags = scores.cpu().numpy(), ambiguous.cpu().numpy()
    valid = np.ones(len(values), dtype=bool)
    valid[list(forbidden)] = False
    allowed = np.flatnonzero(valid & ~flags)
    return allowed[np.lexsort((allowed, values[allowed]))].tolist()


def mixed_ids(base, supplement, ambiguous, forbidden):
    first = ranked_ids(base, ambiguous, forbidden)[:80]
    chosen = set(first)
    for token in ranked_ids(supplement, ambiguous, forbidden):
        if token not in chosen:
            chosen.add(token)
            if len(chosen) == 100:
                break
    flags = set(torch.nonzero(ambiguous).flatten().tolist()) - forbidden
    if len(chosen) != 100:
        raise ValueError("Insufficient valid supplement vocabulary")
    return chosen | flags


def token_metrics(scores, selected, truth, ambiguous, forbidden):
    if not truth:
        raise ValueError("Empty private content reference")
    informative = truth - (set(torch.nonzero(ambiguous).flatten().tolist()) - forbidden)
    ap = None
    if scores is not None and informative:
        positions = {token: i + 1 for i, token in enumerate(ranked_ids(scores, ambiguous, forbidden))}
        ranks = sorted(positions[t] for t in informative)
        ap = sum((i + 1) / r for i, r in enumerate(ranks)) / len(ranks)
    return {"recall": len(truth & selected) / len(truth), "ap": ap,
            "full_coverage": truth <= selected}


def aggregate_metrics(rows, keys, fields):
    result = aggregate(rows, keys, fields)
    for row in result:
        for part in ("question", "answer", "joint"):
            row[f"{part}_full_coverage_count"] = round(row[f"{part}_full_coverage"] * row["n"])
    return result


@torch.no_grad()
def marginal_prior(adapter):
    """Public text-only greedy trajectories; each prompt receives equal mass."""
    lm, encode = adapter.backend.language_model, adapter.tokenizer.encode
    vocabulary, traces, means = adapter.vocab_size, [], []
    device = adapter.embedding().weight.device
    for prompt in PROMPTS:
        context = (encode("USER: ", add_special_tokens=False) + encode("\n", add_special_tokens=False)
                   + encode(prompt, add_special_tokens=False) + encode(" ASSISTANT: ", add_special_tokens=False))
        ids = torch.tensor([context], device=device)
        accumulated = torch.zeros(vocabulary, device=device, dtype=torch.float64)
        past, generated = None, []
        for _ in range(PRIOR_STEPS):
            output = lm(input_ids=ids, past_key_values=past, use_cache=True, return_dict=True)
            probabilities = output.logits[0, -1].float().softmax(-1)
            if len(probabilities) != vocabulary or not torch.isfinite(probabilities).all():
                raise ValueError("Invalid public prior vocabulary")
            accumulated += probabilities.double()
            token = int(probabilities.argmax())
            generated.append(token)
            if token == adapter.eos:
                break
            past, ids = output.past_key_values, torch.tensor([[token]], device=device)
        means.append(accumulated / len(generated))
        traces.append({"prompt": prompt, "steps": len(generated), "generated_ids": generated})
    return -torch.stack(means).mean(0).cpu(), traces


def prepare(output, source):
    prepare_uploads(output, source)
    plan = read_json(output / "plan.json")
    del plan["alphas"]
    original = read_json(source / "plan.json")
    bins = {j["id"].rsplit("-", 1)[1]: j["length_bins"][0]
            for j in original["jobs"] if j["condition"] == "fedsgd-b1"}
    if any(plan["sources"][f"{c}-{i:02d}:gt"]["update_sha256"] !=
           plan["sources"][f"{c}-{i:02d}:random"]["update_sha256"]
           for c in CONDITIONS for i in range(30)):
        raise ValueError("GT and Random did not receive identical uploads")
    baselines = {}
    for condition in CONDITIONS:
        for index in range(30):
            for image in ("gt", "random"):
                path = source / "runs" / f"{condition}-{index:02d}" / "attacks" / image / "token_candidates.safetensors"
                baselines[f"{condition}-{index:02d}:{image}"] = file_hash(path)
    plan.update(module=MODULE, methods=list(METHODS), prompts=list(PROMPTS), prior_steps=PRIOR_STEPS,
        length_bins_by_index=bins, baselines=baselines, score_groups=90, candidate_groups=180,
        candidate_conditions=720, wrong_upload="same condition, (index + 1) modulo 30",
        fusion="80 baseline informative tokens, 20 disjoint supplement tokens, common ambiguity extras",
        fusion_gate="None answer recall/AP beats prior; at least one defense beats both baselines and wrong-upload in both metrics",
        source_boundary="All readout, prior and candidate files commit before token references",
        image_inputs_to_readout=False, image_inputs_to_prior=False)
    write_json(output / "plan.json", plan)


def historical(output, plan, condition, index, image):
    path = Path(plan["source"]) / "runs" / f"{condition}-{index:02d}" / "attacks" / image / "token_candidates.safetensors"
    if file_hash(path) != plan["baselines"][f"{condition}-{index:02d}:{image}"]:
        raise ValueError("Original candidate artifact changed")
    return read_tensors(path)


@torch.no_grad()
def score_all(output):
    plan = verify_plan(output)
    if plan["module"] != MODULE or plan["prompts"] != list(PROMPTS) or plan["prior_steps"] != PRIOR_STEPS:
        raise ValueError("Prespecified readout changed")
    adapter = restore_model(Path(plan["model"]), "cuda:0")
    if adapter.fingerprint() != plan["model_fingerprint"]:
        raise ValueError("Model does not match captured uploads")
    versions = {n: p._version for n, p in adapter.named_parameters()}
    expected_gradient_shape = dict(adapter.named_parameters())[MODULE].shape
    forbidden = forbidden_token_ids(adapter)
    head = adapter.backend.language_model.lm_head.weight.detach().double()
    gamma = adapter.backend.language_model.model.norm.weight.detach().double()
    prior, traces = marginal_prior(adapter)
    write_tensors(output / "prior.safetensors", {"lm_prior": prior})
    write_json(output / "prior.json", {"prompts": list(PROMPTS), "max_steps": PRIOR_STEPS,
        "traces": traces, "sha256": file_hash(output / "prior.safetensors"), "uses_private_reference": False})
    completed = 0
    for condition in CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            public = Path(plan["source"]) / "runs" / job / "captures/random/public"
            if file_hash(public / "observation.json") != plan["sources"][job + ":random"]["metadata_sha256"]:
                raise ValueError("Observation metadata changed")
            observation = load_observation(public, "cpu")
            if observation.public_images is not None or observation.model_fingerprint != plan["model_fingerprint"]:
                raise ValueError("Readout received undeclared image or mismatched model")
            if MODULE not in observation.tensors:
                raise ValueError("Final down-projection B gradient missing")
            if observation.tensors[MODULE].shape != expected_gradient_shape:
                raise ValueError("Uploaded last-B shape differs from public model")
            started = time.monotonic()
            gradient = observation.tensors[MODULE].to(device=head.device, dtype=torch.float64)
            scores = readout_scores(gradient, head, gamma).cpu()
            directory = output / "scores" / job
            if directory.exists():
                raise FileExistsError("No implicit score retry")
            write_tensors(directory / "scores.safetensors", {"last_b": scores})
            write_json(directory / "complete.json", {"status": "completed", "condition": condition,
                "index": index, "module": MODULE, "forbidden_ids": sorted(forbidden),
                "score_sha256": file_hash(directory / "scores.safetensors"),
                "source_sha256": plan["source_sha256"], "seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gradient_shape": list(gradient.shape), "uses_private_reference": False})
            completed += 1
            write_json(output / "progress.json", {"status": "running", "completed": completed,
                "total": 90, "last_group": job})
            print("committed", job, flush=True)
            del observation, gradient
    for condition in CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            current = read_tensors(output / "scores" / job / "scores.safetensors")["last_b"]
            donor = read_tensors(output / "scores" / f"{condition}-{(index + 1) % 30:02d}" / "scores.safetensors")["last_b"]
            for image in ("gt", "random"):
                base = historical(output, plan, condition, index, image)
                flags = base["ambiguous"]
                values = {"baseline": base["scores"], "last_b": current, "lm_prior": prior, "wrong_upload": donor}
                selected = {m: selected_ids(v, flags, forbidden) for m, v in values.items()}
                if selected["baseline"] != set(base["selected_ids"].tolist()) or {len(v) for v in selected.values()} != {109}:
                    raise ValueError("Original candidate or fixed budget mismatch")
                directory = output / "candidates" / image / job
                write_tensors(directory / "candidates.safetensors", {
                    m: torch.tensor(sorted(v), dtype=torch.long) for m, v in selected.items()})
                write_json(directory / "complete.json", {"status": "completed", "baseline_candidates_exact": True,
                    "sha256": file_hash(directory / "candidates.safetensors"), "count": 109})
    if any(p._version != versions[n] for n, p in adapter.named_parameters()):
        raise ValueError("Scoring modified the public model")
    write_json(output / "all-scores-complete.json", {"status": "completed", "readout_groups": 90,
        "candidate_groups": 180, "candidate_conditions": 720, "private_reference_read": False,
        "initial_model_immutable": True, "time": time.time(), "source_sha256": plan["source_sha256"]})
    write_json(output / "progress.json", {"status": "completed", "completed": 90, "total": 90})


def preflight(output, plan):
    completed = read_json(output / "all-scores-complete.json")
    if completed["status"] != "completed" or completed["candidate_groups"] != 180:
        raise ValueError("Incomplete scoring matrix")
    if file_hash(output / "prior.safetensors") != read_json(output / "prior.json")["sha256"]:
        raise ValueError("Prior commitment changed")
    for condition in CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            folder = output / "scores" / job
            if file_hash(folder / "scores.safetensors") != read_json(folder / "complete.json")["score_sha256"]:
                raise ValueError("Readout commitment changed")
            for image in ("gt", "random"):
                folder = output / "candidates" / image / job
                if file_hash(folder / "candidates.safetensors") != read_json(folder / "complete.json")["sha256"]:
                    raise ValueError("Candidate commitment changed")
                historical(output, plan, condition, index, image)
    write_json(output / "evaluation-barrier.json", {"score_groups": 90, "candidate_groups": 180,
        "time": time.time(), "private_reference_read": False})


def fusion_gate(overall):
    def get(condition, image, method):
        return next(r for r in overall if r["condition"] == condition and r["image"] == image and r["method"] == method)
    fields = ("answer_recall", "answer_ap")
    clean = all(get("fedsgd-b1", image, "last_b")[f] > get("fedsgd-b1", image, "lm_prior")[f]
                for image in ("gt", "random") for f in fields)
    defenses = [c for c in CONDITIONS[1:] if all(get(c, image, "last_b")[f] > get(c, image, m)[f]
        for image in ("gt", "random") for m in ("baseline", "wrong_upload") for f in fields)]
    return {"passed": clean and bool(defenses), "clean_beats_prior": clean,
            "defenses_beating_baselines_and_wrong_upload": defenses}


def evaluate_all(output):
    plan = verify_plan(output)
    preflight(output, plan)
    prior = read_tensors(output / "prior.safetensors")["lm_prior"]
    rows, cached = [], {}
    for condition in CONDITIONS:
        for index in range(30):
            job = f"{condition}-{index:02d}"
            meta = read_json(output / "scores" / job / "complete.json")
            forbidden = set(meta["forbidden_ids"])
            current = read_tensors(output / "scores" / job / "scores.safetensors")["last_b"]
            donor = read_tensors(output / "scores" / f"{condition}-{(index + 1) % 30:02d}" / "scores.safetensors")["last_b"]
            for image in ("gt", "random"):
                base = historical(output, plan, condition, index, image)
                refs = read_tensors(Path(plan["source"]) / "runs" / job / "captures" / image / "private/text_tokens.safetensors")
                truth = {"question": set(refs["questions"].flatten().tolist()) - forbidden,
                         "answer": set(refs["targets"].flatten().tolist()) - forbidden}
                truth["joint"] = truth["question"] | truth["answer"]
                flags = base["ambiguous"]
                cached[condition, index, image] = (base, current, truth, forbidden)
                candidates = read_tensors(output / "candidates" / image / job / "candidates.safetensors")
                scores = {"baseline": base["scores"], "last_b": current, "lm_prior": prior, "wrong_upload": donor}
                for method in METHODS:
                    chosen = set(candidates[method].tolist())
                    entry = {"condition": condition, "image": image, "index": index, "method": method,
                        "length_bin": plan["length_bins_by_index"][f"{index:02d}"], "selected_count": len(chosen)}
                    for field in truth:
                        entry.update({f"{field}_{k}": v for k, v in
                            token_metrics(scores[method], chosen, truth[field], flags, forbidden).items()})
                    rows.append(entry)
    fields = tuple(f"{part}_{metric}" for part in ("question", "answer", "joint")
                   for metric in ("recall", "ap", "full_coverage")) + ("selected_count",)
    overall = aggregate_metrics(rows, ("condition", "image", "method"), fields)
    gate = fusion_gate(overall)
    write_json(output / "fusion-gate.json", {**gate, "prespecified": True, "time": time.time()})
    fusion_rows = []
    if gate["passed"]:
        # Both supplements use the identical predeclared 80/20 rule. Candidate
        # construction accepts only scores, ambiguity and public special IDs.
        for (condition, index, image), (base, current, _, forbidden) in cached.items():
            folder = output / "fusion-candidates" / image / f"{condition}-{index:02d}"
            candidates = {m: mixed_ids(base["scores"], supplement, base["ambiguous"], forbidden)
                          for m, supplement in (("mix_last_b", current), ("mix_lm_prior", prior))}
            write_tensors(folder / "candidates.safetensors", {
                m: torch.tensor(sorted(v), dtype=torch.long) for m, v in candidates.items()})
            write_json(folder / "complete.json", {"sha256": file_hash(folder / "candidates.safetensors")})
        write_json(output / "fusion-barrier.json", {"candidate_groups": 180, "time": time.time(),
            "prespecified_rule": True, "development_gate_already_used_reference_metrics": True})
        for (condition, index, image), (base, _, truth, forbidden) in cached.items():
            candidates = read_tensors(output / "fusion-candidates" / image / f"{condition}-{index:02d}" / "candidates.safetensors")
            for method, values in candidates.items():
                chosen = set(values.tolist())
                entry = {"condition": condition, "image": image, "index": index, "method": method,
                    "length_bin": plan["length_bins_by_index"][f"{index:02d}"], "selected_count": len(chosen)}
                for field in truth:
                    entry.update({f"{field}_{k}": v for k, v in
                        token_metrics(None, chosen, truth[field], base["ambiguous"], forbidden).items()})
                fusion_rows.append(entry)
    if len(rows) != 720 or len(overall) != 24:
        raise ValueError("Incomplete evaluation matrix")
    write_json(output / "summary.json", {"status": "completed", "aggregate": overall,
        "by_length": aggregate_metrics(rows, ("condition", "image", "method", "length_bin"), fields),
        "fusion_aggregate": aggregate_metrics(fusion_rows, ("condition", "image", "method"), fields),
        "rows": rows, "fusion_rows": fusion_rows, "fusion_gate": gate,
        "uses_private_reference": False, "contains_private_tokens": False,
        "all_primary_scores_committed_before_truth": True, "source_sha256": plan["source_sha256"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "score", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    state = args.output / f"state-{args.phase}.json"
    if state.exists():
        raise FileExistsError("No implicit retry")
    write_json(state, {"status": "running"})
    try:
        if args.phase == "prepare":
            if args.source is None:
                raise ValueError("Source is required")
            prepare(args.output, args.source)
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
