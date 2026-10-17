"""Paired text-prior ablation; references only after attacks commit.

Development reuses the ten fixed observations. Validation freezes one weight
and captures distinct QA on new eval image groups. No reference enters search.
"""
import argparse
from dataclasses import replace
import gc
import math
import os
from pathlib import Path
import statistics
import time

import torch

from core.adapters.llava import LlavaTextView
from core.artifacts import (file_hash, read_json, read_tensors, source_fingerprint,
                            write_json)
from core.config import load_config
from core.data import read_manifest
from core.fl import load_observation, restore_model
from evaluation.dager_rank_study import GPU5, DATA, attack_public, capture_pair
from evaluation.reconstruction import evaluate


RANK_ROOT = Path("outputs/diagnostics/dager_rank_gpu5_20261010_v2")
SOURCE = RANK_ROOT/"rank-32"
MODEL = RANK_ROOT/"models/rank-32/round-0010"
WEIGHTS = (0.0, 0.001, 0.01)


def study_inputs(rank):
    if rank == 8:
        return (Path("outputs/diagnostics/dager_reconstruction_gpu5_tune_n10_k100"),
                Path("outputs/federation/slake_llava_f_l_fedavg/round-0010"))
    if rank in (16, 32):
        return RANK_ROOT/f"rank-{rank}", RANK_ROOT/f"models/rank-{rank}/round-0010"
    raise ValueError("Unsupported study rank")


def study_weights(phase, fixed_weight, development):
    if fixed_weight is not None:
        if not math.isfinite(fixed_weight) or fixed_weight <= 0:
            raise ValueError("Fixed prior weight must be finite and positive")
        return (0.0, fixed_weight)
    if phase == "development":
        return WEIGHTS
    if development is None:
        raise ValueError("Validation needs a fixed weight or a development summary")
    selected = read_json(Path(development)/"summary.json")["selected_weight"]
    if selected not in WEIGHTS[1:]:
        raise ValueError("Development did not select a positive weight")
    return (0.0, selected)


def label(weight):
    return "weight-" + str(weight).replace(".", "p")


def complete_beams(directory):
    """Recover completed-depth beams, not partial next-beam checkpoints."""
    if read_json(directory/"result.json")["status"] != "completed":
        raise ValueError("Beam audit requires a committed completed reconstruction")
    saved = {}
    for path in (directory/"checkpoints").glob("*/state.json"):
        meta = read_json(path)
        state, depth = meta["state"], meta["state"]["depth"]
        if state["expansion_cursor"] or not state["beam"]:
            continue
        if any(len(row["tokens"]) != depth for row in state["beam"]):
            continue
        if depth not in saved or meta["elapsed"] > saved[depth][0]:
            saved[depth] = (meta["elapsed"], state["beam"])
    return {depth: rows for depth, (_, rows) in saved.items()}


def beam_audit(directory, question, answer):
    """Post-commit private-reference audit; outputs only counts and positions."""
    beams = complete_beams(directory)
    reference = question + answer
    if set(beams) != set(range(len(reference)+1)):
        raise ValueError("Missing completed-depth beam; cannot infer first pruning point")
    selected = set(read_tensors(directory/"token_candidates.safetensors")["selected_ids"].tolist())
    survived = [any(row["tokens"] == reference[:depth] for row in beams[depth])
                for depth in range(len(reference)+1)]
    first_drop = next((d for d in range(1, len(survived)) if not survived[d]), None)
    final = beams[len(reference)]
    return {"full_coverage": set(reference) <= selected,
            "candidate_recall": len(set(reference) & selected)/max(1, len(set(reference))),
            "question_length": len(question), "answer_length": len(answer),
            "first_drop_depth": first_drop,
            "first_drop_role": (None if first_drop is None else
                                "question" if first_drop <= len(question) else "answer"),
            "first_drop_token_in_candidates": (None if first_drop is None else
                                                reference[first_drop-1] in selected),
            "survived_depths": sum(survived[1:]),
            "prefix_survival_fraction": sum(survived[1:])/max(1, len(reference)),
            "question_boundary_hit": survived[len(question)],
            "final_question_hit": any(row["tokens"][:len(question)] == question for row in final),
            "final_answer_hit": any(row["tokens"][len(question):] == answer for row in final),
            "final_joint_hit": survived[-1], "final_beam_size": len(final)}


def reference_content(run):
    public = read_json(run/"captures/private/public/observation.json")
    reference = read_tensors(run/"captures/private/private/text_tokens.safetensors")
    return (reference["questions"][0, :public["public_question_lengths"][0]].tolist(),
            reference["targets"][0, :public["public_target_lengths"][0]].tolist())


def audit_existing(output):
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for index in range(10):
        run = SOURCE/f"run-{index:05d}"
        directory = run/"attacks/random_residual"
        if not (directory/"result.json").exists():
            raise ValueError("Source must commit before reading references")
        q, a = reference_content(run)
        rows.append({"run": index, **beam_audit(directory, q, a)})
    write_json(output, {"uses_private_reference": True, "rows": rows,
                       "source": str(SOURCE), "no_model_forward": True})


def compare_baseline(current, old):
    """Check actual token IDs, complete final beam, and filter scores."""
    for filename in ("text_tokens.safetensors", "token_candidates.safetensors"):
        left, right = read_tensors(current/filename), read_tensors(old/filename)
        if left.keys() != right.keys() or any(not torch.equal(left[k], right[k]) for k in left):
            raise ValueError(f"Zero-weight baseline differs: {filename}")
    def final_beam(path):
        pointer = read_json(path/"checkpoint.json")
        return read_json(path/"checkpoints"/pointer["generation"]/"state.json")["state"]["beam"]
    if final_beam(current) != final_beam(old):
        raise ValueError("Zero-weight final beam differs from original search")
    return {"tokens_equal": True, "filter_equal": True, "final_beam_equal": True}


def select_weight(aggregate):
    """Predeclared development rule; no ROUGE-only win triggers validation."""
    def key(row):
        # Equal count-based rates must remain ties (e.g. .1 + .2 versus .3).
        return (round(row["joint_em"], 12),
                round(row["question_em"]+row["answer_em"], 12), -row["weight"])
    winner = max(aggregate, key=key)
    return winner["weight"] if winner["weight"] > 0 else None


def aggregate_rows(rows):
    result = []
    for weight in sorted({row["weight"] for row in rows}):
        selected = [row for row in rows if row["weight"] == weight]
        fields = ("question_em", "answer_em", "joint_em", "question_token_em", "answer_token_em",
                  "question_rouge_l", "answer_rouge_l", "seconds", "prefix_survival_fraction",
                  "final_question_hit", "final_answer_hit", "final_joint_hit", "full_coverage")
        result.append({"weight": weight, "n": len(selected), **{
            field: statistics.fmean(row[field] for row in selected) for field in fields}})
    return result


def evaluate_attacks(output, source, config, weights, count):
    read_json(output/"attacks_complete.json")
    # Validate every artifact before opening any reference for this phase.
    for index in range(count):
        for weight in weights:
            directory = output/f"run-{index:05d}"/"attacks"/label(weight)
            if read_json(directory/"result.json")["status"] != "completed":
                raise ValueError("All attacks must complete before evaluation")
    rows, qa = [], set()
    for index in range(count):
        src = source/f"run-{index:05d}"
        q, a = reference_content(src)
        qa.add((tuple(q), tuple(a)))
        for weight in weights:
            directory = output/f"run-{index:05d}"/"attacks"/label(weight)
            report = evaluate(directory, src/"captures/private/private", config.evaluation)
            write_json(directory/"evaluation.json", report)
            metrics = report["samples"][0]["metrics"]
            audit = beam_audit(directory, q, a)
            predicted = read_tensors(directory/"text_tokens.safetensors")
            rows.append({"run": index, "weight": weight,
                         "question_em": metrics["question_exact_match"],
                         "answer_em": metrics["target_exact_match"],
                         "joint_em": metrics["question_exact_match"]*metrics["target_exact_match"],
                         "question_token_em": predicted["questions"][0, :len(q)].tolist() == q,
                         "answer_token_em": predicted["targets"][0, :len(a)].tolist() == a,
                         "question_rouge_l": metrics["question_rougeL"],
                         "answer_rouge_l": metrics["target_rougeL"],
                         "seconds": report["costs"]["seconds"],
                         "costs": report["costs"], **audit})
    aggregate = aggregate_rows(rows)
    report = {"uses_private_reference": True, "reference_use": "post-commit metrics and beam audit only",
              "rows": rows, "aggregate": aggregate, "unique_qa": len(qa),
              "selected_weight": select_weight(aggregate), "source_sha256": source_fingerprint()}
    write_json(output/"summary.json", report)
    return report


@torch.no_grad()
def diagnose_prior(adapter, source, report, output):
    """Measure oracle-parent LM ranks only AFTER all reconstructions commit."""
    rows = []
    for item in report["rows"]:
        if item["weight"] != 0 or item["first_drop_depth"] is None:
            continue
        src = source/f"run-{item['run']:05d}"
        obs = load_observation(src/"captures/private/public", "cuda:0")
        adapter.set_training_spec(obs.training)
        size = obs.model.image_size
        images = torch.rand(1, 3, size, size, generator=torch.Generator().manual_seed(0))
        view = LlavaTextView(adapter, replace(obs, public_images=images))
        q, a = reference_content(src)
        reference = q+a
        depth = item["first_drop_depth"]-1
        scores, _ = view.text_prior_nll([reference[:depth]], list(range(adapter.vocab_size)))
        value = scores[0, reference[depth]]
        rows.append({"run": item["run"], "depth": depth+1,
                     "role": item["first_drop_role"], "true_token_nll": float(value),
                     "true_token_lm_rank": int((scores[0] < value).sum())+1,
                     "would_survive_top20": int((scores[0] < value).sum()) < 20})
    write_json(output/"oracle_prior_audit.json", {
        "uses_private_reference": True, "not_used_in_search": True, "rows": rows})


def validation_rows(adapter, count):
    """Victim-side fixed split selection, distinct actual QA, not outcome-based."""
    def key(row):
        q = adapter.encode([row["question"]], adapter.spec.question_length)
        a = adapter.encode([row["target"]], adapter.spec.target_length)
        return tuple(q.flatten().tolist()), tuple(a.flatten().tolist())
    excluded = {key(row) for row in read_manifest(DATA, "vqa", "tune", 0, unique_images=True)[:10]}
    selected, seen = [], set(excluded)
    for row in read_manifest(DATA, "vqa", "eval", 0, unique_images=True):
        sequence = key(row)
        if sequence not in seen:
            selected.append(row)
            seen.add(sequence)
        if len(selected) == count:
            return selected
    raise ValueError("Insufficient distinct eval QA for frozen validation")


def final_audit(output):
    """Post-commit diversity and final-ranking audit; never expose references."""
    diversity, cases, sequences, questions, images = [], [], {}, {}, {}
    for phase in ("development", "validation"):
        folder = output/phase
        read_json(folder/"complete.json")
        summary = read_json(folder/"summary.json")
        source = SOURCE if phase == "development" else folder
        sequences[phase], questions[phase], images[phase] = set(), set(), set()
        rounds = set()
        for index in range(10):
            src = source/f"run-{index:05d}"
            q, a = reference_content(src)
            sequences[phase].add((tuple(q), tuple(a)))
            questions[phase].add(tuple(q))
            images[phase].add(file_hash(src/"captures/private/private/images.safetensors"))
            rounds.add(read_json(src/"captures/private/public/observation.json")["training"]["server_round"])
            for weight in read_json(folder/"plan.json")["weights"]:
                if weight == 0:
                    continue
                directory = folder/f"run-{index:05d}"/"attacks"/label(weight)
                rows = complete_beams(directory)[len(q)+len(a)]
                found = next(((rank+1, row) for rank, row in enumerate(rows) if row["tokens"] == q+a), None)
                item = {"phase": phase, "run": index, "weight": weight,
                        "correct_sequence_rank": found[0] if found else None}
                if found:
                    true = found[1]
                    item.update(true_span=true["span_score"], true_nll=true["lm_nll"],
                                winner_span=rows[0]["span_score"], winner_nll=rows[0]["lm_nll"],
                                combined_gap=true["score"]-rows[0]["score"],
                                span_gap=true["span_score"]-rows[0]["span_score"],
                                nll_gap=true["lm_nll"]-rows[0]["lm_nll"])
                cases.append(item)
        diversity.append({"phase": phase, "distinct_qa": len(sequences[phase]),
                          "distinct_questions": len(questions[phase]), "distinct_images": len(images[phase]),
                          "capture_round_counters": sorted(rounds),
                          "model_snapshot_round": read_json(MODEL/"model.json")["training"]["server_round"],
                          "completed_attacks": len(summary["rows"])})
    write_json(output/"final_audit.json", {"uses_private_reference": True,
        "reference_use": "post-commit text and image-hash audit only",
        "diversity": diversity, "cases": cases,
        "qa_overlap": len(sequences["development"] & sequences["validation"]),
        "question_overlap": len(questions["development"] & questions["validation"]),
        "image_overlap": len(images["development"] & images["validation"])})


def run_study(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a fresh study output; no automatic retry")
    started = time.monotonic()
    configured_source, model = study_inputs(args.rank)
    config = load_config(configured_source/"configs/random_residual.json")
    model_training = read_json(model/"model.json")["training"]
    if config.training.lora_rank != args.rank or model_training["lora_rank"] != args.rank:
        raise ValueError("Configuration or model rank differs from the requested rank")
    model_round = model_training["server_round"]
    config = replace(config, training=replace(config.training, server_round=model_round))
    development = args.phase == "development"
    weights = study_weights(args.phase, args.weight, args.development)
    fingerprint = source_fingerprint()
    plan = {"phase": args.phase, "rank": args.rank, "samples": 10, "weights": weights,
            "model_snapshot_round": model_round, "capture_server_round": config.training.server_round,
            "source_sha256": fingerprint, "model_sha256": file_hash(model/"model.safetensors"),
            "model": str(model), "observation_source": str(configured_source) if development else str(output),
            "fixed_weight": args.weight,
            "config_sha256": file_hash(configured_source/"configs/random_residual.json"),
            "data_sha256": file_hash(DATA), "image": "Random", "seed": 0,
            "candidate_k": 100, "beam_width": 16, "prefix_batch_size": 32,
            "max_evaluations": config.attack.max_evaluations, "seconds": config.attack.seconds,
            "physical_gpu": 5, "reference_policy": "All attacks commit before evaluation reads truth",
            "prior": "Victim text LM, USER/newline/question/ASSISTANT/answer; no image/EOS/pad; content NLL",
            "selection_rule": ("User-fixed weight; no outcome-based tuning" if args.weight is not None else
                               "maximize Joint EM, then Question EM + Answer EM, then prefer smaller weight"),
            "validation_policy": "First ten eval client-0 image groups with distinct tokenized QA absent from development",
            "baseline_comparison": "Development weight zero must exactly reproduce saved tokens, scores and final beam"}
    if not development and args.weight is None:
        plan["development_summary_sha256"] = file_hash(Path(args.development)/"summary.json")
    write_json(output/"plan.json", plan)
    try:
        adapter = restore_model(model, "cuda:0")
        model_fingerprint = adapter.fingerprint()
        torch.cuda.reset_peak_memory_stats()
        source = configured_source if development else output
        if not development:
            rows = validation_rows(adapter, 10)
            for index, row in enumerate(rows):
                write_json(output/"progress.json", {"stage": "capture", "run": index})
                capture_pair(adapter, row, replace(config.training, knowledge="image_known"),
                             output/f"run-{index:05d}", model)
                print(f"capture {index+1}/10 committed", flush=True)
            del rows, row
            gc.collect()
        versions = {name: parameter._version for name, parameter in adapter.named_parameters()}
        audits = []
        for index in range(10):
            src = source/f"run-{index:05d}"
            for weight in weights:
                if source_fingerprint() != fingerprint:
                    raise ValueError("Source changed during the frozen experiment")
                options = replace(config.attack.dager, lm_prior_weight=weight)
                protocol = replace(config, attack=replace(config.attack, dager=options))
                write_json(output/"configs"/f"{label(weight)}.json", protocol)
                directory = output/f"run-{index:05d}"/"attacks"/label(weight)
                write_json(output/"progress.json", {"stage": "attack", "run": index, "weight": weight})
                status, costs = attack_public(adapter, protocol, src/"captures/private/public",
                                             directory, model_fingerprint)
                if status != "completed":
                    raise ValueError(f"Attack did not complete: {status}")
                if weight == 0 and development:
                    audits.append({"run": index, **compare_baseline(directory, src/"attacks/random_residual")})
                    write_json(output/"baseline_audit.json", {"rows": audits})
                if weight > 0:
                    zero = output/f"run-{index:05d}"/"attacks"/label(0.0)
                    if file_hash(zero/"token_candidates.safetensors") != file_hash(directory/"token_candidates.safetensors"):
                        raise ValueError("The prior changed the first-stage candidate filter")
                print(f"run {index+1}/10 weight={weight}: {status}; seconds={costs['seconds']:.1f}; "
                      f"prior_parents={costs['lm_prior_prefixes']}", flush=True)
                gc.collect()
        if any(parameter._version != versions[name] for name, parameter in adapter.named_parameters()):
            raise ValueError("Search mutated the victim model")
        write_json(output/"attacks_complete.json", {"attacks": 10*len(weights),
                   "source_sha256": fingerprint, "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20})
        write_json(output/"progress.json", {"stage": "evaluation"})
        report = evaluate_attacks(output, source, config, weights, 10)
        diagnose_prior(adapter, source, report, output)
        if adapter.fingerprint() != model_fingerprint:
            raise ValueError("Model fingerprint changed")
        write_json(output/"complete.json", {"summary_sha256": file_hash(output/"summary.json"),
                   "oracle_audit_sha256": file_hash(output/"oracle_prior_audit.json"),
                   "seconds": time.monotonic()-started})
        write_json(output/"progress.json", {"stage": "completed"})
        print(report["aggregate"], flush=True)
    except Exception as error:
        write_json(output/"failed.json", {"error_type": type(error).__name__, "reason": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("development", "validation", "audit", "final-audit"),
                        default="development")
    parser.add_argument("--development")
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--weight", type=float, help="Freeze a positive weight and compare only against zero")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "audit":
        if args.rank != 32 or args.weight is not None:
            parser.error("Legacy audit phases support only the original rank-32 study")
        audit_existing(Path(args.output))
        return
    if args.phase == "final-audit":
        if args.rank != 32 or args.weight is not None:
            parser.error("Legacy audit phases support only the original rank-32 study")
        final_audit(Path(args.output))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU5 or torch.cuda.device_count() != 1:
        raise ValueError("Expose only authorized GPU5")
    run_study(args)


if __name__ == "__main__":
    main()
