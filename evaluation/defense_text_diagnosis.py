"""Post-hoc defense audit of committed text results; no new attack or oracle selection.

The candidate-cap sweep scores the already frozen full-vocabulary distances.
Private token IDs are used only for evaluation and are never exported.
"""
import argparse
import ast
from collections import defaultdict
import math
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_file
import yaml

from core.artifacts import file_hash, read_json, write_json
from core.config import DAGEROptions


CONDITIONS = ("fedsgd-b1", "clipping", "noise", "sparsify")
CAPS = (100, 200, 500, 1000, 5000)
REFERENCE_RUNS = (
    "dager_gaussian_noise_Llama3-Lora_enron_email_sgd_2026-04-16_23-36-17",
    "dager_gradient_compression_Llama3-Lora_enron_email_sgd_2026-04-16_23-47-08",
    "recfedllm_gaussian_noise_Llama3-Lora_enron_email_sgd_2026-04-16_23-59-32",
    "recfedllm_gradient_compression_Llama3-Lora_enron_email_sgd_2026-04-17_00-08-52",
)


def audit(source, reference, output):
    if output.exists():
        raise FileExistsError("Audit outputs are immutable; use a new output")
    plan = read_json(source / "plan.json")
    summary = read_json(source / "summary.json")
    jobs = [j for j in plan["jobs"] if j["condition"] in CONDITIONS]
    # Validate every intended result before opening any reference tokens.
    for job in jobs:
        for image in ("gt", "random"):
            result = read_json(source / "runs" / job["id"] / "attacks" / image / "result.json")
            if result["status"] != "completed":
                raise ValueError("Diagnosis requires all original reconstructions committed")
    spectra, curves = defaultdict(list), defaultdict(list)
    legacy_cap_exact = 0
    for job in jobs:
        run = source / "runs" / job["id"]
        for image in ("gt", "random"):
            directory = run / "attacks" / image
            result = read_json(directory / "result.json")
            options = DAGEROptions(**result["provenance"]["options"])
            for layer, span in enumerate(result["provenance"]["subspaces"]):
                # Each nonzero Q/K/V block is unit-normalized before projection.
                m = sum(v > 0 for v in span["block_norms"])
                norm_ratio = math.sqrt(sum(v * v for v in span["singular_values"]) / m)
                spectra[(job["condition"], image, layer)].append({
                    "normalized_residual_norm_ratio": norm_ratio,
                    "public_rank": span["public_rank"], "rank": span["rank"],
                    "first_singular": span["singular_values"][0]})
            tensors = load_file(str(directory / "token_candidates.safetensors"))
            forbidden = set(read_json(directory / "token_candidates.json")["forbidden_ids"])
            refs = load_file(str(run / "captures" / image / "private/text_tokens.safetensors"))
            truth = set(refs["questions"].flatten().tolist() + refs["targets"].flatten().tolist()) - forbidden
            allowed = torch.ones_like(tensors["ambiguous"], dtype=torch.bool)
            allowed[list(forbidden)] = False
            ambiguous = tensors["ambiguous"].bool()
            uncertain = torch.nonzero(allowed & ambiguous).flatten().tolist()
            informative = torch.nonzero(allowed & ~ambiguous).flatten()
            scores = tensors["scores"]
            if not torch.isfinite(scores[informative]).all():
                raise ValueError("Nonfinite saved candidate scores")
            # informative IDs start ascending. Stable sorting therefore keeps
            # the exact original (score, token_id) ordering, including ties.
            ranked = informative[torch.argsort(scores[informative], stable=True)]
            if options.token_selection == "threshold":
                ranked = ranked[scores[ranked] <= options.token_threshold]
            ranked = ranked.tolist()
            for cap in CAPS:
                candidates = ranked[:cap] + uncertain
                if cap == 100:
                    if set(candidates) != set(tensors["selected_ids"].tolist()):
                        raise ValueError("Original top-100 candidate set was not reproduced")
                    legacy_cap_exact += 1
                curves[(job["condition"], image, cap)].append({
                    "recall": len(truth & set(candidates)) / len(truth),
                    "full_coverage": truth <= set(candidates), "size": len(candidates)})
    spectrum_rows = [{"condition": c, "image": image, "layer": layer, "n": len(values),
        "residual_norm_ratio_mean": statistics.fmean(v["normalized_residual_norm_ratio"] for v in values),
        "residual_norm_ratio_median": statistics.median(v["normalized_residual_norm_ratio"] for v in values),
        "public_rank_mean": statistics.fmean(v["public_rank"] for v in values),
        "rank_min": min(v["rank"] for v in values), "rank_max": max(v["rank"] for v in values)}
        for (c, image, layer), values in spectra.items()]
    curve_rows = [{"condition": c, "image": image, "cap": cap, "n": len(values),
        "candidate_recall": statistics.fmean(v["recall"] for v in values),
        "full_coverage_count": sum(v["full_coverage"] for v in values),
        "candidate_size_mean": statistics.fmean(v["size"] for v in values)}
        for (c, image, cap), values in curves.items()]
    reference_rows = []
    for name in REFERENCE_RUNS:
        root = reference / "outputs" / name
        config = yaml.safe_load((root / ".hydra/config.yaml").read_text())
        raw = read_json(root / "avg_result.json")
        rouge = read_json(root / "rouge_result.json")
        if raw["skipped_oom_count"] or len(set(raw["run_id_list"])) != raw["successful_run_count"]:
            raise ValueError("Reference run is incomplete or has duplicate IDs")
        if raw["llm_match_metrics"] != rouge["overall_metrics"]:
            raise ValueError("Reference entity aggregate files disagree")
        logs = [ast.literal_eval(line.split("Defense metrics: ", 1)[1])
                for line in (root / "enron_example.log").read_text().splitlines()
                if "Defense metrics: " in line]
        reference_rows.append({"source_directory": str(root), "n": raw["successful_run_count"],
            "attack": config["attack"]["name"], "defense": config["defense"],
            "precision": config["model"]["precision"], "batch": config["data"]["batch_size"],
            "max_candidates": config["attack"]["maxB_factor"],
            "n_layers": config["attack"]["n_layers"], "add_l1_tokens": config["attack"]["add_l1_tokens"],
            "full_text_rouge1": raw["avg_roug1"], "full_text_rouge2": raw["avg_roug2"],
            "full_text_rougeL": raw["avg_rougl"],
            "pii_micro_recall": raw["llm_match_metrics"]["micro_recall"],
            "pii_macro_recall": raw["llm_match_metrics"]["macro_recall"],
            "log_mean_norm_ratio": statistics.fmean(v["norm_ratio"] for v in logs),
            "log_mean_cosine": statistics.fmean(v["avg_cosine_similarity"] for v in logs),
            "cosine_note": "reference logs average per-tensor cosine; not global relative L2",
            "source_sha256": {n: file_hash(root / n) for n in
                [".hydra/config.yaml", "avg_result.json", "rouge_result.json"]}})
    write_json(output, {
        "status": "completed", "method": "posthoc_existing_scores_and_singular_spectra",
        "contains_private_tokens": False, "uses_private_reference": True,
        "private_reference_note": "仅在原重构全部提交后评价固定 K 的 token 覆盖；不读取原文、不运行或选择新重构",
        "new_gpu_runs": 0, "original_candidate_sets_reproduced": legacy_cap_exact,
        "source_summary_sha256": file_hash(source / "summary.json"),
        "script_sha256": file_hash(Path(__file__)),
        "vlm_aggregate": [r for r in summary["aggregate"] if r["condition"] in CONDITIONS],
        "residual_spectra": spectrum_rows, "candidate_cap_curves": curve_rows,
        "reference_aggregates": reference_rows,
        "limitations": ["Reference model, dataset, precision, candidate cap and metric differ",
                        "Residual norm ratio is an RMS of unit-normalized blocks, not raw upload energy",
                        "No measured post-projection noise-to-signal ratio or new sequence recovery",
                        "Current reference code is not an immutable snapshot of the April runs"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.source, args.reference, args.output)


if __name__ == "__main__":
    main()
