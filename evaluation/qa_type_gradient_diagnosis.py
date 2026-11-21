"""Post-hoc public-only spectrum and QA score-shift audit; no target retuning."""

import argparse
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from attacks.analytic.dager_subspace import residual
from core.artifacts import file_hash, read_json, read_tensors, write_json
from evaluation.image_property_scoring import classification_stats
from evaluation.multilayer_property_gpu import CLASSES


def spectrum(value, supervised):
    _, singular, vh = torch.linalg.svd(value.double(), full_matrices=False)
    energy = singular.square() / singular.square().sum()
    ranks = {"original": int((singular > max(1e-10, float(singular[0]) * 1e-6)).sum()),
             "relative_0p01": int((singular > float(singular[0]) * .01).sum()),
             "energy_0p9999": int(torch.searchsorted(energy.cumsum(0), .9999).item()) + 1,
             "public_supervised_rank": supervised}
    if max(ranks.values()) > len(singular):
        raise ValueError("Requested rank exceeds the uploaded matrix rank budget")
    summary = {"original_rank": ranks["original"], "relative_0p01_rank": ranks["relative_0p01"],
               "energy_0p9999_rank": ranks["energy_0p9999"],
               "energy_0p999_rank": int(torch.searchsorted(energy.cumsum(0), .999).item()) + 1,
               "top1_energy": float(energy[0]), "tail_energy_after_supervised": float(energy[supervised:].sum()),
               "norm": float(singular.norm())}
    return summary, {name: vh[:rank] for name, rank in ranks.items()}


def summarize(values):
    return {key: {"min": min(row[key] for row in values), "mean": float(np.mean([row[key] for row in values])),
                  "max": max(row[key] for row in values)} for key in values[0]}


def run(root):
    from transformers import AutoTokenizer
    out = root / "multi-qa/type-diagnosis"
    if (out / "started.json").exists():
        raise FileExistsError("No automatic retry or overwrite of the diagnostic")
    write_json(out / "started.json", {"source_sha256": file_hash(Path(__file__)), "mode": "CPU public artifacts only"})
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        plan = read_json(root / "multi-qa/plan.json")
        commit = read_json(root / "multi-qa/public-scores-complete.json")
        if file_hash(root / "multi-qa/public-scores.json") != commit["sha256"]:
            raise ValueError("Original public scores changed")
        for path, sha in commit["gradient_hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Saved gradient changed")
        bank = read_json(plan["public_bank"])
        model = read_json(Path(plan["model"]) / "model.json")["model"]
        tokenizer = AutoTokenizer.from_pretrained(model["name"], revision=model["revision"], local_files_only=True)
        qa = []
        for row in plan["qa"]:
            answer_ids = tokenizer.encode(row["a"], add_special_tokens=False)
            qa.append({**row, "question_content_tokens": len(tokenizer.encode(row["q"], add_special_tokens=False)),
                       "answer_content_tokens": len(answer_ids), "supervised_tokens": len(answer_ids) + 1})
        prototype_paths = {"head": Path(plan["head_prototypes"]), "abdomen": root / "public/abdomen-prototypes.safetensors"}
        proto = {name: read_tensors(path) for name, path in prototype_paths.items()}
        rows, spectral, score_error = [], {}, 0.
        original = {(row["image"], row["qa"]): row["scores"] for row in read_json(root / "multi-qa/public-scores.json")}
        for item in qa:
            spectral[item["id"]] = {}
            for layer in (24, 32):
                values = []
                for image, public in enumerate(bank):
                    data = read_tensors(root / "multi-qa/gradients" / f"{image:03d}-{item['id']}.safetensors")
                    summary, bases = spectrum(data[f"gradient_{layer}"], item["supervised_tokens"])
                    values.append(summary)
                    scores = {prop: {name: residual(proto[prop][f"visual_{layer}"].double(), basis).norm(dim=1).tolist()
                              for name, basis in bases.items()} for prop in public["labels"]}
                    for prop in scores:
                        score_error = max(score_error, float(np.max(np.abs(np.asarray(scores[prop]["original"]) -
                                                      original[image, item["id"]][prop][str(layer)]))))
                    rows.append({"image": image, "qa": item["id"], "layer": layer, "spectrum": summary, "scores": scores})
                spectral[item["id"]][str(layer)] = summarize(values)
            print("QA spectrum", item["id"], flush=True)
        if score_error > 1e-12:
            raise ValueError("Original distance replay differs")
        write_json(out / "public-diagnostics.json", rows)
        predictions_hash = file_hash(out / "public-diagnostics.json")
        # Only public labels are joined below; no private target files are read.
        groups = {}
        for prop, classes in CLASSES.items():
            groups[prop] = {}
            for layer in (24, 32):
                groups[prop][str(layer)] = {}
                for item in qa:
                    selected = [row for row in rows if row["layer"] == layer and row["qa"] == item["id"]
                                and prop in row["scores"] and bank[row["image"]]["calibration_split"] == "audit"]
                    labels = [bank[row["image"]]["labels"][prop] for row in selected]
                    methods = {}
                    for method in selected[0]["scores"][prop]:
                        scores = [row["scores"][prop][method] for row in selected]
                        margin = np.asarray([s[1] - s[0] for s in scores])
                        positive = np.asarray(labels) == classes[0]
                        methods[method] = {**classification_stats(labels, scores, classes),
                            "positive_margin_mean": float(margin[positive].mean()),
                            "negative_margin_mean": float(margin[~positive].mean()),
                            "positive_margin_min": float(margin[positive].min()),
                            "positive_margin_max": float(margin[positive].max()),
                            "negative_margin_min": float(margin[~positive].min()),
                            "negative_margin_max": float(margin[~positive].max()),
                            "class_mean_gap": float(margin[positive].mean() - margin[~positive].mean())}
                    groups[prop][str(layer)][item["id"]] = methods
        shift = {}
        for prop, layers in groups.items():
            shift[prop] = {}
            for layer, by_qa in layers.items():
                base = by_qa["description"]["original"]
                shift[prop][layer] = {name: {
                    "positive_mean_change": row["original"]["positive_margin_mean"] - base["positive_margin_mean"],
                    "negative_mean_change": row["original"]["negative_margin_mean"] - base["negative_margin_mean"],
                    "class_gap_change": row["original"]["class_mean_gap"] - base["class_mean_gap"]}
                    for name, row in by_qa.items()}
        report = {"scope": "Post-hoc public-only diagnostic; all methods are exploratory and are not applied to medical targets",
                  "public_images": len(bank), "qa": qa, "layers": [24, 32], "spectral_summary": spectral,
                  "public_audit_metrics": groups, "mean_shift_vs_description": shift,
                  "methods": {"original": "rtol=1e-6, atol=1e-10, identical existing scorer",
                              "relative_0p01": "Keep singular values greater than 1% of largest",
                              "energy_0p9999": "Smallest leading basis preserving 99.99% gradient energy",
                              "public_supervised_rank": "First K directions where K is true public answer token count including EOS; oracle length diagnostic only, unavailable in private condition"},
                  "original_distance_max_error": score_error, "uses_private_reference": False,
                  "source_sha256": file_hash(Path(__file__)), "public_diagnostics_sha256": predictions_hash,
                  "parent_scores_sha256": commit["sha256"], "prototype_hashes": {name: file_hash(path) for name, path in prototype_paths.items()}}
        write_json(out / "report.json", report)
        write_json("docs/experiments/data/visual-concept-probe/qa-type-diagnosis.json", report)
        source_copy = out / "analysis-source.py"
        source_copy.write_bytes(Path(__file__).read_bytes())
        write_json("docs/experiments/data/visual-concept-probe/qa-type-manifest.json", {
            "uncommitted": True, "uses_private_reference": False,
            "artifacts": {str(path): file_hash(path) for path in [Path("docs/experiments/data/visual-concept-probe/qa-type-diagnosis.json"), source_copy]},
            "gradient_commit_sha256": file_hash(root / "multi-qa/public-scores-complete.json")})
        write_json(out / "complete.json", {"status": "complete", "seconds": time.monotonic() - start,
                   "report_sha256": file_hash(out / "report.json"), "new_gpu_runs": 0})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
