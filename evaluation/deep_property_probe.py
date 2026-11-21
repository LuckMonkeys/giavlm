"""Frozen public visual prototypes tested on historical private Observations."""
import argparse
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from attacks.analytic.dager_subspace import residual, row_basis
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import load_observation
from evaluation.image_property_scoring import classification_stats, paired_bootstrap


HISTORICAL = Path("outputs/diagnostics/synthetic_head_n20_gpu5_20261017")


def distance_to_gradient(prototypes, gradient):
    """Unsigned subspace distances require two distinct class prototypes."""
    basis, _ = row_basis(gradient.double().cpu(), 1e-6, 1e-10)
    return residual(prototypes.double().cpu(), basis).norm(dim=1).tolist()


def public_prepare(root):
    out = root / "deep-probe"
    if (out / "plan.json").exists():
        raise FileExistsError("No refitting after historical evaluation")
    development = read_json(root / "styles/development_report.json")
    layers = [1, 8, 16, 24, 32]
    criteria = {}
    for layer in layers:
        stats = [v[f"gradient_{layer}"] for v in development["prototype_probes"].values()]
        criteria[layer] = (float(np.mean([v["balanced_accuracy"] for v in stats])),
                           float(np.mean([v["class_auroc"]["head"] for v in stats])))
    selected = max(layers, key=lambda layer: (*criteria[layer], -layer))
    if criteria[selected][0] < .65:
        raise ValueError("Public gradient prototype prerequisite did not pass")
    rows = read_json(root / "styles/public_manifest.json")
    labels = read_json(root / "styles/fit_labels.json")
    ids = [i for i, row in enumerate(rows) if row["split"] == "fit"]
    y = np.array([labels[rows[i]["id"]] for i in ids])
    null = np.random.default_rng(20261028).permutation(y)
    prototypes, hashes = {}, {}
    for layer in layers:
        values = []
        for i in ids:
            p = root / "styles/features" / f"{i:03d}.safetensors"
            hashes[str(p)] = file_hash(p)
            metadata = read_json(root / "styles/predictions" / f"{i:03d}.json")
            begin, end = metadata["visual_token_start"], metadata["visual_token_stop"]
            values.append(read_tensors(p)[f"input_{layer}"][0, begin:end].double().mean(0))
        values = torch.stack(values)
        for name, assignments in [("visual", y), ("shuffled_labels", null)]:
            centers = torch.stack([values[np.flatnonzero(assignments == c)].mean(0)
                                   for c in ["head", "non-head"]])
            prototypes[f"{name}_{layer}"] = torch.nn.functional.normalize(centers, dim=1)
    write_tensors(out / "public-prototypes.safetensors", prototypes)
    model = read_json(root / "plan.json")["model"]
    write_json(out / "plan.json", {
        "selected_layer": selected, "public_development": {str(k): list(v) for k, v in criteria.items()},
        "public_fit_images": len(ids), "fit_styles": ["photo", "line", "render"],
        "layers": layers, "source_sha256": file_hash(Path(__file__)),
        "model_fingerprint": read_json(Path(model) / "model.json")["fingerprint"],
        "hashes": hashes, "prototype_sha256": file_hash(out / "public-prototypes.safetensors"),
        "scope": "Repeated historical target diagnostic, no independent confirmation; layer selected only on public development",
        "knowledge": "private; no image, QA or reconstructed QA used by scoring",
        "selection": "Maximum mean public-fold gradient BA, then public-fold Head AUROC, then shallower layer; refit all 48 public fit images",
        "negative_composition": "Mostly limbs; no medical auxiliary images or target-domain labels"})


def run(root):
    out = root / "deep-probe"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(out / "started.json", {"pid": os.getpid(), "device": "cpu"})
    start = time.monotonic()
    try:
        torch.set_num_threads(1)
        plan = read_json(out / "plan.json")
        if file_hash(Path(__file__)) != plan["source_sha256"]:
            raise ValueError("Frozen probe source changed")
        if file_hash(out / "public-prototypes.safetensors") != plan["prototype_sha256"]:
            raise ValueError("Public prototypes changed")
        prototypes = read_tensors(out / "public-prototypes.safetensors")
        random_basis = torch.linalg.qr(torch.randn(4096, 32, dtype=torch.float64,
            generator=torch.Generator().manual_seed(20261028)), mode="reduced").Q.T
        predictions, input_hashes = [], {}
        for index in range(20):
            path = HISTORICAL / "captures" / f"run-{index:05d}" / "private"
            observation = load_observation(path, "cpu")
            if (observation.model_fingerprint != plan["model_fingerprint"] or
                    observation.training.knowledge != "private" or observation.public_images is not None or
                    observation.public_questions or observation.public_targets or observation.sample_count != 1):
                raise ValueError("Only matching private Observations are allowed")
            scores = {}
            for layer in plan["layers"]:
                key = f"backend.language_model.model.layers.{layer - 1}.self_attn.q_proj.lora_A.default.weight"
                for variant in ["visual", "shuffled_labels"]:
                    scores[f"{variant}_{layer}"] = distance_to_gradient(
                        prototypes[f"{variant}_{layer}"], observation.tensors[key])
            scores["random_space"] = residual(prototypes[f"visual_{plan['selected_layer']}"], random_basis).norm(dim=1).tolist()
            predictions.append({"index": index, "scores": scores})
            for name in ["observation.json", "update.safetensors"]:
                input_hashes[str(path / name)] = file_hash(path / name)
            write_json(out / "progress.json", {"completed": index + 1, "total": 20,
                       "seconds": time.monotonic() - start})
        p = out / "predictions.json"
        write_json(p, predictions)
        write_json(out / "predictions_complete.json", {"predictions_sha256": file_hash(p),
                   "input_hashes": input_hashes, "selected_layer": plan["selected_layer"],
                   "labels_read": False})
        # Reference join follows the complete score commit; do not expose rows.
        truth = read_json(HISTORICAL / "private/targets.json")
        groups = {"all": list(range(20)), **{m: [i for i, row in enumerate(truth) if row["modality"] == m]
                  for m in ["CT", "MRI", "X-Ray"]},
                  "text_does_not_mention_head": [i for i, row in enumerate(truth) if not row["head_terms_present"]]}
        results = {}
        for group, ids in groups.items():
            if not ids:
                results[group] = {"n": 0}
                continue
            labels = [truth[i]["label"] for i in ids]
            results[group] = {}
            for method in predictions[0]["scores"]:
                scores = [predictions[i]["scores"][method] for i in ids]
                results[group][method] = {**classification_stats(labels, scores, ["head", "non-head"]),
                                         **paired_bootstrap(labels, scores, ["head", "non-head"])}
        report = {"selected_layer": plan["selected_layer"], "public_development": plan["public_development"],
                  "groups": results, "seconds": time.monotonic() - start, "scope": plan["scope"],
                  "uses_target_text": False, "uses_private_reference": "Labels for post-commit evaluation only",
                  "public_fit_images": plan["public_fit_images"],
                  "sources": {str(out / name): file_hash(out / name) for name in ["plan.json", "predictions_complete.json"]}}
        write_json(out / "report.json", report)
        write_json("docs/experiments/data/visual-concept-probe/deep-probe-aggregate.json", report)
        write_json(out / "complete.json", {"report_sha256": file_hash(out / "report.json")})
        print("Selected layer", plan["selected_layer"], results["all"][f"visual_{plan['selected_layer']}"], flush=True)
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        (public_prepare if args.stage == "prepare" else run)(args.root)
    except BaseException:
        write_json(args.root / "deep-probe" / f"{args.stage}_failed.json",
                   {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
