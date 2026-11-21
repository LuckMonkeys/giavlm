"""Controlled public cross-style development; query images never reach recovery."""
import argparse
from dataclasses import replace
import gc
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from core.fl import restore_model, save_observation
from core.types import Observation
from evaluation.cross_style_geometry import score_cross_style
from evaluation.image_property_scoring import classification_stats
from evaluation.property_geometry import score_geometry
from evaluation.visual_concept_probe import inject_logits, native_hard_loss
from evaluation.visual_token_recovery import parameter_grads


def controlled_training(training):
    """Declare the gradient capture independently of the snapshot's trainer."""
    from core.config import validate_training_config
    result = replace(training, algorithm="fedsgd", local_optimizer="sgd", weight_decay=0.,
                     knowledge="text_known", batch_size=1, local_steps=1,
                     gradient_accumulation_steps=1, token_lengths_known=False,
                     upload_parameters=[])
    validate_training_config(result)
    return result


def probe_predictions(root, rows, fit_labels, predictions):
    """Same-coordinate q-proj input prototypes and uploaded LoRA-A row spaces."""
    from attacks.analytic.dager_subspace import row_basis, residual
    out = root / "styles"
    layers = read_json(out / "plan.json")["layers"]
    output = {}
    for style, fold in predictions.items():
        fit, query = fold["fit_indices"], fold["query_indices"]
        scores, diagnostics = {}, {}
        for layer in layers:
            values = {}
            for i in fit + query:
                data = read_tensors(out / "features" / f"{i:03d}.safetensors")
                record = read_json(out / "predictions" / f"{i:03d}.json")
                begin, end = record["visual_token_start"], record["visual_token_stop"]
                values[i] = data[f"input_{layer}"][0, begin:end].double().mean(0)
            prototypes = torch.stack([torch.stack([values[i] for i in fit if
                fit_labels[rows[i]["id"]] == category]).mean(0) for category in ["head", "non-head"]])
            prototypes = torch.nn.functional.normalize(prototypes, dim=1)
            activation, gradients = [], []
            for i in query:
                activation.append((1 - torch.nn.functional.normalize(values[i], dim=0) @ prototypes.T).tolist())
                gradient = read_tensors(out / "features" / f"{i:03d}.safetensors")[f"gradient_{layer}"].double()
                basis, _ = row_basis(gradient, 1e-6, 1e-10)
                gradients.append(residual(prototypes, basis).norm(dim=1).tolist())
            scores[f"activation_{layer}"] = activation
            scores[f"gradient_{layer}"] = gradients
            diagnostics[str(layer)] = {"prototype_cosine": float((prototypes[0] * prototypes[1]).sum()),
                                      "centering": "none; never opposite signed versions of one delta"}
        output[style] = {"query_indices": query, "scores": scores, "diagnostics": diagnostics}
    path = out / "probe_predictions.json"
    write_json(path, output)
    write_json(out / "probe_predictions_complete.json", {"sha256": file_hash(path),
               "scope": "Exploratory q-proj LoRA-A distances; does not apply LM head to parameter gradients"})
    return output


def check_source(root, stage):
    plan = read_json(root / f"{stage}/plan.json")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly the selected GPU")
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError(f"Frozen file changed: {path}")
    return plan


def collect(root):
    out = root / "styles"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or retries")
    write_json(out / "started.json", {"pid": os.getpid()})
    start = time.monotonic()
    try:
        plan = check_source(root, "styles")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.cuda.mem_get_info()[0] < 40 * 2**30:
            raise RuntimeError("Need 40 GiB free; no changed conditions")
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        fingerprint = adapter.fingerprint()
        training = controlled_training(adapter.training_spec)
        adapter.set_training_spec(training)
        q = adapter.encode([plan["training_question"]], adapter.spec.question_length)
        a = adapter.encode([plan["training_answer"]], adapter.spec.target_length)
        read_q = adapter.encode([plan["readout_question"]], adapter.spec.question_length)
        yes, no = [adapter.encode([v], adapter.spec.target_length) for v in ["yes", "no"]]
        yes_id, no_id = int(yes[0, 0]), int(no[0, 0])
        lm = adapter.backend.language_model
        if len(lm.model.layers) != 32:
            raise ValueError("Frozen layer grid requires exactly 32 blocks")
        gradients, predictions, hashes = [], [], {}
        names = list(adapter.trainable())
        # Model calls use no labels, only opaque ids, image paths and public split metadata.
        for index, item in enumerate(read_json(out / "public_manifest.json")):
            if index < plan.get("reuse_prefix", 0):
                # Preserve previously committed public activations and readouts.
                # Only full gradients (not persisted by attempt 1) are replayed.
                visual = read_tensors(out / "visual" / f"{index:03d}.safetensors")["visual"].to(adapter.device)
                grads = parameter_grads(adapter, visual, q, a)
                cpu = {name: grad.detach().cpu() for name, grad in zip(names, grads, strict=True)}
                gradients.append(cpu)
                for folder, suffix in [("visual", "safetensors"), ("features", "safetensors"),
                                       ("predictions", "json")]:
                    p = out / folder / f"{index:03d}.{suffix}"
                    hashes[str(p)] = file_hash(p)
                if index == 0:
                    old = read_tensors(out / "features/000.safetensors")
                    for layer in plan["layers"]:
                        name = f"backend.language_model.model.layers.{layer - 1}.self_attn.q_proj.lora_A.default.weight"
                        if not torch.equal(old[f"gradient_{layer}"], cpu[name]):
                            raise ValueError("Replayed public gradient differs from preserved attempt")
                    write_json(out / "first_checkpoint.json", {"reused_public_activations": True,
                               "first_five_gradient_blocks_exact": True,
                               "source": plan["reuse_manifest"], "gradient_dtype": str(grads[0].dtype)})
                write_json(out / "progress.json", {"completed": index + 1, "total": 108,
                           "phase": "replay-preserved-public-tokens", "seconds": time.monotonic() - start})
                print(f"Public gradient replay {index + 1}/96; activations preserved", flush=True)
                del grads, cpu, visual
                continue
            image = adapter.prepare_image(Image.open(item["file"]).convert("RGB")).unsqueeze(0).to(adapter.device, adapter.dtype)
            with torch.no_grad():
                visual = adapter.visual_embeddings(image)[0].detach()
                values = {}
                handles = []
                for layer in plan["layers"]:
                    def hook(module, args, key=layer, collected=values):
                        collected[key] = args[0].detach().cpu()
                    handles.append(lm.model.layers[layer - 1].self_attn.q_proj.register_forward_pre_hook(hook))
                try:
                    pieces = __import__("core.adapters.llava", fromlist=["llava_input_pieces"]).llava_input_pieces(
                        adapter, visual, adapter.embedding()(read_q), adapter.embedding()(yes[:, :-1]))
                    output = lm(inputs_embeds=torch.cat(pieces, 1), use_cache=False,
                                output_hidden_states=True, return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()
                begin, end = len(pieces[0][0]), len(pieces[0][0]) + visual.shape[1]
                # HF's final hidden_states entry is already final-normalized. Reapply
                # final norm to every selected block output via hooks below instead.
                layer_scores = {}
                for layer in plan["layers"]:
                    h = output.hidden_states[layer]
                    if layer != 32:
                        h = lm.model.norm(h)
                    first = h[:, -yes.shape[1]]
                    pooled = h[:, begin:end].mean(1)
                    layer_scores[str(layer)] = {}
                    for position, representation in [("answer", first), ("visual_mean", pooled)]:
                        logits = lm.lm_head(representation).float()
                        layer_scores[str(layer)][position] = [float(-logits[0, yes_id]), float(-logits[0, no_id])]
                readout = [float(native_hard_loss(inject_logits(adapter, visual, read_q, y), y, adapter.pad))
                           for y in [yes, no]]
                zero = [float(native_hard_loss(inject_logits(adapter, torch.zeros_like(visual), read_q, y), y, adapter.pad))
                        for y in [yes, no]]
                del output
            grads = parameter_grads(adapter, visual, q, a)
            cpu = {name: grad.detach().cpu() for name, grad in zip(names, grads, strict=True)}
            # Exact native update consistency check on first controlled sample.
            if index == 0:
                native = torch.autograd.grad(adapter(image, q, a), tuple(adapter.trainable().values()))
                err = sum(float((cpu[n] - g.detach().float().cpu()).double().square().sum())
                          for n, g in zip(names, native, strict=True))
                if err != 0:
                    raise ValueError("Controlled upload differs from native image update")
                del native
            visual_path = out / "visual" / f"{index:03d}.safetensors"
            write_tensors(visual_path, {"visual": visual.cpu()})
            features = {}
            for layer, features_at_layer in values.items():
                features[f"input_{layer}"] = features_at_layer
                key = f"backend.language_model.model.layers.{layer - 1}.self_attn.q_proj.lora_A.default.weight"
                features[f"gradient_{layer}"] = cpu[key]
            feature_path = out / "features" / f"{index:03d}.safetensors"
            write_tensors(feature_path, features)
            if item["split"] == "query":
                obs = Observation(model=replace(adapter.spec), training=training, tensors=cpu,
                                  model_fingerprint=fingerprint,
                                  public_questions=[plan["training_question"]], public_targets=[plan["training_answer"]],
                                  public_question_ids=q.cpu().tolist(), public_target_ids=a.cpu().tolist())
                obs.validate()
                save_observation(out / "observations" / f"query-{index - 96:02d}", obs)
            else:
                gradients.append(cpu)
            row = {"index": index, "layers": layer_scores, "readout": readout, "zero_readout": zero,
                   "visual_token_start": begin, "visual_token_stop": end,
                   "layer_inputs": {str(k): list(v.shape) for k, v in values.items()}}
            prediction = out / "predictions" / f"{index:03d}.json"
            write_json(prediction, row)
            predictions.append(row)
            for p in [visual_path, feature_path, prediction]:
                hashes[str(p)] = file_hash(p)
            write_json(out / "progress.json", {"completed": index + 1, "total": 108,
                       "seconds": time.monotonic() - start})
            if index == 0:
                write_json(out / "first_checkpoint.json", {"native_gradient_abs_error": 0,
                           "prediction_sha256": file_hash(prediction), "layers": row["layer_inputs"]})
            print(f"Style collection {index + 1}/108", flush=True)
            del grads, cpu, image, visual, features, values
            gc.collect()
        grams = []
        for name in names:
            x = torch.stack([g.pop(name).flatten().double() for g in gradients]).to("cuda:0")
            grams.append((x @ x.T).cpu())
            del x
        path = out / "grams.safetensors"
        write_tensors(path, {"gram": torch.stack(grams)})
        write_json(out / "grams.json", {"names": names})
        if adapter.fingerprint() != fingerprint:
            raise ValueError("Public model mutated")
        hashes[str(path)] = file_hash(path)
        write_json(out / "complete.json", {"hashes": hashes, "images": 108,
                   "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
                   "model_fingerprint": fingerprint})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def development(root):
    out = root / "styles"
    if (out / "development_complete.json").exists():
        raise FileExistsError("No overwriting public development")
    complete = read_json(out / "complete.json")
    for path, sha in complete["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen collection changed")
    rows = read_json(out / "public_manifest.json")
    fit_labels = read_json(out / "fit_labels.json")
    gram = read_tensors(out / "grams.safetensors")["gram"].numpy()
    folds, predictions = {}, {}
    for held_style in ["photo", "line", "render"]:
        domains = [s for s in ["photo", "line", "render"] if s != held_style]
        query = [i for i, r in enumerate(rows[:96]) if r["style"] == held_style and r["split"] == "dev"]
        fit = []
        # Equal total budget: 16 candidates, four per class per fit style.
        for domain in domains:
            for label in ["head", "non-head"]:
                fit += [i for i, r in enumerate(rows[:96]) if r["style"] == domain and
                        r["split"] == "fit" and fit_labels[r["id"]] == label][:4]
        labels = [0 if fit_labels[rows[i]["id"]] == "head" else 1 for i in fit]
        styles = [rows[i]["style"] for i in fit]
        scores, diag = score_cross_style(gram, fit, query, labels, styles)
        # Two equally budgeted single-domain controls, 16 candidates each.
        for domain in domains:
            single = [i for i, r in enumerate(rows[:96]) if r["style"] == domain and r["split"] == "fit"]
            y = [0 if fit_labels[rows[i]["id"]] == "head" else 1 for i in single]
            old, _ = score_geometry(gram, single, query, y)
            scores[f"single_{domain}"] = old["centered_equal_tensor"]
        predictions[held_style] = {"query_indices": query, "fit_indices": fit,
                                   "scores": {k: [row.tolist() if np.isfinite(row).all() else None for row in v]
                                              for k, v in scores.items()}, "diagnostics": diag}
    prediction_path = out / "development_predictions.json"
    write_json(prediction_path, predictions)
    write_json(out / "predictions_complete.json", {"sha256": file_hash(prediction_path),
               "note": "All cross-style scores fixed before joining development labels"})
    probes = probe_predictions(root, rows, fit_labels, predictions)
    # Labels are loaded only after both score grids are committed.
    evaluation = read_json(out / "evaluation_labels.json")
    for style, prediction in predictions.items():
        y = [evaluation[rows[i]["id"]] for i in prediction["query_indices"]]
        folds[style] = {method: classification_stats(y, scores, ["head", "non-head"])
                        for method, scores in prediction["scores"].items()}
        # Controlled public order is fixed: eight head groups then eight non-head
        # groups. Reindexing reuses identical inference with the exchanged upload.
        mapping = [(i + 8) % 16 for i in range(16)]
        for method in ["equal", "consistency_b0.5", "domain_kall_a0.5"]:
            scores = prediction["scores"][method]
            folds[style][method + "_wrong_gradient"] = classification_stats(
                y, [scores[i] for i in mapping], ["head", "non-head"])
    candidates = [m for m in folds["photo"] if m.startswith(("consistency_", "domain_")) and
                  not m.endswith("wrong_gradient")]
    selection = {}
    for method in candidates:
        baseline = "prototype" if method.startswith("domain_") else "equal"
        gains = [folds[s][method]["balanced_accuracy"] - folds[s][baseline]["balanced_accuracy"] for s in folds]
        aucs = [folds[s][method]["class_auroc"]["head"] - folds[s][baseline]["class_auroc"]["head"] for s in folds]
        selection[method] = {"mean_ba_gain": float(np.mean(gains)), "improved_folds": sum(g > 0 for g in gains),
                             "mean_auc_gain": float(np.mean(aucs)),
                             "passes": bool(sum(g > 0 for g in gains) >= 2 and
                                            np.mean(gains) >= .05 and np.mean(aucs) >= 0)}
    eligible = [m for m in candidates if selection[m]["passes"]]
    chosen = max(eligible, key=lambda m: selection[m]["mean_ba_gain"]) if eligible else None
    probe_results, readout_results = {}, {}
    for style, prediction in probes.items():
        y = [evaluation[rows[i]["id"]] for i in prediction["query_indices"]]
        probe_results[style] = {m: classification_stats(y, s, ["head", "non-head"])
                               for m, s in prediction["scores"].items()}
        ids = prediction["query_indices"]
        scores = {}
        raw = [read_json(out / "predictions" / f"{i:03d}.json") for i in ids]
        for layer in [1, 8, 16, 24, 32]:
            for position in ["answer", "visual_mean"]:
                scores[f"layer_{layer}_{position}"] = [r["layers"][str(layer)][position] for r in raw]
        for method in ["readout", "zero_readout"]:
            scores[method] = [r[method] for r in raw]
        readout_results[style] = {m: classification_stats(y, s, ["head", "non-head"]) for m, s in scores.items()}
    report = {"folds": folds, "selection": selection, "selected": chosen,
              "prototype_probes": probe_results, "layer_readout": readout_results,
              "independent_target_confirmation_authorized_by_gate": chosen is not None,
              "scope": "Three public-style development folds; controlled known QA, not reconstructed natural QA or real target confirmation"}
    write_json(out / "development_report.json", report)
    write_json(out / "development_complete.json", {"report_sha256": file_hash(out / "development_report.json")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "development"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        (collect if args.stage == "collect" else development)(args.root)
    except BaseException:
        write_json(args.root / "styles" / f"{args.stage}_failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
