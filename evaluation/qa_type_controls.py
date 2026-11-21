"""Public paired QA interventions with actual Q/K/V hooks and token-loss gradients."""

import argparse
import os
from pathlib import Path
import string
import time
import traceback

from PIL import Image
import torch

from attacks.analytic.dager_subspace import residual
from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from evaluation.public_multiqa_calibration import QA
from evaluation.qa_type_gradient_diagnosis import spectrum


LAYERS = (24, 32)
MODULES = ("q", "k", "v")


def cases():
    questions = {"description": "Describe the image.", "present": "Is there an image to inspect?",
                 "examine": "Can this picture be examined?"}
    answers = {"description": "The image is shown.", "yes": "Yes.", "long_yes": "Yes, an image is shown."}
    old = {(qa["q"], qa["a"]): qa["id"] for qa in QA}
    rows = []
    for qid, question in questions.items():
        for aid, answer in answers.items():
            prior = old.get((question, answer))
            rows.append({"id": prior or f"{qid}_{aid}", "q": question, "a": answer,
                         "question_factor": qid, "answer_factor": aid, "old_qa": prior,
                         "token_decomposition": (qid, aid) in [("description", "description"),
                                                              ("present", "yes"), ("present", "long_yes")]})
    for prior in QA:
        if prior["id"] not in {row["id"] for row in rows}:
            rows.append({**prior, "question_factor": prior["id"], "answer_factor": prior["id"],
                         "old_qa": prior["id"], "token_decomposition": prior["id"] == "blank"})
    return rows


def token_subset(bank):
    """Two public content families per anatomical source, with rotating styles."""
    selected = []
    for prefix in ("head-", "non-head-", "abdomen-"):
        groups = sorted({row["group"] for row in bank if row["group"].startswith(prefix)})[:2]
        if len(groups) != 2:
            raise ValueError("Expected two public families for every token diagnostic source")
        for group in groups:
            style = ("photo", "line", "render")[len(selected) % 3]
            available = [i for i, row in enumerate(bank) if row["group"] == group and row["style"] == style]
            if not available:
                raise ValueError("Public token diagnostic style missing; do not change selection")
            selected.append(available[0])
    return selected


def prepare(root):
    destination = root / "qa-controls-plan-template.json"
    if destination.exists():
        raise FileExistsError("Paired controls already prepared")
    parent = read_json(root / "multi-qa/plan.json")
    bank = [{**row, "old_image": i} for i, row in enumerate(read_json(parent["public_bank"]))
            if row["calibration_split"] == "audit"]
    if len(bank) != 33:
        raise ValueError("Expected 33 already-declared public audit images")
    out = root / "qa-controls"
    write_json(out / "public-bank.json", bank)
    rows = cases()
    from transformers import AutoTokenizer
    model = read_json(Path(parent["model"]) / "model.json")["model"]
    tokenizer = AutoTokenizer.from_pretrained(model["name"], revision=model["revision"], local_files_only=True)
    for row in rows:
        row["question_tokens"] = len(tokenizer.encode(row["q"], add_special_tokens=False))
        row["answer_token_ids"] = tokenizer.encode(row["a"], add_special_tokens=False) + [tokenizer.eos_token_id]
        row["supervised_tokens"] = len(row["answer_token_ids"])
        row["token_text"] = [tokenizer.decode([token]).strip() for token in row["answer_token_ids"]]
    files = [Path(parent["model"]) / name for name in ("model.json", "model.safetensors")]
    files += [Path(parent["head_prototypes"]), root / "public/abdomen-prototypes.safetensors",
              out / "public-bank.json", root / "multi-qa/public-scores.json", root / "multi-qa/public-scores-complete.json"]
    files += [Path(row["file"]) for row in bank]
    files += [root / "multi-qa/gradients" / f"{row['old_image']:03d}-{qa['id']}.safetensors" for row in bank for qa in QA]
    write_json(destination, {"model": parent["model"], "gpu_uuid": None, "layers": list(LAYERS),
        "modules": list(MODULES), "cases": rows, "images": 33,
        "head_prototypes": parent["head_prototypes"], "token_images": token_subset(bank),
        "primary_contrast": ["present", "present_long_yes"],
        "question_contrast": ["description", "present_description", "examine_description"],
        "scope": "Public mechanism diagnostic only; no medical images, QA or uploads accessed",
        "loss": "Original native answer mean NLL including EOS, padding excluded",
        "token_loss": "Same complete forward graph; individual target-position NLL divided by original full supervised count. Diagnostic intervention, not a deployment upload.",
        "full_baseline_policy": "Every old Q gradient must exactly reproduce saved same-image old QA gradients; fail on mismatch",
        "token_additivity_policy": "Report raw mismatch between sum of bf16 token gradients and native full gradient; no post-hoc tolerance/dtype change, no assertion of bitwise additivity",
        "interface_policy": "Within each forward, Q/K/V and their actual LoRA-A branches must receive identical visual-position inputs; fail on mismatch",
        "input_layout": "Original fixed public prefix, 64 question and 32 answer slots; native bf16/eager/eval",
        "interpretation": "Long/short answers change content and length together. Crossed short answer to description prompt is an artificial mechanism condition, not natural VQA performance.",
        "threshold": "Original direct comparison of two public prototype distances, no fitting or model selection",
        "source_sha256": file_hash(Path(__file__)),
        "hashes": {str(path.resolve()): file_hash(path) for path in files}})
    print({"public_images": 33, "cases": len(rows), "native_forwards": 33 * len(rows),
           "token_diagnostic_images": len(token_subset(bank)), "QA": [(row["id"], row["supervised_tokens"]) for row in rows]})


def native_token_losses(logits, targets):
    return -(torch.nn.functional.one_hot(targets, logits.shape[-1]).float() * logits.float().log_softmax(-1)).sum(-1)


def loss_support(gradient, begin, end, supervised):
    value = gradient[0].detach().double().cpu()
    energy = value.square().sum(1)
    total = float(energy.sum())
    active = torch.zeros(len(energy), dtype=torch.bool)
    active[supervised] = True
    visual = torch.zeros(len(energy), dtype=torch.bool)
    visual[begin:end] = True
    return {"total_energy": total, "visual_energy_fraction": float(energy[visual].sum()) / max(total, 1e-60),
            "supervised_energy_fraction": float(energy[active].sum()) / max(total, 1e-60),
            "outside_supervised_energy_fraction": float(energy[~active].sum()) / max(total, 1e-60),
            "nonzero_visual_positions": int((energy[visual] > 0).sum()),
            "nonzero_outside_supervised_positions": int((energy[~active] > 0).sum())}


def collect_one(adapter, visual, case, prototypes, token_diagnostic):
    from evaluation.visual_concept_probe import inject_logits
    q = adapter.encode([case["q"]], adapter.spec.question_length)
    a = adapter.encode([case["a"]], adapter.spec.target_length)
    valid = torch.nonzero(a[0] != adapter.pad, as_tuple=True)[0]
    if a[0, valid].tolist() != case["answer_token_ids"]:
        raise ValueError("Tokenizer or target slots changed")
    inputs, branch_inputs, outputs, handles = {}, {}, {}, []
    modules, names = {}, []
    trainable = adapter.trainable()
    for layer in LAYERS:
        for letter in MODULES:
            name = f"L{layer}_{letter}"
            module = getattr(adapter.backend.language_model.model.layers[layer - 1].self_attn, letter + "_proj")
            modules[name] = module
            names.append(name)
            def pre(module, args, key=name):
                inputs[key] = args[0].detach()
            def branch(module, args, key=name):
                branch_inputs[key] = args[0].detach()
            def post(module, args, value, key=name):
                outputs[key] = value
            handles.extend([module.register_forward_pre_hook(pre),
                            module.lora_A["default"].register_forward_pre_hook(branch),
                            module.register_forward_hook(post)])
    try:
        logits = inject_logits(adapter, visual, q, a)
    finally:
        for handle in handles:
            handle.remove()
    losses = native_token_losses(logits, a)
    mask = a != adapter.pad
    loss = (losses * mask).sum() / mask.sum()
    parameter_names = [f"backend.language_model.model.layers.{layer-1}.self_attn.{letter}_proj.lora_A.default.weight"
                       for layer in LAYERS for letter in MODULES]
    parameters = tuple(trainable[name] for name in parameter_names)
    values = torch.autograd.grad(loss, parameters + tuple(outputs[name] for name in names), retain_graph=token_diagnostic)
    gradients = {name: value.detach().cpu() for name, value in zip(names, values[:len(names)], strict=True)}
    begin = adapter.public_embeddings("USER: ", 1).shape[1]
    end = begin + visual.shape[1]
    seq = inputs[names[0]].shape[1]
    supervised = (valid + seq - a.shape[1]).cpu().tolist()
    interface, spectra, scores, support = {}, {}, {}, {}
    pooled = {}
    for layer in LAYERS:
        reference = inputs[f"L{layer}_q"][0, begin:end]
        pooled[f"mean_{layer}"] = reference.detach().double().cpu().mean(0)
        for letter in MODULES:
            name = f"L{layer}_{letter}"
            interface[name] = {"module_input_vs_q": float((inputs[name][0, begin:end] - reference).abs().max()),
                               "lora_A_input_vs_q": float((branch_inputs[name][0, begin:end] - reference).abs().max())}
            if any(interface[name].values()):
                raise ValueError("Q/K/V visual input coordinate interface differs")
            spectra[name], bases = spectrum(gradients[name], len(valid))
            scores[name] = {prop: residual(proto[f"visual_{layer}"].double(), bases["original"]).norm(dim=1).tolist()
                            for prop, proto in prototypes.items()}
    for name, value in zip(names, values[len(names):], strict=True):
        support[name] = loss_support(value, begin, end, supervised)
    token_rows, token_tensors, additivity = [], {}, {}
    if token_diagnostic:
        for index in range(len(valid)):
            grads = torch.autograd.grad(losses[0, valid[index]] / len(valid), parameters,
                                        retain_graph=index < len(valid) - 1)
            text = case["token_text"][index]
            group = "eos" if index == len(valid) - 1 else (
                "punctuation" if text and all(char in string.punctuation for char in text) else "content")
            row = {"index": index, "token_text": text, "group": group, "nll": float(losses[0, valid[index]].detach()),
                   "gradients": {}}
            for name, grad in zip(names, grads, strict=True):
                value = grad.detach().cpu()
                token_tensors[f"token_{index:02d}_{name}"] = value
                row["gradients"][name] = {"norm": float(value.double().norm()),
                    "cosine_to_full": float(torch.nn.functional.cosine_similarity(value.double().flatten(),
                                            gradients[name].double().flatten(), dim=0))}
            token_rows.append(row)
        for name in names:
            summed = sum(token_tensors[f"token_{index:02d}_{name}"].double() for index in range(len(valid)))
            original = gradients[name].double()
            additivity[name] = {"relative_l2": float((summed - original).norm() / original.norm().clamp_min(1e-30)),
                                "max_absolute_error": float((summed - original).abs().max())}
    tensors = {**gradients, **pooled}
    row = {"case": case["id"], "loss": float(loss.detach()), "supervised_tokens": len(valid),
           "sequence_length": seq, "visual_start": begin, "visual_stop": end, "interface": interface,
           "spectra": spectra, "scores": scores, "loss_support": support,
           "token_diagnostic": token_rows, "token_additivity": additivity}
    return row, tensors, token_tensors


def run(root):
    from core.fl import restore_model
    from evaluation.public_style_study import controlled_training
    out = root / "qa-controls"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(out / "started.json", {"pid": os.getpid(), "host": os.uname().nodename, "time": time.time()})
    start = time.monotonic()
    try:
        plan = read_json(out / "plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Frozen public input changed")
        if file_hash(Path(__file__)) != plan["source_sha256"]:
            raise ValueError("Frozen controls source changed")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one selected GPU")
        if torch.cuda.mem_get_info()[0] < 60 * 2**30:
            raise RuntimeError("At least 60 GiB free required; no changes")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        fingerprint = adapter.fingerprint()
        adapter.set_training_spec(controlled_training(adapter.training_spec))
        bank = read_json(out / "public-bank.json")
        prototypes = {"head": read_tensors(plan["head_prototypes"]),
                      "abdomen": read_tensors(root / "public/abdomen-prototypes.safetensors")}
        rows, hashes = [], {}
        for image, public in enumerate(bank):
            pixels = adapter.prepare_image(Image.open(public["file"]).convert("RGB")).unsqueeze(0).to(adapter.device, adapter.dtype)
            with torch.no_grad():
                visual = adapter.visual_embeddings(pixels)[0].detach()
            for case in plan["cases"]:
                token_diagnostic = image in plan["token_images"] and case["token_decomposition"]
                row, tensors, token_tensors = collect_one(adapter, visual, case, prototypes, token_diagnostic)
                row["image"] = image
                if case["old_qa"]:
                    old = read_tensors(root / "multi-qa/gradients" / f"{public['old_image']:03d}-{case['old_qa']}.safetensors")
                    row["old_q_gradient_errors"] = {str(layer): float((tensors[f"L{layer}_q"] - old[f"gradient_{layer}"]).abs().max()) for layer in LAYERS}
                    if any(row["old_q_gradient_errors"].values()):
                        raise ValueError("Native Q baseline replay differs; no tolerance change or silent retry")
                stem = f"{image:03d}-{case['id']}"
                vector_path, row_path = out / "gradients" / f"{stem}.safetensors", out / "rows" / f"{stem}.json"
                write_tensors(vector_path, tensors)
                write_json(row_path, row)
                for path in (vector_path, row_path):
                    hashes[str(path)] = file_hash(path)
                if token_tensors:
                    path = out / "token-gradients" / f"{stem}.safetensors"
                    write_tensors(path, token_tensors)
                    hashes[str(path)] = file_hash(path)
                rows.append(row)
                write_json(out / "progress.json", {"completed": len(rows), "total": len(bank) * len(plan["cases"]),
                           "image": image, "case": case["id"], "seconds": time.monotonic() - start})
                if len(rows) == 1:
                    write_json(out / "first-artifact.json", {"row_sha256": file_hash(row_path),
                               "old_q_gradient_errors": row["old_q_gradient_errors"], "interface": row["interface"]})
                print(f"QA controls {len(rows)}/{len(bank) * len(plan['cases'])}", flush=True)
        if adapter.fingerprint() != fingerprint:
            raise ValueError("Model changed")
        write_json(out / "predictions.json", rows)
        write_json(out / "predictions-complete.json", {"sha256": file_hash(out / "predictions.json"),
                   "hashes": hashes, "private_reference_read": False, "public_metrics_not_selected": True})
        write_json(out / "complete.json", {"status": "complete", "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
                   "predictions_sha256": file_hash(out / "predictions.json"), "model_fingerprint": fingerprint,
                   "runtime_source": str(Path(__file__).resolve()), "source_sha256": file_hash(Path(__file__))})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc(), "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.root.resolve())


if __name__ == "__main__":
    main()
