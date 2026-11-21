"""Public-only causal suffix ablation: visual inputs, LoRA-A inputs and gradients."""

import argparse
import hashlib
import math
import os
from pathlib import Path
import time
import traceback

import torch
from PIL import Image

from attacks.analytic.dager_subspace import row_basis
from core.artifacts import file_hash, read_json, write_json, write_tensors
from evaluation.multilayer_property import LAYERS, OLD


QUESTION = "Describe the image."
ANSWER = "The image is shown."
OTHER_QUESTION = "Please describe what is visible in this picture."
OTHER_ANSWER = "This picture contains a visible object."
VARIANTS = ("baseline", "question_only", "answer_only", "both_changed", "compact_suffix",
            "prefix_only", "changed_public_prefix", "image_only", "repeat_baseline")
GRADIENT_VARIANTS = ("baseline", "question_only", "answer_only", "both_changed", "repeat_baseline")


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def differences(value, reference):
    if value.shape != reference.shape:
        raise ValueError("Feature/gradient comparison shapes differ")
    value, reference = value.double().cpu(), reference.double().cpu()
    if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
        raise ValueError("Nonfinite values in causal check")
    delta = value - reference
    return {"max_absolute_error": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / reference.norm().clamp_min(1e-30)),
            "bit_equal": bool(torch.equal(value, reference))}


def gradient_comparison(value, reference):
    result = differences(value, reference)
    a, _ = row_basis(value.double().cpu(), 1e-6, 1e-10)
    b, _ = row_basis(reference.double().cpu(), 1e-6, 1e-10)
    shared = float((a @ b.T).square().sum())
    denominator = len(a) + len(b)
    distance = math.sqrt(max(0., denominator - 2 * shared) / denominator) if denominator else 0.
    if result["bit_equal"]:
        distance = 0.
    result.update(rank=len(a), reference_rank=len(b), rowspace_distance=distance)
    return result


def prepare(root):
    template = root / "causal-plan-template.json"
    if template.exists():
        raise FileExistsError("Causal check already prepared")
    public = read_json(OLD / "styles/public_manifest.json")
    cases = []
    for name, group, style in [("photo_head", "head-00", "photo"), ("photo_limb", "non-head-00", "photo"),
                               ("line_head", "head-00", "line")]:
        row = next(row for row in public if row["group"] == group and row["style"] == style)
        cases.append({"name": name, "file": str(Path(row["file"]).resolve())})
    row = next(row for row in read_json(root / "public/abdomen-bank.json") if row["label"] == "abdomen"
               and row["style"] == "render" and row["split"] == "fit")
    cases.append({"name": "render_abdomen", "file": str(Path(row["file"]).resolve())})
    bank = Path("outputs/diagnostics/synthetic_head_n20_gpu5_20261017")
    for name, region in [("synthetic_head_CT", "head"), ("synthetic_abdomen_CT", "abdomen")]:
        row = next(row for row in read_json(bank / "public/bank.json")
                   if row["region_prompt"] == region and row["modality_prompt"] == "CT")
        cases.append({"name": name, "file": str((bank / row["generated_file"]).resolve())})
    prior = read_json(root / "probe-plan.json")
    paths = [Path(prior["model"]) / "model.json", Path(prior["model"]) / "model.safetensors"]
    paths.extend(Path(row["file"]) for row in cases)
    write_json(template, {"model": prior["model"], "gpu_uuid": None, "public_cases": cases,
               "variants": list(VARIANTS), "gradient_variants": list(GRADIENT_VARIANTS),
               "feature_layers": list(range(1, 33)), "gradient_layers": list(LAYERS),
               "question": QUESTION, "answer": ANSWER, "other_question": OTHER_QUESTION,
               "other_answer": OTHER_ANSWER, "dtype": "native bfloat16",
               "feature_mode": "eval, no_grad, use_cache=false; compare all visual positions and pooled means",
               "gradient_mode": "Separate native gradient-enabled pass in identical conditions; q-proj LoRA-A only",
               "positive_controls": ["changed image", "changed preceding public prefix", "removed public prefix"],
               "numerical_policy": "Report bit equality and raw errors, no post-hoc tolerance or dtype changes. Prefix-only changes sequence shape and may change numerical kernels.",
               "uses_private_reference": False, "source_sha256": file_hash(Path(__file__)),
               "hashes": {str(path): file_hash(path) for path in paths}})


def inputs(adapter, visual, variant):
    from core.adapters.llava import llava_input_pieces
    question = OTHER_QUESTION if variant in ["question_only", "both_changed"] else QUESTION
    answer = OTHER_ANSWER if variant in ["answer_only", "both_changed"] else ANSWER
    if len(adapter.tokenizer.encode(question, add_special_tokens=False)) >= adapter.spec.question_length:
        raise ValueError("Question would truncate; do not silently change the condition")
    if len(adapter.tokenizer.encode(answer, add_special_tokens=False)) >= adapter.spec.target_length:
        raise ValueError("Answer would truncate; do not silently change the condition")
    q, a = adapter.encode([question], adapter.spec.question_length), adapter.encode([answer], adapter.spec.target_length)
    if variant == "compact_suffix":
        q = q[:, :adapter.content_lengths(q)[0] + 1]
        a = a[:, :adapter.content_lengths(a)[0] + 1]
    pieces = llava_input_pieces(adapter, visual, adapter.embedding()(q), adapter.embedding()(a[:, :-1]))
    if variant == "changed_public_prefix":
        pieces[0] = adapter.public_embeddings("SYSTEM: A helpful assistant.\nUSER: ", 1)
    if variant == "prefix_only":
        pieces = pieces[:2]
    if variant == "image_only":
        pieces = [visual]
        begin = 0
    else:
        begin = pieces[0].shape[1]
    return torch.cat(pieces, 1), begin, begin + visual.shape[1]


def features(adapter, visual, variant):
    with torch.no_grad():
        embedding, begin, end = inputs(adapter, visual, variant)
        q_values, a_values, handles = {}, {}, []
        for layer in range(1, 33):
            def q_hook(module, args, key=layer):
                q_values[key] = args[0][0, begin:end].detach().cpu().contiguous()
            def a_hook(module, args, key=layer):
                a_values[key] = args[0][0, begin:end].detach().cpu().contiguous()
            module = adapter.backend.language_model.model.layers[layer - 1].self_attn.q_proj
            handles.append(module.register_forward_pre_hook(q_hook))
            handles.append(module.lora_A["default"].register_forward_pre_hook(a_hook))
        try:
            adapter.backend.language_model(inputs_embeds=embedding, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
    if len(q_values) != 32 or len(a_values) != 32:
        raise ValueError("Every expected q-proj and LoRA-A hook must fire")
    return q_values, {str(layer): differences(a_values[layer], q_values[layer]) for layer in q_values}, embedding.shape[1]


def gradients(adapter, visual, variant):
    from evaluation.visual_concept_probe import inject_logits, native_hard_loss
    question = OTHER_QUESTION if variant in ["question_only", "both_changed"] else QUESTION
    answer = OTHER_ANSWER if variant in ["answer_only", "both_changed"] else ANSWER
    q = adapter.encode([question], adapter.spec.question_length)
    a = adapter.encode([answer], adapter.spec.target_length)
    named = adapter.trainable()
    keys = [f"backend.language_model.model.layers.{layer-1}.self_attn.q_proj.lora_A.default.weight" for layer in LAYERS]
    loss = native_hard_loss(inject_logits(adapter, visual, q, a), a, adapter.pad)
    values = torch.autograd.grad(loss, tuple(named[key] for key in keys))
    return {layer: value.detach().cpu() for layer, value in zip(LAYERS, values, strict=True)}, float(loss.detach())


def run(root):
    from core.fl import restore_model
    from evaluation.public_style_study import controlled_training
    out = root / "causal"
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(out / "started.json", {"pid": os.getpid(), "host": os.uname().nodename})
    start = time.monotonic()
    try:
        plan = read_json(out / "plan.json")
        for path, sha in plan["hashes"].items():
            if file_hash(path) != sha:
                raise ValueError("Public input changed")
        if file_hash(Path(__file__)) != plan["source_sha256"]:
            raise ValueError("Causal verifier source changed")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != plan["gpu_uuid"] or torch.cuda.device_count() != 1:
            raise ValueError("Expose only the selected GPU")
        if torch.cuda.mem_get_info()[0] < 60 * 2 ** 30:
            raise RuntimeError("At least 60 GiB free required; no silent changes")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        adapter = restore_model(plan["model"], "cuda:0")
        adapter.eval()
        fingerprint = adapter.fingerprint()
        adapter.set_training_spec(controlled_training(adapter.training_spec))
        rows, first_image = [], None
        for index, case in enumerate(plan["public_cases"]):
            image = adapter.prepare_image(Image.open(case["file"]).convert("RGB")).unsqueeze(0).to(adapter.device, adapter.dtype)
            with torch.no_grad():
                visual = adapter.visual_embeddings(image)[0].detach()
            baseline, baseline_grads = None, None
            for variant in VARIANTS:
                value, hook_match, length = features(adapter, visual, variant)
                if baseline is None:
                    baseline = value
                row = {"case": case["name"], "variant": variant, "sequence_length": length,
                       "visual_feature_shape": list(value[1].shape),
                       "features": {str(layer): {**differences(value[layer], baseline[layer]),
                            "mean_max_absolute_error": float((value[layer].double().mean(0) - baseline[layer].double().mean(0)).abs().max()),
                            "sha256": tensor_hash(value[layer]), "reference_sha256": tensor_hash(baseline[layer])}
                            for layer in value}, "lora_A_input_vs_q_proj_input": hook_match}
                stored = {f"mean_{layer}": value[layer].double().mean(0) for layer in value}
                if variant in GRADIENT_VARIANTS:
                    grads, loss = gradients(adapter, visual, variant)
                    if baseline_grads is None:
                        baseline_grads = grads
                    row["gradient_loss"] = loss
                    row["gradients"] = {str(layer): gradient_comparison(grads[layer], baseline_grads[layer]) for layer in grads}
                    stored.update({f"gradient_{layer}": gradient for layer, gradient in grads.items()})
                if variant == "baseline" and first_image is not None:
                    row["different_image_control"] = {str(layer): differences(value[layer], first_image[layer]) for layer in value}
                write_tensors(out / "vectors" / f"{index:02d}-{variant}.safetensors", stored)
                write_json(out / "rows" / f"{index:02d}-{variant}.json", row)
                rows.append(row)
                write_json(out / "progress.json", {"case": index + 1, "cases": len(plan["public_cases"]),
                           "variant": variant, "completed": len(rows), "total": len(plan["public_cases"]) * len(VARIANTS)})
                if len(rows) == 1:
                    write_json(out / "first-artifact.json", {"sha256": file_hash(out / "rows/00-baseline.json")})
                print(f"Causal check {len(rows)}/{len(plan['public_cases']) * len(VARIANTS)} {variant}", flush=True)
            if first_image is None:
                first_image = baseline
        if adapter.fingerprint() != fingerprint:
            raise ValueError("Model changed during causal check")
        report = {"public_images": len(plan["public_cases"]), "feature_layers": 32,
                  "gradient_layers": list(LAYERS), "rows": rows, "uses_private_reference": False,
                  "numerical_policy": plan["numerical_policy"], "source_sha256": file_hash(Path(__file__))}
        write_json(out / "report.json", report)
        write_json("docs/experiments/data/visual-concept-probe/causal-feature-aggregate.json", report)
        write_json(out / "complete.json", {"status": "complete", "seconds": time.monotonic() - start,
                   "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
                   "report_sha256": file_hash(out / "report.json")})
    except BaseException:
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.root.resolve())


if __name__ == "__main__":
    main()
