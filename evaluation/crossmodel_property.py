"""Frozen cross-model head-property pilot, native QA and fixed-QA paired controls."""

import argparse
from dataclasses import asdict, replace
import gc
import os
from pathlib import Path
import random
import socket
import time
import traceback
import uuid

import torch
from PIL import Image

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
from evaluation.deep_property_probe import distance_to_gradient
from evaluation.image_property_data import pixel_hash


LAYERS = (24, 32)
METHODS = tuple(f"{module.upper()}{layer}" for layer in LAYERS for module in ("q", "k", "v"))
QUESTION, ANSWER = "Describe the image.", "The image is shown."
PUBLIC = Path("outputs/diagnostics/image_property_next_20261028/styles")
MODELS = Path("outputs/diagnostics/crossmodel_text_20261017/models/slake")


def module_key(family, method):
    """Both selected Qwen3.5 blocks are full attention, never DeltaNet substitutes."""
    letter, layer = method[0].lower(), int(method[1:])
    if layer not in LAYERS or letter not in ("q", "k", "v"):
        raise ValueError("Unregistered module")
    prefix = {"llava": "backend.language_model.model", "qwen35": "backend.model.language_model"}[
        family
    ]
    return f"{prefix}.layers.{layer - 1}.self_attn.{letter}_proj.lora_A.default.weight"


def score_upload(tensors, prototypes, keys):
    """Classifier accepts tensors and public vectors only; no target QA/labels/paths."""
    return {
        method: distance_to_gradient(prototypes[method], tensors[keys[method]])
        for method in METHODS
    }


def prepare(root):
    if (root / "plan.json").exists():
        raise FileExistsError("Plan already frozen")
    model_paths = {name: (MODELS / name / "round-0010").resolve() for name in ("llava", "qwen35")}
    public = [
        row for row in read_json(PUBLIC / "public_manifest.json") if row["split"] in ("fit", "dev")
    ]
    if len(public) != 96 or {row["split"] for row in public} != {"fit", "dev"}:
        raise ValueError("Use the original 48 fit and 48 development public images")
    files = [PUBLIC / "fit_labels.json", PUBLIC / "evaluation_labels.json"]
    files.extend(Path(row["file"]) for row in public)
    files.extend(
        path / name for path in model_paths.values() for name in ("model.json", "model.safetensors")
    )
    for dataset in ("slake", "rad"):
        path = root / "private" / f"{dataset}-targets.json"
        targets = read_json(path)
        if len(targets) != 20 or len({row["row"]["image_id"] for row in targets}) != 20:
            raise ValueError("Exactly twenty distinct images per dataset required")
        if {row["label"] for row in targets} != {"head", "non-head"}:
            raise ValueError("Both target classes required")
        files.append(path)
        files.extend(Path(row["row"]["image"]) for row in targets)
    write_json(root / "public-bank.json", public)
    files.append(root / "public-bank.json")
    write_json(
        root / "plan.json",
        {
            "primary": "Q32",
            "methods": list(METHODS),
            "model_paths": {k: str(v) for k, v in model_paths.items()},
            "datasets": ["slake", "rad"],
            "images_per_dataset": 20,
            "qa_conditions": ["natural", "fixed"],
            "public_question": QUESTION,
            "public_answer": ANSWER,
            "public_fit_images": 48,
            "public_dev_images": 48,
            "classes": ["head", "non-head"],
            "selection": "Q32 declared before new gradients; Q24 and K/V secondary, no public or private reselection",
            "victim_policy": "Same SLAKE-trained Round10 snapshot per model for both datasets; no retraining",
            "knowledge": "private; no image, QA, token lengths or property labels used by scoring",
            "prototype_policy": "Re-extract each model's native LoRA-A visual inputs; average within class then unit normalize",
            "scope": "Pilot; SLAKE A repeated images, twenty visually reviewed VQA-RAD eval images; patient independence unknown",
            "hashes": {str(path.resolve()): file_hash(path) for path in files},
            "source_sha256": file_hash(Path(__file__)),
            "minimum_free_gib": 60,
        },
    )


def check_lengths(adapter, question, answer):
    """Fail instead of silently truncating the original task."""
    lengths = []
    for text, capacity in (
        (question, adapter.spec.question_length),
        (answer, adapter.spec.target_length),
    ):
        length = len(adapter.tokenizer.encode(text, add_special_tokens=False))
        if length + 1 > capacity:
            raise ValueError(
                "Original QA exceeds frozen input capacity; never truncate or resample"
            )
        lengths.append(length)
    return lengths


def hook_features(adapter, family, image, question, answer, keys):
    """Record actual LoRA-A inputs at every visual position in a native forward."""
    check_lengths(adapter, question, answer)
    batch = adapter.batch(image, [question], [answer])
    modules = dict(adapter.named_modules())
    recorded, handles, visual_info = {}, [], {}
    original_visual = adapter.visual_embeddings

    def visual(images):
        values, grid = original_visual(images)
        visual_info["count"] = values.shape[1]
        return values, grid

    adapter.visual_embeddings = visual
    try:
        for method, key in keys.items():
            module = modules[key.removesuffix(".weight")]

            def hook(_, args, name=method):
                recorded[name] = args[0].detach()

            handles.append(module.register_forward_pre_hook(hook))
        with torch.no_grad():
            loss = adapter(batch.images, batch.questions, batch.targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native public feature loss")
            if family == "llava":
                start = adapter.public_embeddings("USER: ", 1).shape[1]
            else:
                from core.adapters.qwen35 import USER

                start = (
                    adapter.public_embeddings(USER, 1).shape[1]
                    + adapter.public_embeddings("<|vision_start|>", 1).shape[1]
                )
            stop = start + visual_info["count"]
            result = {name: value[0, start:stop].double().cpu() for name, value in recorded.items()}
            if set(result) != set(keys) or any(
                value.shape[0] != visual_info["count"] for value in result.values()
            ):
                raise ValueError("Wrong visual range or missing LoRA module hooks")
            for layer in LAYERS:
                if not all(
                    torch.equal(result[f"Q{layer}"], result[f"{letter}{layer}"])
                    for letter in ("K", "V")
                ):
                    raise ValueError("Q/K/V do not share the declared visual input coordinates")
    finally:
        adapter.visual_embeddings = original_visual
        for handle in handles:
            handle.remove()
    return result, {"visual_start": start, "visual_count": visual_info["count"]}


def validate_observation(observation, fingerprint):
    if (
        observation.model_fingerprint != fingerprint
        or observation.sample_count != 1
        or observation.training.knowledge != "private"
        or observation.training.token_lengths_known
        or observation.public_images is not None
        or observation.public_questions
        or observation.public_targets
        or observation.public_question_ids
        or observation.public_target_ids
        or observation.public_question_lengths
        or observation.public_target_lengths
    ):
        raise ValueError("Only matching anonymous private observations allowed")


def capture_fixed_model(adapter, batch, fingerprint):
    """Native gradient capture with an invariant fingerprint checked at run boundaries."""
    from core.fl import simulate_update
    from core.types import Observation

    training = adapter.training_spec
    if training.knowledge != "private" or training.token_lengths_known:
        raise ValueError("Fast capture is restricted to this private length-unknown protocol")
    update = {
        name: value.detach().clone()
        for name, value in simulate_update(adapter, batch, training).items()
    }
    observation = Observation(
        model=replace(adapter.spec),
        training=replace(training),
        tensors=update,
        model_fingerprint=fingerprint,
    )
    observation.validate()
    return observation


def public_templates(adapter, family, root, keys, fingerprint):
    out = root / family
    rows = read_json(root / "public-bank.json")
    fit_labels = read_json(PUBLIC / "fit_labels.json")
    means, records, gradient_paths = {}, {}, []
    for index, row in enumerate(rows):
        image = (
            adapter.prepare_image(Image.open(row["file"]).convert("RGB"))
            .unsqueeze(0)
            .to(adapter.device, adapter.dtype)
        )
        features, info = hook_features(adapter, family, image, QUESTION, ANSWER, keys)
        means[row["id"]] = {method: value.mean(0) for method, value in features.items()}
        records[row["id"]] = info
        if row["split"] == "dev":
            batch = adapter.batch(image, [QUESTION], [ANSWER])
            observation = capture_fixed_model(adapter, batch, fingerprint)
            if not gradient_paths:
                from core.fl import capture

                reference = capture(adapter, batch, [], [])
                if (
                    reference.model_fingerprint != fingerprint
                    or reference.tensors.keys() != observation.tensors.keys()
                ):
                    raise ValueError("Native capture gate identity differs")
                if not all(
                    torch.equal(reference.tensors[name], value)
                    for name, value in observation.tensors.items()
                ):
                    raise ValueError(
                        "Cached-fingerprint capture differs from original native gradients"
                    )
                write_json(
                    out / "native-capture-gate.json",
                    {
                        "all_tensors_exact": True,
                        "tensors": len(reference.tensors),
                        "fingerprint": fingerprint,
                        "original_capture_sha256_comparison": "torch.equal for every tensor",
                        "model_identity_policy": "Full fingerprint at start, first original-native gate and end; no parameter mutation",
                    },
                )
                del reference
            path = out / "public/gradients" / f"{index:03d}.safetensors"
            write_tensors(path, {method: observation.tensors[key] for method, key in keys.items()})
            gradient_paths.append({"id": row["id"], "path": str(path)})
            del observation
        write_json(
            out / "progress.json",
            {"stage": "public_templates", "completed": index + 1, "total": 96},
        )
        print(f"{family} public {index + 1}/96", flush=True)
    prototypes = {}
    for method in METHODS:
        centers = torch.stack(
            [
                torch.stack(
                    [
                        means[row["id"]][method]
                        for row in rows
                        if row["split"] == "fit" and fit_labels[row["id"]] == label
                    ]
                ).mean(0)
                for label in ("head", "non-head")
            ]
        )
        prototypes[method] = torch.nn.functional.normalize(centers, dim=1)
    write_tensors(out / "public/prototypes.safetensors", prototypes)
    write_tensors(
        out / "public/visual-means.safetensors",
        {f"{i}_{m}": v for i, values in means.items() for m, v in values.items()},
    )
    write_json(out / "public/feature-ranges.json", records)
    write_json(out / "public/gradient-list.json", gradient_paths)
    # Scores commit before joining public dev evaluation labels; not used for selection.
    predictions = [
        {
            "id": r["id"],
            "scores": {
                m: distance_to_gradient(prototypes[m], read_tensors(r["path"])[m]) for m in METHODS
            },
        }
        for r in gradient_paths
    ]
    write_json(out / "public/predictions.json", predictions)
    write_json(
        out / "public/predictions-complete.json",
        {
            "sha256": file_hash(out / "public/predictions.json"),
            "prototype_sha256": file_hash(out / "public/prototypes.safetensors"),
            "primary": "Q32",
            "reselection": False,
        },
    )
    return prototypes


def medical_capture(adapter, family, root, keys, fingerprint):
    from core.data import load_batch
    from core.fl import save_observation

    out = root / family
    uploads, mapping, length_audit = [], {}, []
    entries = [
        (dataset, row)
        for dataset in ("slake", "rad")
        for row in read_json(root / "private" / f"{dataset}-targets.json")
    ]
    random.SystemRandom().shuffle(entries)
    for index, (dataset, item) in enumerate(entries):
        if pixel_hash(Image.open(item["row"]["image"])) != item["pixel_sha256"]:
            raise ValueError("Frozen target pixels changed")
        native_lengths = check_lengths(adapter, item["row"]["question"], item["row"]["target"])
        length_audit.append({"dataset": dataset, "run": item["run"], "lengths": native_lengths})
        for qa in ("natural", "fixed"):
            row = (
                item["row"]
                if qa == "natural"
                else {**item["row"], "question": QUESTION, "target": ANSWER}
            )
            batch = load_batch([row], adapter)
            observation = capture_fixed_model(adapter, batch, fingerprint)
            validate_observation(observation, fingerprint)
            if any(
                not torch.isfinite(observation.tensors[key]).all()
                or observation.tensors[key].norm() == 0
                for key in keys.values()
            ):
                raise ValueError("Selected native gradient is empty or nonfinite")
            identifier = uuid.uuid4().hex
            path = out / "observations" / identifier
            save_observation(path, observation)
            uploads.append(
                {"id": identifier, "dataset": dataset, "qa": qa, "observation": str(path)}
            )
            mapping[identifier] = {"dataset": dataset, "run": item["run"], "qa": qa}
            if index == 0 and qa == "natural":
                write_json(
                    out / "first-artifact.json",
                    {
                        "parameter_tensors": len(observation.tensors),
                        "update_sha256": file_hash(path / "update.safetensors"),
                        "knowledge": "private",
                        "token_lengths_known": False,
                        "fingerprint": fingerprint,
                        "module_keys": keys,
                    },
                )
            del observation, batch
            gc.collect()
        write_json(
            out / "progress.json", {"stage": "medical_capture", "completed": index + 1, "total": 40}
        )
        print(f"{family} medical {index + 1}/40 (both QA)", flush=True)
    write_json(out / "private/upload-map.json", mapping)
    write_json(out / "private/length-audit.json", length_audit)
    write_json(out / "upload-list.json", uploads)
    write_json(
        out / "capture-complete.json",
        {
            "images": 40,
            "uploads": 80,
            "hashes": {
                str(Path(row["observation"]) / name): file_hash(Path(row["observation"]) / name)
                for row in uploads
                for name in ("observation.json", "update.safetensors")
            },
        },
    )


def run(root, family):
    from core.fl import load_observation, restore_model

    out = root / family
    if (out / "started.json").exists():
        raise FileExistsError("No overwrite or automatic retry")
    write_json(
        out / "started.json",
        {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    started = time.monotonic()
    plan = read_json(root / "plan.json")
    if torch.cuda.device_count() != 1 or torch.cuda.mem_get_info()[0] < plan["minimum_free_gib"] * 2**30:
        raise RuntimeError("Exactly one selected idle GPU with the declared memory reserve required")
    for path, sha in plan["hashes"].items():
        if file_hash(path) != sha:
            raise ValueError("Frozen input changed")
    if file_hash(Path(__file__)) != plan["source_sha256"]:
        raise ValueError("Frozen cross-model source changed")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = restore_model(plan["model_paths"][family], "cuda:0")
    adapter.eval()
    fingerprint = adapter.fingerprint()
    training = replace(
        adapter.training_spec,
        algorithm="fedsgd",
        local_optimizer="sgd",
        batch_size=1,
        local_steps=1,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        knowledge="private",
        token_lengths_known=False,
    )
    adapter.set_training_spec(training)
    keys = {method: module_key(family, method) for method in METHODS}
    if family == "qwen35" and any(
        adapter.backend.model.language_model.config.layer_types[layer - 1] != "full_attention"
        for layer in LAYERS
    ):
        raise ValueError("Qwen selected layers must be actual full attention")
    if not all(key in adapter.trainable() for key in keys.values()):
        raise ValueError("Selected architecture module not present")
    write_json(
        out / "runtime.json",
        {
            "model": asdict(adapter.spec),
            "training": asdict(training),
            "fingerprint": fingerprint,
            "keys": keys,
            "runtime_source": str(Path(__file__).resolve()),
            "torch": torch.__version__,
            "source_sha256": file_hash(Path(__file__)),
        },
    )
    prototypes = public_templates(adapter, family, root, keys, fingerprint)
    medical_capture(adapter, family, root, keys, fingerprint)
    if adapter.fingerprint() != fingerprint:
        raise ValueError("Model changed during read-only experiment")
    peak = torch.cuda.max_memory_allocated() / 2**20
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    predictions = []
    for row in read_json(out / "upload-list.json"):
        observation = load_observation(row["observation"], "cpu")
        validate_observation(observation, fingerprint)
        predictions.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "qa": row["qa"],
                "scores": score_upload(observation.tensors, prototypes, keys),
            }
        )
        del observation
    write_json(out / "predictions.json", predictions)
    write_json(
        out / "predictions-complete.json",
        {
            "sha256": file_hash(out / "predictions.json"),
            "primary": "Q32",
            "labels_read_for_scoring": False,
            "target_text_or_lengths_used_by_classifier": False,
        },
    )
    write_json(
        out / "complete.json",
        {
            "status": "complete",
            "images": 40,
            "uploads": 80,
            "seconds": time.monotonic() - started,
            "peak_memory_mib": peak,
            "prediction_sha256": file_hash(out / "predictions.json"),
            "runtime_source": str(Path(__file__).resolve()),
        },
    )
    print(f"{family} complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("llava", "qwen35"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.stage == "prepare":
            prepare(args.root.resolve())
        else:
            run(args.root.resolve(), args.family)
    except BaseException:
        out = args.root / (args.family or "setup")
        write_json(out / "failed.json", {"traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
