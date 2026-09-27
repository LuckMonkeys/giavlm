"""Gradient diagnostics that replay the private reference and perturbations of it.

These probes read private references and sit outside the threat model. Every
report carries ``uses_private_reference: true``; none of it may feed attack
selection, presets or benchmark tables. Attacks started from a perturbed
reference use the general ``attack.init_source=private_reference`` path instead.
"""
from pathlib import Path

import torch

from attacks.init import perturb
from core.artifacts import read_json, read_tensors, source_fingerprint
from core.types import Batch

PRIVATE_REFERENCE_NOTE = ("Reads private references and is outside the threat model; "
                          "never an attack input or a benchmark result.")


def load_capture(capture_dir, device=None):
    """Restore the observed model, public observation and private truth of one capture."""
    from core.data import load_batch
    from core.fl import load_observation, restore_model

    capture_dir = Path(capture_dir)
    public, private = capture_dir / "public", capture_dir / "private"
    observation = load_observation(public, device)
    model_path = public / "model"
    if (public / "model_ref.json").exists():
        model_path = Path(read_json(public / "model_ref.json")["path"])
    if read_json(model_path / "model.json")["fingerprint"] != observation.model_fingerprint:
        raise ValueError("Capture model differs from the observed model state")
    adapter = restore_model(model_path, observation.model.device)
    adapter.set_training_spec(observation.training)
    observation.tensors = {k: v.to(adapter.trainable()[k].device)
                           for k, v in observation.tensors.items()}

    truth = read_json(private / "truth.json")
    if truth["observation_id"] != read_json(public / "observation.json")["observation_id"]:
        raise ValueError("Private truth belongs to a different observation")
    images = read_tensors(private / "images.safetensors")["images"].float()
    # Reload through the capture data path to check preprocessing and tokenization.
    batch = load_batch(truth["samples"], adapter)
    return adapter, observation, Batch(images, batch.questions, batch.targets), batch.images


def residuals(adapter, observation, batch):
    """Observable matching residuals of one candidate batch against the upload."""
    from attacks.objectives import matching_loss
    from core.fl import simulate_update

    batch = Batch(batch.images.to(adapter.device, adapter.dtype), batch.questions, batch.targets)
    update = simulate_update(adapter, batch, observation.training)
    predicted = {k: update[k].detach() for k in observation.tensors}
    observed = observation.tensors
    squared = matching_loss(predicted, observed, "l2")
    denominator = sum(y.float().square().sum().to(squared.device) for y in observed.values())
    per_tensor = {}
    for name in sorted(observed):
        x, y = predicted[name].float(), observed[name].float()
        per_tensor[name] = {"cosine": _cosine(x, y), "norm_ratio": _ratio(x.norm(), y.norm())}
    return {"cosine_loss": matching_loss(predicted, observed, "cosine").item(),
            "relative_l2": (squared / denominator.clamp_min(1e-20)).item(),
            "max_abs_error": max((predicted[k].float() - observed[k].float()).abs().max().item()
                                 for k in observed),
            "per_tensor": per_tensor}


def connector_features(adapter, batch):
    """Connector input and output for a batch, captured without gradients."""
    name, module = next((n, m) for n, m in adapter.named_modules()
                        if n and adapter.is_connector(n + "."))
    captured = {}

    def hook(_, inputs, output):
        captured["input"] = inputs[0].detach().float()
        captured["output"] = output.detach().float()

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            adapter.target_logits(batch.images.to(adapter.device, adapter.dtype),
                                  adapter.probabilities(batch.questions),
                                  adapter.probabilities(batch.targets))
    finally:
        handle.remove()
    return name, captured


def feature_agreement(reference, candidate):
    result = {}
    for key in ["input", "output"]:
        x, y = candidate[key].flatten(), reference[key].flatten()
        result[f"connector_{key}"] = {"cosine": _cosine(x, y),
                                      "relative_error": _ratio((x - y).norm(), y.norm())}
    return result


def reconstruction_images(directory):
    """Committed attack images, or the best images of an unfinished attack's last checkpoint."""
    directory = Path(directory)
    if (directory / "images.safetensors").exists():
        return read_tensors(directory / "images.safetensors")["images"].float()
    generation = read_json(directory / "checkpoint.json")["generation"]
    state = read_tensors(directory / "checkpoints" / generation / "state.safetensors")
    if "best.images" not in state:
        raise ValueError(f"No reconstruction images under {directory}")
    return state["best.images"].float()


def diagnose(capture_dir, device=None, seed=0, other_captures=(), reconstructions=(),
             gaussian_levels=(0.02, 0.05, 0.1, 0.2), blur_levels=(1.0, 2.0, 4.0),
             mix_levels=(0.1, 0.25, 0.5, 0.75, 1.0)):
    """Replay truth and perturbations; report gradient and connector-feature agreement."""
    from metrics.image import image_metrics

    adapter, observation, truth_batch, reloaded = load_capture(capture_dir, device)
    truth = truth_batch.images
    report = {"schema_version": 1, "uses_private_reference": True, "note": PRIVATE_REFERENCE_NOTE,
              "observation_id": read_json(Path(capture_dir) / "public" / "observation.json")[
                  "observation_id"],
              "condition": _condition(adapter, observation), "seed": seed,
              "source_sha256": source_fingerprint(),
              "data_path": _data_path_check(observation, truth_batch, reloaded)}
    connector, reference = connector_features(adapter, truth_batch)
    report["connector_module"] = connector

    candidates = [("truth", None, truth)]
    candidates += [("gaussian", s, perturb(truth, "gaussian", s, seed))
                   for s in gaussian_levels]
    candidates += [("blur", s, perturb(truth, "blur", s, seed)) for s in blur_levels]
    candidates += [("uniform_mix", a, perturb(truth, "uniform_mix", a, seed))
                   for a in mix_levels]
    for directory in other_captures:
        other = read_tensors(Path(directory) / "private" / "images.safetensors")["images"].float()
        if other.shape != truth.shape:
            raise ValueError("Other-capture images differ in shape from the observed batch")
        candidates.append(("other_image", Path(directory).parent.name, other))
    for directory in reconstructions:
        images = reconstruction_images(directory)
        if images.shape != truth.shape:
            raise ValueError("Reconstruction images differ in shape from the observed batch")
        candidates.append(("reconstruction", str(directory), images))

    report["variants"] = []
    for family, level, images in candidates:
        batch = Batch(images.clamp(0, 1), truth_batch.questions, truth_batch.targets)
        _, features = connector_features(adapter, batch)
        row = {"family": family, "level": level,
               "image": [image_metrics(truth[i], batch.images[i]) for i in range(len(truth))],
               **residuals(adapter, observation, batch), **feature_agreement(reference, features)}
        report["variants"].append(row)
    return report


def _condition(adapter, observation):
    training = observation.training
    return {"model": observation.model.name, "revision": observation.model.revision,
            "fine_tuning_strategy": training.fine_tuning_strategy,
            "fine_tuning_stage": training.fine_tuning_stage,
            "server_round": training.server_round, "task": training.task,
            "algorithm": training.algorithm, "knowledge": training.knowledge,
            "batch_size": training.batch_size, "local_steps": training.local_steps,
            "observed_tensors": len(observation.tensors),
            "observed_parameters": sum(t.numel() for t in observation.tensors.values()),
            "dtype": str(adapter.dtype)}


def _data_path_check(observation, truth_batch, reloaded):
    """Compare the stored private batch with a fresh load and with public token slots."""
    result = {"image_max_abs_error": (reloaded.float().cpu() - truth_batch.images.cpu())
              .abs().max().item()}
    for field, public in [("questions", observation.public_question_ids),
                          ("targets", observation.public_target_ids)]:
        ids = getattr(truth_batch, field).detach().cpu().tolist()
        result[f"public_{field}_match"] = ids == public if public else None
    return result


def _cosine(x, y):
    return ((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-20)).item()


def _ratio(numerator, denominator):
    return (numerator / denominator.clamp_min(1e-20)).item()
