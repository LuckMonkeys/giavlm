from dataclasses import asdict, replace
import fnmatch
from pathlib import Path
import re

import torch
from torch.func import functional_call

from core.artifacts import read_json, read_tensors, write_json, write_tensors
from core.config import ModelSpec, TrainingSpec, digest
from core.types import Batch, Observation


def canonicalize_lora_parameter_name(name: str) -> str:
    """Align PEFT adapter labels; full-tuning parameter names remain unchanged."""
    return re.sub(r"(\.lora_(?:A|B|embedding_A|embedding_B)\.)[^.]+(?=\.)",
                  r"\1<adapter>", name)


class NamedGradientAccumulator:
    """CPU streaming mean aligned by names, never positional zip across clients."""

    def __init__(self, target_gradients, target_parameter_names):
        self.names = list(target_parameter_names)
        values = self._map(target_gradients, self.names)
        self.sums = {key: tensor.detach().cpu().clone() for key, tensor in values.items()}
        self.client_count = 1

    @staticmethod
    def _map(gradients, names):
        if len(gradients) != len(names) or not names:
            raise ValueError("Nonempty, equal gradient and name counts are required")
        values = {}
        for name, tensor in zip(names, gradients):
            key = canonicalize_lora_parameter_name(name)
            if key in values:
                raise ValueError("Duplicate canonical parameter name")
            if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
                raise ValueError("Gradients must be finite tensors")
            values[key] = tensor
        return values

    def add_client(self, gradients, parameter_names):
        values = self._map(gradients, parameter_names)
        if values.keys() != self.sums.keys():
            raise ValueError("Client parameter names differ")
        for key, tensor in values.items():
            if tensor.shape != self.sums[key].shape or tensor.dtype != self.sums[key].dtype:
                raise ValueError(f"Client gradient shape or dtype differs: {key}")
        for key, tensor in values.items():
            self.sums[key].add_(tensor.detach().cpu())
        self.client_count += 1

    def finalize(self):
        return tuple(self.sums[canonicalize_lora_parameter_name(name)] / self.client_count
                     for name in self.names), {"num_clients": self.client_count}


def resolve_upload(names, patterns):
    """Expand fnmatch patterns into an explicit, ordered allowlist.

    Patterns keep configs readable for models with hundreds of adapters, but the
    resolved names are what travels and what the attacker sees, so an unmatched
    pattern is an error rather than a silently empty upload.
    """
    if not patterns:
        return sorted(names)
    selected = set()
    for pattern in patterns:
        matched = fnmatch.filter(names, pattern)
        if not matched:
            raise ValueError(f"Upload pattern {pattern!r} matched no trainable parameter")
        selected.update(matched)
    return sorted(selected)


def mask_upload(updates, parameter_names):
    """Explicit parameter allowlist applied to a client update before upload."""
    if not parameter_names or len(set(parameter_names)) != len(parameter_names):
        raise ValueError("Upload names must be unique and nonempty")
    if set(parameter_names) - set(updates):
        raise ValueError("Upload mask references unknown parameters")
    return {name: updates[name] for name in parameter_names}


def simulate_secure_aggregation(client_updates, model_fingerprints):
    """Numerical aggregate only, not cryptography or a multi-client inversion protocol."""
    if not client_updates or len(client_updates) != len(model_fingerprints):
        raise ValueError("Each update requires its initial model fingerprint")
    if not all(model_fingerprints) or len(set(model_fingerprints)) != 1:
        raise ValueError("Clients must share the exact initial model and LoRA basis")
    first = client_updates[0]
    accumulator = NamedGradientAccumulator(list(first.values()), list(first))
    for update in client_updates[1:]:
        accumulator.add_client(list(update.values()), list(update))
    mean, _ = accumulator.finalize()
    return dict(zip(first, mean))


def simulate_update(adapter, batch: Batch, spec: TrainingSpec, differentiable=False):
    """Replay the public SGD rule with one candidate batch per local step."""
    if len(batch.images) != spec.batch_size * spec.local_steps:
        raise ValueError("Each local step requires exactly batch_size candidate slots")
    initial = adapter.trainable()
    params = dict(initial)
    for step in range(spec.local_steps):
        start = step * spec.batch_size
        chunk = batch.slice(start, start + spec.batch_size)
        loss = functional_call(adapter, params, (chunk.images, chunk.questions, chunk.targets))
        grads = torch.autograd.grad(loss, tuple(params.values()),
                                    create_graph=differentiable, allow_unused=True)
        updates = {name: torch.zeros_like(p) if grad is None else grad
                   for (name, p), grad in zip(params.items(), grads)}
        if spec.observation == "gradient":
            if spec.local_steps != 1:
                raise ValueError("A gradient observation has exactly one step")
            return updates
        params = {name: p - spec.lr * updates[name] for name, p in params.items()}
        if not differentiable:
            params = {name: value.detach().requires_grad_(True) for name, value in params.items()}
    return {name: params[name] - initial[name] for name in initial}


def capture(adapter, batch, questions: list[str], targets: list[str]):
    spec = adapter.training_spec
    update = {k: v.detach().clone() for k, v in simulate_update(adapter, batch, spec).items()}
    uploaded = resolve_upload(list(update), spec.upload_parameters)
    obs = Observation(model=replace(adapter.spec), training=replace(spec),
                      tensors=mask_upload(update, uploaded),
                      model_fingerprint=adapter.fingerprint(),
                      public_questions=list(questions) if spec.knowledge != "private" and spec.task == "vqa" else [],
                      public_targets=list(targets) if spec.knowledge == "text_known" else [],
                      public_question_ids=batch.questions.detach().cpu().tolist()
                      if spec.knowledge != "private" and spec.task == "vqa" else [],
                      public_target_ids=batch.targets.detach().cpu().tolist() if spec.knowledge == "text_known" else [])
    obs.validate()
    return obs


def save_observation(directory, observation):
    directory = Path(directory)
    if (directory / "observation.json").exists():
        raise FileExistsError(directory)
    observation.validate()
    metadata = {"schema_version": 1, "model": asdict(observation.model),
                "training": asdict(observation.training),
                "model_fingerprint": observation.model_fingerprint,
                "public_questions": observation.public_questions,
                "public_targets": observation.public_targets,
                "public_question_ids": observation.public_question_ids,
                "public_target_ids": observation.public_target_ids,
                "parameter_names": sorted(observation.tensors)}
    write_tensors(directory / "update.safetensors", observation.tensors)
    from core.artifacts import file_hash
    metadata["update_sha256"] = file_hash(directory / "update.safetensors")
    metadata["observation_id"] = digest(metadata)
    write_json(directory / "observation.json", metadata)
    return metadata["observation_id"]


def load_observation(directory, device=None):
    directory = Path(directory)
    meta = read_json(directory / "observation.json")
    allowed = {"schema_version", "model", "training", "model_fingerprint", "public_questions",
               "public_targets", "public_question_ids", "public_target_ids",
               "parameter_names", "update_sha256", "observation_id"}
    if set(meta) != allowed:
        raise ValueError("Observation contains unknown or missing fields")
    check = dict(meta)
    observed_id = check.pop("observation_id")
    if digest(check) != observed_id:
        raise ValueError("Observation metadata integrity check failed")
    from core.artifacts import file_hash
    if file_hash(directory / "update.safetensors") != meta["update_sha256"]:
        raise ValueError("Observed update integrity check failed")
    model = ModelSpec(**meta["model"])
    if device:
        model.device = device
    obs = Observation(model, TrainingSpec(**meta["training"]),
                      read_tensors(directory / "update.safetensors", "cpu"),
                      meta["model_fingerprint"], meta["public_questions"], meta["public_targets"],
                      meta["public_question_ids"], meta["public_target_ids"], meta["schema_version"])
    if sorted(obs.tensors) != meta["parameter_names"]:
        raise ValueError("Observed parameter names differ from metadata")
    obs.validate()
    return obs


def fedavg(adapter, client_batches, weights=None):
    """Average uploaded parameter deltas, including LoRA A/B in a shared basis."""
    if not client_batches:
        raise ValueError("No clients selected")
    weights = weights or [len(b.images) for b in client_batches]
    if len(weights) != len(client_batches) or any(w <= 0 for w in weights):
        raise ValueError("Client weights must be positive")
    spec = replace(adapter.training_spec, observation="client_delta")
    average = {name: torch.zeros_like(p) for name, p in adapter.trainable().items()}
    for batch, weight in zip(client_batches, weights):
        delta = simulate_update(adapter, batch, spec)
        for name in average:
            average[name].add_(delta[name].detach(), alpha=weight / sum(weights))
    with torch.no_grad():
        for name, parameter in adapter.trainable().items():
            parameter.add_(average[name])
    return average


def save_model(directory, adapter, metadata=None):
    directory = Path(directory)
    write_tensors(directory / "model.safetensors", adapter.state_dict())
    write_json(directory / "model.json", {"model": asdict(adapter.spec),
                                         "training": asdict(adapter.training_spec),
                                         "fingerprint": adapter.fingerprint(),
                                         "description": adapter.description(), **(metadata or {})})


def restore_model(directory, device=None):
    from core.vlm_wrapper import build_model
    directory = Path(directory)
    meta = read_json(directory / "model.json")
    spec = ModelSpec(**meta["model"])
    if device:
        spec.device = device
    adapter = build_model(spec, TrainingSpec(**meta["training"]))
    adapter.load_state_dict(read_tensors(directory / "model.safetensors", "cpu"), strict=True)
    if adapter.fingerprint() != meta["fingerprint"]:
        raise ValueError("Model fingerprint mismatch")
    return adapter
