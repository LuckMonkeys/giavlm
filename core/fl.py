from dataclasses import asdict, replace
import fnmatch
from pathlib import Path
import re

import torch
from torch.func import functional_call

from core.artifacts import file_hash, read_json, read_tensors, write_json, write_tensors
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


def _accumulated_gradients(adapter, params, batch, start, spec, differentiable):
    """Average microbatch gradients for one optimizer step."""
    accumulated = {name: torch.zeros_like(param) for name, param in params.items()}
    for accumulation_id in range(spec.gradient_accumulation_steps):
        offset = start + accumulation_id * spec.batch_size
        chunk = batch.slice(offset, offset + spec.batch_size)
        loss = functional_call(adapter, params, (chunk.images, chunk.questions, chunk.targets))
        grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=differentiable,
                                    allow_unused=True)
        for (name, _), grad in zip(params.items(), grads, strict=True):
            if grad is not None:
                accumulated[name] = accumulated[name] + grad / spec.gradient_accumulation_steps
    return accumulated


def _adamw_step(params, gradients, moments, variances, decay_names, spec, step):
    """Functional AdamW update matching torch.optim.AdamW's default rule."""
    beta1, beta2 = spec.adam_beta1, spec.adam_beta2
    next_params, next_moments, next_variances = {}, {}, {}
    bias_correction1 = 1 - beta1 ** step
    bias_correction2_sqrt = (1 - beta2 ** step) ** 0.5
    for name, param in params.items():
        grad = gradients[name]
        moment = beta1 * moments[name] + (1 - beta1) * grad
        variance = beta2 * variances[name] + (1 - beta2) * grad.square()
        decayed = param * (1 - spec.lr * spec.weight_decay) if name in decay_names else param
        # At an exactly zero moment, sqrt has an infinite derivative although the
        # AdamW update is zero. Clamp only that underflow region so attack replay
        # retains finite higher-order derivatives without changing nonzero steps.
        stable_variance = variance.clamp_min(torch.finfo(variance.dtype).tiny)
        denominator = stable_variance.sqrt() / bias_correction2_sqrt + spec.adam_epsilon
        next_params[name] = decayed - (spec.lr / bias_correction1) * moment / denominator
        next_moments[name], next_variances[name] = moment, variance
    return next_params, next_moments, next_variances


def _simulate_local_update(adapter, batch: Batch, spec: TrainingSpec, update_type: str,
                           differentiable=False):
    """Replay the public local optimizer without mutating the victim model."""
    expected = spec.sample_count
    if len(batch.images) != expected:
        raise ValueError(f"Local update requires exactly {expected} candidate slots")
    initial = adapter.trainable()
    params = dict(initial)
    moments = {name: torch.zeros_like(param) for name, param in params.items()}
    variances = {name: torch.zeros_like(param) for name, param in params.items()}
    decay_names = adapter.decay_parameter_names()

    for step in range(spec.local_steps):
        start = step * spec.batch_size * spec.gradient_accumulation_steps
        updates = _accumulated_gradients(adapter, params, batch, start, spec, differentiable)
        if update_type == "gradient":
            if spec.local_steps != 1 or spec.gradient_accumulation_steps != 1:
                raise ValueError("A gradient observation has exactly one step")
            return updates

        if spec.local_optimizer == "sgd":
            params = {name: param - spec.lr * updates[name] for name, param in params.items()}
        elif spec.local_optimizer == "adamw":
            params, moments, variances = _adamw_step(
                params, updates, moments, variances, decay_names, spec, step + 1)
        else:
            raise ValueError(f"Unsupported local optimizer: {spec.local_optimizer}")

        if not differentiable:
            params = {name: value.detach().requires_grad_(True) for name, value in params.items()}
            moments = {name: value.detach() for name, value in moments.items()}
            variances = {name: value.detach() for name, value in variances.items()}
    return {name: params[name] - initial[name] for name in initial}


def simulate_update(adapter, batch: Batch, spec: TrainingSpec, differentiable=False):
    """Replay the client upload dictated by the configured federated algorithm."""
    from core.aggregation import create_federated_algorithm
    return create_federated_algorithm(spec).client_update(adapter, batch, differentiable)


def capture(adapter, batch, questions: list[str], targets: list[str]):
    spec = adapter.training_spec
    update = {k: v.detach().clone() for k, v in simulate_update(adapter, batch, spec).items()}
    obs = Observation(model=replace(adapter.spec), training=replace(spec),
                      tensors=update,
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
    metadata = {"schema_version": 3, "model": asdict(observation.model),
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
    if meta.get("schema_version") != 3:
        raise ValueError(
            f"Unsupported observation schema v{meta.get('schema_version')}; expected schema v3")
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


def save_model(directory, adapter, metadata=None):
    directory = Path(directory)
    tensors = adapter.federated_state()
    tensor_path = directory / "model.safetensors"
    write_tensors(tensor_path, tensors)
    write_json(directory / "model.json", {
        **(metadata or {}),
        "schema_version": 4,
        "state_scope": "strategy_mutable",
        "model": asdict(adapter.spec),
        "training": asdict(adapter.training_spec),
        "parameter_names": list(tensors),
        "state_sha256": file_hash(tensor_path),
        "fingerprint": adapter.fingerprint(),
        "description": adapter.description(),
    })


def restore_model(directory, device=None):
    from core.vlm_wrapper import build_model
    directory = Path(directory)
    meta = read_json(directory / "model.json")
    if meta.get("schema_version") != 4:
        raise ValueError(f"Unsupported model schema v{meta.get('schema_version')}; expected schema v4")
    required = {"state_scope", "model", "training", "parameter_names", "state_sha256",
                "fingerprint", "description"}
    missing = required - set(meta)
    if missing:
        raise ValueError(f"Model metadata is missing required fields: {sorted(missing)}")
    if meta["state_scope"] != "strategy_mutable":
        raise ValueError(f"Unsupported model state scope: {meta['state_scope']}")
    tensor_path = directory / "model.safetensors"
    if file_hash(tensor_path) != meta["state_sha256"]:
        raise ValueError("Model state integrity check failed")
    spec = ModelSpec(**meta["model"])
    if device:
        spec.device = device
    adapter = build_model(spec, TrainingSpec(**meta["training"]))
    expected = adapter.federated_state()
    expected_names = list(expected)
    if meta["parameter_names"] != expected_names:
        raise ValueError("Model state parameter names differ from the configured strategy")
    tensors = read_tensors(tensor_path, "cpu")
    if list(sorted(tensors)) != expected_names:
        raise ValueError("Model state tensor names differ from metadata")
    with torch.no_grad():
        for name in expected_names:
            saved, current = tensors[name], expected[name]
            if saved.shape != current.shape:
                raise ValueError(f"Model state shape differs: {name}")
            if saved.dtype != current.dtype:
                raise ValueError(f"Model state dtype differs: {name}")
            if not torch.isfinite(saved).all():
                raise ValueError(f"Model state contains nonfinite values: {name}")
            current.copy_(saved.to(current.device))
    if adapter.fingerprint() != meta["fingerprint"]:
        raise ValueError("Model fingerprint mismatch")
    return adapter
