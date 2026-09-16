"""High-level federated algorithms and their complete update semantics."""
from abc import ABC, abstractmethod

import torch

from core.config import TrainingSpec
from core.types import Batch


def weighted_mean_updates(updates, weights):
    """Stream a weighted mean without retaining every client update."""
    weights = list(weights)
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("Client weights must be positive")
    total = sum(weights)
    average = None
    for update, weight in zip(updates, weights, strict=True):
        if not update:
            raise ValueError("Client update cannot be empty")
        if average is None:
            average = {name: torch.zeros_like(value) for name, value in update.items()}
        if set(update) != set(average):
            raise ValueError("Client updates have different parameter names")
        for name, value in update.items():
            if value.shape != average[name].shape:
                raise ValueError(f"Client update shape mismatch for {name}")
            if value.dtype != average[name].dtype or value.device != average[name].device:
                raise ValueError(f"Client update placement mismatch for {name}")
            if not torch.isfinite(value).all():
                raise ValueError(f"Client update is nonfinite for {name}")
            average[name].add_(value.detach(), alpha=weight / total)
    if average is None:
        raise ValueError("No client updates to aggregate")
    return average


def apply_server_update(adapter, update, scale=1.0):
    """Validate a server update completely before mutating the global model."""
    parameters = adapter.trainable()
    if not update or set(update) - set(parameters):
        raise ValueError("Server update contains no parameters or unknown parameters")
    for name, value in update.items():
        if value.shape != parameters[name].shape:
            raise ValueError(f"Server update shape mismatch for {name}")
        if value.dtype != parameters[name].dtype or value.device != parameters[name].device:
            raise ValueError(f"Server update placement mismatch for {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Server update is nonfinite for {name}")
    with torch.no_grad():
        for name, value in update.items():
            parameters[name].add_(value, alpha=scale)


class BaseFederatedAlgorithm(ABC):
    """Own client computation, upload meaning, aggregation, and application."""

    name = ""
    upload_type = ""

    def __init__(self, training: TrainingSpec):
        self.training = training

    def compute_client_update(self, adapter, batch: Batch, differentiable=False):
        """Run the algorithm's local client computation before upload selection."""
        # Local import avoids a module cycle: core.fl exposes the public replay API.
        from core.fl import _simulate_sgd_update
        return _simulate_sgd_update(adapter, batch, self.training, self.upload_type,
                                    differentiable)

    def prepare_upload(self, update):
        """Select the named tensors exposed to the server."""
        from core.fl import mask_upload, resolve_upload
        names = resolve_upload(list(update), self.training.upload_parameters)
        return mask_upload(update, names)

    def client_update(self, adapter, batch: Batch, differentiable=False):
        """Compute exactly the value this algorithm makes visible to the server."""
        return self.prepare_upload(self.compute_client_update(adapter, batch, differentiable))

    def aggregate(self, updates, weights):
        return weighted_mean_updates(updates, weights)

    @abstractmethod
    def apply(self, adapter, update):
        """Apply one aggregate to the global model."""
        raise NotImplementedError


class FedSGDAlgorithm(BaseFederatedAlgorithm):
    name = "fedsgd"
    upload_type = "gradient"

    def apply(self, adapter, update):
        apply_server_update(adapter, update, scale=-self.training.lr)


class FedAvgAlgorithm(BaseFederatedAlgorithm):
    name = "fedavg"
    upload_type = "client_delta"

    def apply(self, adapter, update):
        apply_server_update(adapter, update)


def create_federated_algorithm(training: TrainingSpec) -> BaseFederatedAlgorithm:
    """Construct the configured end-to-end federated rule."""
    if training.algorithm == "fedsgd":
        if training.local_steps != 1:
            raise ValueError("FedSGD requires local_steps=1")
        return FedSGDAlgorithm(training)
    if training.algorithm == "fedavg":
        return FedAvgAlgorithm(training)
    raise ValueError(f"Unknown federated algorithm: {training.algorithm}")
