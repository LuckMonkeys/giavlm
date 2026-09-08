import math
import torch
from defenses.base import BaseDefense


class ClippingDefense(BaseDefense):
    """Global L2 clipping of one client update, not per-example DP clipping."""

    def __init__(self, max_norm=1.0):
        if not math.isfinite(max_norm) or max_norm <= 0:
            raise ValueError("max_norm must be finite and positive")
        self.max_norm = max_norm

    def apply(self, updates, *, generator=None):
        self.validate(updates)
        device = next(iter(updates.values())).device
        norm = torch.stack([v.float().square().sum().to(device) for v in updates.values()]).sum().sqrt()
        scale = (self.max_norm / norm.clamp_min(1e-20)).clamp(max=1)
        return {name: value * scale.to(value.device) for name, value in updates.items()}
