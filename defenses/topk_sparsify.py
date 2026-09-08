import math
import torch
from defenses.base import BaseDefense


class TopKSparsifyDefense(BaseDefense):
    """Keep ceil(ratio * numel) coordinates per tensor, retaining dense wire shapes."""

    def __init__(self, ratio=0.1):
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        self.ratio = ratio

    def apply(self, updates, *, generator=None):
        self.validate(updates)
        result = {}
        for name, value in updates.items():
            flat = value.flatten()
            indices = flat.abs().topk(math.ceil(self.ratio * flat.numel())).indices
            result[name] = torch.zeros_like(flat).scatter(0, indices, flat[indices]).reshape_as(value)
        return result
