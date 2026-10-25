"""Gaussian upload perturbation, matching the reference's per-tensor max scaling.

No clipping or differential-privacy guarantee is implied by this mechanism.
"""
import math
import secrets

import torch

from defenses.base import BaseDefense


class GaussianNoiseDefense(BaseDefense):
    def __init__(self, noise_ratio):
        if not math.isfinite(noise_ratio) or noise_ratio < 0:
            raise ValueError("noise_ratio must be finite and nonnegative")
        self.noise_ratio = noise_ratio

    def apply(self, updates, *, generator=None):
        self.validate(updates)
        if generator is None:
            generator = torch.Generator().manual_seed(secrets.randbits(63))
        result = {}
        for name, value in sorted(updates.items()):
            sigma = value.detach().float().abs().max() * self.noise_ratio
            noise = torch.randn(value.shape, generator=generator, device="cpu", dtype=torch.float32)
            result[name] = (value.float() + noise.to(value.device) * sigma).to(value.dtype)
        return result
