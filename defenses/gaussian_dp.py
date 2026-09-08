import math
import secrets
import torch
from defenses.clipping import ClippingDefense


class GaussianDPDefense(ClippingDefense):
    """Client clipping + Gaussian perturbation; no epsilon/delta guarantee or accountant."""

    def __init__(self, max_norm=1.0, noise_multiplier=1.0):
        super().__init__(max_norm)
        if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
            raise ValueError("noise_multiplier must be finite and nonnegative")
        self.noise_multiplier = noise_multiplier

    def apply(self, updates, *, generator=None):
        clipped = super().apply(updates)
        if generator is None:
            generator = torch.Generator().manual_seed(secrets.randbits(63))
        return {name: value + torch.randn(value.shape, generator=generator, device="cpu",
                                          dtype=torch.float32).to(value) * self.max_norm * self.noise_multiplier
                for name, value in clipped.items()}
