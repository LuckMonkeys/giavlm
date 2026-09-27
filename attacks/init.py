"""Candidate image initialization: random noise or attacker-held images, optionally perturbed."""
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from core.artifacts import read_tensors

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_initial_images(path, adapter, count):
    """Load start images as the model's public RGB view, shaped like the candidate slots.

    ``path`` is a safetensors file with an ``images`` tensor in [0, 1], one image
    file, or a directory of image files (sorted by name). A single image is shared
    by every slot.
    """
    path = Path(path)
    if path.suffix == ".safetensors":
        images = read_tensors(path)["images"].float().cpu()
    else:
        from PIL import Image, ImageOps
        files = (sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
                 if path.is_dir() else [path])
        if not files:
            raise FileNotFoundError(f"No initialization images under {path}")
        views = []
        for file in files:
            with Image.open(file) as source:
                views.append(adapter.prepare_image(ImageOps.exif_transpose(source).convert("RGB")))
        images = torch.stack(views).float()
    if len(images) == 1 and count > 1:
        images = images.expand(count, -1, -1, -1).clone()
    size = adapter.spec.image_size
    if tuple(images.shape) != (count, 3, size, size):
        raise ValueError(f"Initialization images have shape {tuple(images.shape)}, "
                         f"expected {(count, 3, size, size)}")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Initialization images must be finite RGB values in [0, 1]")
    return images


def gaussian_blur(images, sigma):
    radius = max(1, math.ceil(3 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=images.dtype, device=images.device)
    kernel = torch.exp(-offsets.square() / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels = images.shape[1]
    padded = F.pad(images, (radius, radius, radius, radius), mode="reflect")
    horizontal = F.conv2d(padded, kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1),
                          groups=channels)
    return F.conv2d(horizontal, kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1),
                    groups=channels)


def perturb(images, perturbation, level, seed):
    """Deterministic start perturbation; uniform_mix level 1 is pure uniform noise."""
    images = images.float().cpu()
    generator = torch.Generator().manual_seed(seed)
    if perturbation == "none":
        return images
    if perturbation == "uniform_mix":
        noise = torch.rand(images.shape, generator=generator)
        return (1 - level) * images + level * noise
    if perturbation == "gaussian":
        return (images + level * torch.randn(images.shape, generator=generator)).clamp(0, 1)
    if perturbation == "blur":
        return gaussian_blur(images, level).clamp(0, 1)
    raise ValueError(f"Unknown initialization perturbation: {perturbation}")


def perturb_tokens(ids, perturbation, level, seed, vocab_size, special_ids):
    """Replace each content token with a random non-special token with probability ``level``.

    EOS, PAD and other special tokens keep their positions, so a start never adds
    length information beyond the template or reference it came from.
    """
    ids = ids.long().cpu()
    if perturbation == "none":
        return ids
    if perturbation != "replace":
        raise ValueError(f"Unknown text initialization perturbation: {perturbation}")
    generator = torch.Generator().manual_seed(seed)
    special = torch.tensor(sorted(special_ids), dtype=torch.long)
    allowed = torch.ones(vocab_size, dtype=torch.bool)
    allowed[special] = False
    choices = allowed.nonzero().flatten()
    content = ~torch.isin(ids, special)
    replace = content & (torch.rand(ids.shape, generator=generator) < level)
    random = choices[torch.randint(len(choices), ids.shape, generator=generator)]
    return torch.where(replace, random, ids)
