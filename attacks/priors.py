from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from core.artifacts import file_hash, read_tensors


def total_variation(images):
    return ((images[:, :, 1:] - images[:, :, :-1]).abs().mean()
            + (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean())


def patch_prior(images, patch):
    terms = []
    for axis in [-2, -1]:
        borders = torch.arange(patch, images.shape[axis], patch, device=images.device)
        if len(borders):
            difference = images.index_select(axis, borders) - images.index_select(axis, borders - 1)
            terms.append(difference.square().mean().sqrt())
    return sum(terms, images.new_zeros(()))


def document_prior(images):
    channels = images.shape[1]
    lap = images.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    gaussian = images.new_tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    edges = F.conv2d(padded, lap.expand(channels, 1, 3, 3), groups=channels).abs().mean()
    smooth = F.conv2d(padded, gaussian.expand(channels, 1, 3, 3), groups=channels)
    color = (images[:, 1:] - images[:, :-1]).abs().mean()
    return total_variation(images) + color + edges + (smooth - images).abs().mean()


class BatchNormPrior(nn.Module):
    """GradViT BN prior; explicit local MoCo-compatible ResNet50 weights."""

    def __init__(self, checkpoint, device):
        super().__init__()
        from torchvision.models import resnet50
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("GradViT needs a local ResNet50 prior .safetensors checkpoint")
        self.net = resnet50(weights=None)
        weights = read_tensors(checkpoint)
        expected = self.net.state_dict()
        absent = set(expected) - set(weights)
        if absent - {"fc.weight", "fc.bias"} or set(weights) - set(expected):
            raise ValueError("Prior checkpoint must use torchvision ResNet50 backbone names")
        self.net.load_state_dict(weights, strict=False)
        self.net.eval().requires_grad_(False).to(device)
        self.terms = []
        self.handles = [module.register_forward_pre_hook(self._hook)
                        for module in self.net.modules() if isinstance(module, nn.BatchNorm2d)]
        self.provenance = {"architecture": "resnet50", "sha256": file_hash(checkpoint),
                           "weight_source": "user-supplied; document training source in experiment notes"}

    def _hook(self, module, inputs):
        x = inputs[0]
        mean = x.mean((0, 2, 3))
        var = x.var((0, 2, 3), unbiased=False)
        self.terms.append((mean - module.running_mean).norm() + (var - module.running_var).norm())

    def forward(self, images):
        self.terms.clear()
        images = F.interpolate(images.float(), size=(224, 224), mode="bilinear", align_corners=False)
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.net((images - mean) / std)
        result = sum(self.terms)
        self.terms.clear()
        return result


class TextPrior:
    """Discrete language-prior scoring for LAMP-style permutation proposals."""

    def __init__(self, name, revision, device, allow_download=False):
        if not name:
            raise ValueError("lamp_adapted requires prior_model and a pinned prior_revision")
        self.name = name
        self.provenance = {"model": name, "revision": revision}
        if name == "tiny-public-bigram":
            # Only an offline fixture, never presented as a pretrained language model.
            corpus = ["what color is the object", "what shape is the object"]
            corpus += [f"a {c} {s}" for c in ["red", "green", "blue"] for s in ["circle", "square"]]
            self.counts, self.totals = {}, {}
            for sentence in corpus:
                words = ["<bos>"] + sentence.split() + ["<eos>"]
                for first, second in zip(words, words[1:]):
                    self.counts[first, second] = self.counts.get((first, second), 0) + 1
                    self.totals[first] = self.totals.get(first, 0) + 1
            return
        if not revision:
            raise ValueError("Pin the public text prior revision")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"revision": revision, "local_files_only": not allow_download}
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device).eval()
        self.model.requires_grad_(False)

    def score(self, texts):
        import math
        scores = []
        for text in texts:
            if self.name == "tiny-public-bigram":
                words = ["<bos>"] + text.split() + ["<eos>"]
                scores.append(sum(-math.log((self.counts.get((x, y), 0) + 1) /
                                           (self.totals.get(x, 0) + 27))
                                  for x, y in zip(words, words[1:])) / max(1, len(words) - 1))
            else:
                with torch.no_grad():
                    ids = self.tokenizer.encode(text, add_special_tokens=False)
                    start = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
                    ids = torch.tensor([[start] + ids + [self.tokenizer.eos_token_id]],
                                       device=next(self.model.parameters()).device)
                    scores.append(self.model(input_ids=ids, labels=ids).loss.item())
        return sum(scores) / max(1, len(scores))
