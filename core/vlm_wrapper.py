from dataclasses import asdict
from abc import ABC, abstractmethod
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from core.config import (MODEL_DEVICE_MAPS, MODEL_DTYPES, MODEL_FAMILIES,
                         ModelSpec, TrainingSpec)
from core.types import Batch


def canonical_probabilities(probs: torch.Tensor, eos: int, pad: int):
    """Make every position after EOS padding, continuously for soft candidates."""
    survival = torch.cat([torch.ones_like(probs[:, :1, 0]),
                          (1 - probs[:, :-1, eos]).cumprod(dim=1)], dim=1)
    pad_vector = F.one_hot(torch.tensor(pad, device=probs.device), probs.shape[-1]).to(probs)
    return probs * survival[..., None] + (1 - survival[..., None]) * pad_vector, survival


class VLMAdapter(nn.Module, ABC):
    protocol_version = "fixed-block-eos-v1"

    def __init__(self, spec: ModelSpec, training: TrainingSpec):
        super().__init__()
        for field in ["image_size", "question_length", "target_length"]:
            if getattr(spec, field) <= 0:
                raise ValueError(f"model.{field} must be positive")
        self.spec, self.training_spec = spec, training
        self.position_gradient_names = []
        self.tokenizer = None

    @abstractmethod
    def embedding(self):
        raise NotImplementedError

    @abstractmethod
    def is_language(self, name):
        raise NotImplementedError

    @abstractmethod
    def target_logits(self, images, questions, targets):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def vocab_size(self):
        return self.embedding().weight.shape[0]

    @property
    def eos(self):
        return self.tokenizer.eos_token_id

    @property
    def pad(self):
        return self.tokenizer.pad_token_id

    def encode(self, texts: list[str], length: int):
        result = []
        for text in texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)[:length - 1]
            ids = ids + [self.eos]
            result.append(ids + [self.pad] * (length - len(ids)))
        return torch.tensor(result, device=self.device, dtype=torch.long)

    def decode(self, tokens: torch.Tensor):
        if tokens.ndim == 3:
            tokens = tokens.argmax(-1)
        result = []
        for row in tokens.detach().cpu().tolist():
            if self.eos in row:
                row = row[:row.index(self.eos)]
            result.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return result

    def probabilities(self, tokens):
        if tokens.ndim == 2:
            return F.one_hot(tokens.long(), self.vocab_size).to(self.dtype)
        return tokens.to(self.dtype)

    def batch(self, images, questions, targets):
        return Batch(images.to(device=self.device, dtype=self.dtype),
                     self.encode(questions, self.spec.question_length),
                     self.encode(targets, self.spec.target_length))

    def trainable(self):
        return {name: p for name, p in self.named_parameters() if p.requires_grad}

    def configure_training(self):
        mode = self.training_spec.mode
        for name, p in self.named_parameters():
            p.requires_grad_(mode == "full" or (mode == "llm_full" and self.is_language(name)))
        if mode == "lora_llm":
            if self.training_spec.lora_rank <= 0 or self.training_spec.lora_alpha <= 0:
                raise ValueError("LoRA rank and alpha must be positive")
            from peft import LoraConfig, inject_adapter_in_model
            names = [name for name, module in self.named_modules()
                     if isinstance(module, nn.Linear) and self.is_language(name)
                     and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj"}]
            if not names:
                raise ValueError("No language attention projections found for LoRA")
            config = LoraConfig(r=self.training_spec.lora_rank,
                                lora_alpha=self.training_spec.lora_alpha,
                                lora_dropout=0.0, bias="none", target_modules=names)
            inject_adapter_in_model(config, self)
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        if not self.trainable():
            raise ValueError("Empty trainable parameter set")

    def forward(self, images, questions, targets):
        q = self.probabilities(questions)
        raw_targets = self.probabilities(targets)
        q, _ = canonical_probabilities(q, self.eos, self.pad)
        y, alive = canonical_probabilities(raw_targets, self.eos, self.pad)
        logits = self.target_logits(images, q, y)
        # Include EOS itself; exclude padding and positions following EOS.
        weights = (alive * (1 - raw_targets[..., self.pad])).to(logits.device)
        loss = -(raw_targets.float().to(logits.device) * logits.float().log_softmax(-1)).sum(-1)
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def generate(self, images, questions):
        """Greedy output under this adapter's fixed-block training format."""
        with torch.no_grad():
            q = self.probabilities(questions)
            q, _ = canonical_probabilities(q, self.eos, self.pad)
            ids = torch.full((len(images), self.spec.target_length), self.pad,
                             dtype=torch.long, device=self.device)
            ended = torch.zeros(len(images), dtype=torch.bool, device=self.device)
            for pos in range(ids.shape[1]):
                logits = self.target_logits(images, q, self.probabilities(ids))
                next_id = logits[:, pos].argmax(-1)
                ids[:, pos] = torch.where(ended, self.pad, next_id)
                ended |= next_id == self.eos
            return self.decode(ids)

    def fingerprint(self):
        h = hashlib.sha256()
        for name, value in self.state_dict().items():
            h.update(name.encode())
            h.update(str((tuple(value.shape), value.dtype)).encode())
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def description(self):
        return {"protocol": self.protocol_version, "model": asdict(self.spec),
                "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.trainable().values()),
                "trainable_names": list(self.trainable()),
                "position_gradient_names": self.position_gradient_names}



# Compatibility for existing adapters and saved experiment scripts.
ModelAdapter = VLMAdapter

def build_model(spec: ModelSpec, training: TrainingSpec):
    from core.adapters.tiny_llava import TinyAdapter
    from core.adapters.llava import LlavaAdapter
    from core.adapters.blip2 import Blip2Adapter
    from core.adapters.qwen_vl import QwenVLAdapter
    classes = {"tiny": TinyAdapter, "llava": LlavaAdapter,
               "blip2": Blip2Adapter, "qwen2_5_vl": QwenVLAdapter}
    if spec.family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {spec.family}")
    if spec.family != "tiny" and not spec.revision:
        raise ValueError("A pinned model revision is required; run doctor --resolve-revision")
    if spec.dtype not in MODEL_DTYPES:
        raise ValueError(f"Unsupported model dtype: {spec.dtype}")
    if spec.family != "tiny" and spec.dtype == "float64":
        raise ValueError("float64 is only supported for the tiny correctness fixture")
    if spec.device_map not in MODEL_DEVICE_MAPS:
        raise ValueError(f"Unsupported model device_map: {spec.device_map}")
    if spec.family == "tiny" and spec.device_map:
        raise ValueError("device_map is supported for Hugging Face models only")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(spec.seed)
        adapter = classes[spec.family](spec, training)
        adapter.configure_training()
    if spec.device_map:
        adapter.image_mean = adapter.image_mean.to(adapter.device)
        adapter.image_std = adapter.image_std.to(adapter.device)
        return adapter
    return adapter.to(device=spec.device, dtype=getattr(torch, spec.dtype))
