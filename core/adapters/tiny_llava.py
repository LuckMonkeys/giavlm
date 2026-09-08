"""Deterministic miniature LLaVA-like fixture, not a pretrained TinyLLaVA."""
import math
import torch
from torch import nn
from core.vlm_wrapper import VLMAdapter

class TinyTokenizer:
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "what", "color", "shape", "is", "the",
             "object", "red", "green", "blue", "square", "circle", "a", "on", "black",
             "background", "one", "two", "left", "right", "yes", "no", "white", "small"]
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2, 3]

    def __len__(self):
        return len(self.words)

    def encode(self, text, add_special_tokens=False):
        tokens = [self.words.index(w) if w in self.words else 3
                  for w in text.lower().replace("?", "").replace(".", "").split()]
        return ([1] + tokens + [2]) if add_special_tokens else tokens

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids if not skip_special_tokens or i > 3)


class TinyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(x.shape[-1])
        mask = torch.ones(scores.shape[-2:], device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, -torch.inf)
        return self.o_proj(scores.softmax(-1) @ v)


class TinyLM(nn.Module):
    def __init__(self, vocab, dim, length):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.position = nn.Parameter(torch.randn(1, length, dim) * 0.02)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = TinyAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.head = nn.Linear(dim, vocab)

    def forward(self, x):
        x = x + self.position[:, :x.shape[1]]
        x = x + self.attention(self.norm1(x))
        return self.head(x + self.mlp(self.norm2(x)))


class TinyAdapter(VLMAdapter):
    def __init__(self, spec, training):
        super().__init__(spec, training)
        self.tokenizer = TinyTokenizer()
        dim = spec.hidden_size
        patches = (spec.image_size // spec.patch_size) ** 2
        self.vision_patch = nn.Conv2d(3, dim, spec.patch_size, stride=spec.patch_size)
        self.vision_position = nn.Parameter(torch.randn(1, patches, dim) * 0.02)
        self.connector = nn.Linear(dim, dim)
        self.language = TinyLM(len(self.tokenizer), dim,
                               patches + spec.question_length + spec.target_length + 4)
        self.position_gradient_names = ["vision_position"]

    def embedding(self):
        return self.language.embed_tokens

    def is_language(self, name):
        return name.startswith("language.")

    def target_logits(self, images, q, y):
        image = self.vision_patch(images).flatten(2).transpose(1, 2) + self.vision_position
        image = self.connector(image)
        bos = self.embedding().weight[self.tokenizer.bos_token_id].expand(len(images), 1, -1)
        pieces = [bos, image]
        if self.training_spec.task == "vqa":
            pieces.append(q @ self.embedding().weight)
        pieces.extend([bos, y[:, :-1] @ self.embedding().weight])
        logits = self.language(torch.cat(pieces, dim=1))
        return logits[:, -y.shape[1]:]
