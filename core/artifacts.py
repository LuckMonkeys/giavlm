from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid

import torch
from safetensors.torch import load_file, save_file


def write_json(path: str | Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(asdict(value) if is_dataclass(value) else value,
                               indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_tensors(path: str | Path, tensors: dict[str, torch.Tensor]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    save_file({k: v.detach().cpu().contiguous().clone() for k, v in tensors.items()}, str(temp))
    os.replace(temp, path)


def read_tensors(path: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    return load_file(str(path), device=device)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    versions = {}
    for name in ["giavlm", "torch", "torchvision", "transformers", "peft", "safetensors",
                 "numpy", "scipy", "rouge-score", "scikit-image", "lpips"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": source_fingerprint(),
            "versions": versions, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}


def source_fingerprint():
    root = Path(__file__).parent.parent
    h = hashlib.sha256()
    for package in ["core", "attacks", "defenses", "metrics", "evaluation", "utils"]:
        for path in sorted((root / package).rglob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()
