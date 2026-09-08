from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class ModelSpec:
    family: str = "tiny"
    name: str = "tiny-vlm"
    revision: str = ""
    seed: int = 17
    device: str = "cpu"
    dtype: str = "float32"
    image_size: int = 8
    question_length: int = 6
    target_length: int = 5
    hidden_size: int = 24
    patch_size: int = 4
    local_files_only: bool = True
    device_map: str = ""
    max_memory: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingSpec:
    mode: str = "full"
    observation: str = "gradient"
    task: str = "vqa"
    knowledge: str = "private"
    batch_size: int = 1
    local_steps: int = 1
    lr: float = 0.01
    lora_rank: int = 8
    lora_alpha: int = 16
    clients: int = 10
    clients_per_round: int = 2
    rounds: int = 20
    snapshots: list[int] = field(default_factory=lambda: [0, 10, 20])
    seed: int = 42
    # fnmatch patterns selecting which trainable parameters the client uploads.
    # Empty means the whole trainable set. Resolved against the model at capture
    # time and recorded there, so the wire stays an explicit allowlist.
    upload_parameters: list[str] = field(default_factory=list)


@dataclass
class AttackSpec:
    method: str = "ig_adapted"
    text_method: str = "tag_adapted"
    iterations: int = 100
    max_evaluations: int = 400
    restarts: int = 1
    seconds: float = 3600.0
    seed: int = 0
    lr: float = 0.1
    text_lr: float = 0.15
    tv: float = 0.001
    text_prior: float = 0.01
    prior_model: str = ""
    prior_revision: str = ""
    prior_interval: int = 10
    checkpoint_interval: int = 10
    april_weight: float = 0.1
    gradvit_prior_model: str = "resnet50"
    gradvit_prior_checkpoint: str = ""
    gradvit_prior_weight: float = 0.0001
    patch_weight: float = 0.0001
    allow_prior_download: bool = False


@dataclass
class EvalSpec:
    lpips: bool = False
    clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_revision: str = ""
    bootstrap: int = 1000
    seed: int = 42


@dataclass
class Config:
    model: ModelSpec = field(default_factory=ModelSpec)
    training: TrainingSpec = field(default_factory=TrainingSpec)
    attack: AttackSpec = field(default_factory=AttackSpec)
    evaluation: EvalSpec = field(default_factory=EvalSpec)


def validate(cfg: Config) -> Config:
    m, t, a = cfg.model, cfg.training, cfg.attack
    if m.family not in {"tiny", "llava", "blip2", "qwen2_5_vl"}:
        raise ValueError(f"Unknown model family: {m.family}")
    if m.family != "tiny" and not m.revision:
        raise ValueError("A pinned model revision is required; run doctor --resolve-revision")
    if m.dtype not in {"float32", "bfloat16", "float64"}:
        raise ValueError("Use float32, float64 (tiny), or bfloat16; no quantized attack path")
    if m.device_map not in {"", "auto", "balanced"} or (m.family == "tiny" and m.device_map):
        raise ValueError("device_map is auto/balanced for HF models only")
    if m.family != "tiny" and m.dtype == "float64":
        raise ValueError("float64 is only supported for the tiny correctness fixture")
    if t.mode not in {"full", "llm_full", "lora_llm"}:
        raise ValueError(f"Unknown training mode: {t.mode}")
    if t.observation not in {"gradient", "client_delta"}:
        raise ValueError("Only individual gradients and client deltas are supported")
    if t.knowledge not in {"private", "question_known", "text_known"}:
        raise ValueError("Unknown knowledge condition")
    if t.task not in {"vqa", "caption"}:
        raise ValueError("Task must be vqa or caption")
    if t.task == "caption" and t.knowledge == "question_known":
        raise ValueError("Caption has a public task instruction, not a private question")
    for value in [m.image_size, m.question_length, m.target_length, m.hidden_size,
                  m.patch_size, t.batch_size, t.local_steps, t.lora_rank, t.lora_alpha,
                  t.clients, t.clients_per_round, a.iterations, a.max_evaluations,
                  a.restarts, a.checkpoint_interval, a.prior_interval]:
        if value <= 0:
            raise ValueError("Dimensions, counts and intervals must be positive")
    if t.lr <= 0 or a.lr <= 0 or a.text_lr <= 0 or a.seconds <= 0:
        raise ValueError("Learning rates and time budget must be positive")
    if any(not isinstance(p, str) or not p.strip() for p in t.upload_parameters):
        raise ValueError("Upload patterns must be nonempty strings")
    if len(set(t.upload_parameters)) != len(t.upload_parameters):
        raise ValueError("Upload patterns must be unique")
    if t.observation == "gradient" and t.local_steps != 1:
        raise ValueError("gradient observations require local_steps=1")
    if t.clients_per_round > t.clients or t.rounds < 0:
        raise ValueError("Invalid federation size")
    if m.image_size % m.patch_size and m.family == "tiny":
        raise ValueError("Tiny image size must be divisible by patch size")
    if a.text_method not in {"none", "tag_adapted", "lamp_adapted"}:
        raise ValueError("Unknown text method")
    if a.text_method == "none" and t.knowledge != "text_known" and a.method != "random":
        raise ValueError("Private text needs an explicit reconstruction component")
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    config = OmegaConf.structured(Config)
    if path:
        config = OmegaConf.merge(config, OmegaConf.load(path))
    config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides or []))
    return validate(OmegaConf.to_object(config))


def digest(value) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()
