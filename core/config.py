from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


# Configuration vocabulary. Tuples keep the supported surface ordered and easy
# to review; validation and runtime consumers must reuse these definitions.
MODEL_FAMILIES = ("tiny", "llava", "blip2", "qwen2_5_vl")
MODEL_DTYPES = ("float32", "bfloat16", "float64")
MODEL_DEVICE_MAPS = ("", "auto", "balanced")
FINE_TUNING_STRATEGIES = ("f_c", "f_l", "f_cl", "f_2stage")
FEDERATED_ALGORITHMS = ("fedsgd", "fedavg")
LOCAL_OPTIMIZERS = ("sgd", "adamw")
TRAINING_PROTOCOLS = ("native-sft-v2",)
KNOWLEDGE_CONDITIONS = ("private", "question_known", "text_known")
TASK_TYPES = ("vqa", "caption")
TEXT_METHODS = ("none", "tag_adapted", "lamp_adapted")


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
    fine_tuning_strategy: str = "f_l"
    algorithm: str = "fedsgd"
    task: str = "vqa"
    knowledge: str = "private"
    token_lengths_known: bool = False
    batch_size: int = 1
    local_steps: int = 1
    gradient_accumulation_steps: int = 1
    lr: float = 0.01
    local_optimizer: str = "sgd"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    training_protocol: str = "native-sft-v2"
    lora_rank: int = 8
    lora_alpha: int = 16
    clients: int = 10
    clients_per_round: int = 2
    rounds: int = 20
    server_round: int = 0
    two_stage_connector_rounds: int = 1
    snapshots: list[int] = field(default_factory=lambda: [0, 10, 20])
    seed: int = 42
    # fnmatch patterns selecting which trainable parameters the client uploads.
    # Empty means the whole trainable set. Resolved against the model at capture
    # time and recorded there, so the wire stays an explicit allowlist.
    upload_parameters: list[str] = field(default_factory=list)

    @property
    def sample_count(self) -> int:
        """Number of examples consumed by one client update."""
        return self.batch_size * self.local_steps * self.gradient_accumulation_steps

    @property
    def fine_tuning_stage(self) -> str:
        """Active parameter group for the next client update."""
        if self.fine_tuning_strategy == "f_2stage":
            return ("connector" if self.server_round < self.two_stage_connector_rounds
                    else "llm")
        return {"f_c": "connector", "f_l": "llm", "f_cl": "joint"}[
            self.fine_tuning_strategy]


@dataclass
class AttackSpec:
    method: str = "ig_adapted"
    text_method: str = "tag_adapted"
    iterations: int = 100
    max_evaluations: int | None = None
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


def _require_choice(field, value, choices):
    if value not in choices:
        raise ValueError(f"Unknown {field}: {value}")


def _require_positive(field, value):
    if value <= 0:
        raise ValueError(f"{field} must be positive, got {value}")


def validate_model_config(model: ModelSpec) -> None:
    """Validate only the model identity needed to select an adapter."""
    _require_choice("model family", model.family, MODEL_FAMILIES)
    if model.family != "tiny" and not model.revision:
        raise ValueError("A pinned model revision is required; run doctor --resolve-revision")


def validate_training_config(training: TrainingSpec) -> None:
    """Validate the minimal local/federated training protocol."""
    _require_choice("fine-tuning strategy", training.fine_tuning_strategy,
                    FINE_TUNING_STRATEGIES)
    _require_choice("federated algorithm", training.algorithm, FEDERATED_ALGORITHMS)
    _require_choice("local optimizer", training.local_optimizer, LOCAL_OPTIMIZERS)
    _require_choice("training protocol", training.training_protocol, TRAINING_PROTOCOLS)
    _require_choice("knowledge condition", training.knowledge, KNOWLEDGE_CONDITIONS)
    _require_choice("task", training.task, TASK_TYPES)
    if not isinstance(training.token_lengths_known, bool):
        raise ValueError("training.token_lengths_known must be boolean")
    for name in ["batch_size", "local_steps", "gradient_accumulation_steps",
                 "clients", "clients_per_round"]:
        _require_positive(f"training.{name}", getattr(training, name))
    _require_positive("training.lr", training.lr)
    _require_positive("training.adam_epsilon", training.adam_epsilon)
    if training.weight_decay < 0:
        raise ValueError("training.weight_decay must be nonnegative")
    if not 0 <= training.adam_beta1 < 1 or not 0 <= training.adam_beta2 < 1:
        raise ValueError("Adam betas must be in [0, 1)")
    if training.rounds < 0 or training.server_round < 0:
        raise ValueError("Training rounds must be nonnegative")
    if training.two_stage_connector_rounds <= 0:
        raise ValueError("training.two_stage_connector_rounds must be positive")
    if training.clients_per_round > training.clients:
        raise ValueError("training.clients_per_round cannot exceed training.clients")
    if training.algorithm == "fedsgd" and (
            training.local_optimizer != "sgd" or training.local_steps != 1
            or training.gradient_accumulation_steps != 1 or training.weight_decay != 0):
        raise ValueError(
            "FedSGD requires local_optimizer=sgd, local_steps=1, "
            "gradient_accumulation_steps=1, and weight_decay=0")


def validate_attack_config(attack: AttackSpec) -> None:
    """Validate only controls required by every optimization run."""
    _require_choice("text method", attack.text_method, TEXT_METHODS)
    for name in ["iterations", "restarts"]:
        _require_positive(f"attack.{name}", getattr(attack, name))
    if attack.max_evaluations is not None:
        _require_positive("attack.max_evaluations", attack.max_evaluations)
    _require_positive("attack.seconds", attack.seconds)


def validate_protocol_compatibility(cfg: Config) -> None:
    """Reject cross-component combinations with ambiguous research semantics."""
    training, attack = cfg.training, cfg.attack
    if training.task == "caption" and training.knowledge == "question_known":
        raise ValueError("Caption has a public task instruction, not a private question")
    if attack.text_method == "none" and training.knowledge != "text_known" and attack.method != "random":
        raise ValueError("Private text needs an explicit reconstruction component")


def validate(cfg: Config) -> Config:
    """Validate the small set of invariants shared by all experiment paths."""
    validate_model_config(cfg.model)
    validate_training_config(cfg.training)
    validate_attack_config(cfg.attack)
    validate_protocol_compatibility(cfg)
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
