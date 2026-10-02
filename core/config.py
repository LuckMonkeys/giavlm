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
# Each named condition is a set of public input fields; everything else is private.
KNOWLEDGE_FIELDS = {
    "private": frozenset(),
    "question_known": frozenset({"question"}),
    "text_known": frozenset({"question", "target"}),
    "image_known": frozenset({"image"}),
    "image_question_known": frozenset({"image", "question"}),
}
KNOWLEDGE_CONDITIONS = tuple(KNOWLEDGE_FIELDS)


def caption_condition(knowledge):
    """A known question without a known target only duplicates another caption condition."""
    fields = KNOWLEDGE_FIELDS[knowledge]
    return "question" not in fields or "target" in fields
TASK_TYPES = ("vqa", "caption")
TEXT_METHODS = ("none", "tag_adapted", "lamp_adapted")
# private_reference derives the start from the victim's own image: diagnostics only.
INIT_SOURCES = ("random", "public_image", "private_reference")
INIT_PERTURBATIONS = ("none", "uniform_mix", "gaussian", "blur")
INIT_TEXT_SOURCES = ("random", "public_text", "private_reference")
INIT_TEXT_PERTURBATIONS = ("none", "replace")
CANDIDATE_IMAGE_DTYPES = ("float32", "float64", "bfloat16", "float16")


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

    # Public/private status of each input field under the knowledge condition.
    @property
    def image_public(self) -> bool:
        return "image" in KNOWLEDGE_FIELDS[self.knowledge]

    @property
    def question_public(self) -> bool:
        """Captioning has no private question slot, so this is VQA-only."""
        return self.task == "vqa" and "question" in KNOWLEDGE_FIELDS[self.knowledge]

    @property
    def question_private(self) -> bool:
        return self.task == "vqa" and not self.question_public

    @property
    def target_public(self) -> bool:
        return "target" in KNOWLEDGE_FIELDS[self.knowledge]

    @property
    def private_text(self) -> bool:
        return self.question_private or not self.target_public

    @property
    def fine_tuning_stage(self) -> str:
        """Active parameter group for the next client update."""
        if self.fine_tuning_strategy == "f_2stage":
            return ("connector" if self.server_round < self.two_stage_connector_rounds
                    else "llm")
        return {"f_c": "connector", "f_l": "llm", "f_cl": "joint"}[
            self.fine_tuning_strategy]


@dataclass
class DAGEROptions:
    mode: str = "public_residual"
    projections: str = "qkv"
    analysis_dtype: str = "float64"
    rank_rtol: float = 1e-5
    rank_atol: float = 1e-8
    zero_tolerance: float = 1e-7
    token_selection: str = "threshold"
    token_threshold: float = 0.05
    max_candidates: int = 256
    beam_width: int = 16
    vocab_chunk_size: int = 1024
    prefix_batch_size: int = 16
    rerank_candidates: int = 0


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
    # Candidate initialization; see docs/protocol.md#Attack Initialization.
    init_source: str = "random"
    init_images: str = ""
    init_perturbation: str = "none"
    init_level: float = 0.0
    init_text_source: str = "random"
    init_question: str = ""
    init_target: str = ""
    init_text_perturbation: str = "none"
    init_text_level: float = 0.0
    init_text_scale: float = 10.0
    # Precision of the optimized candidate images, independent of the victim model dtype.
    image_dtype: str = "float32"
    dager: DAGEROptions = field(default_factory=DAGEROptions)


@dataclass
class EvalSpec:
    lpips: bool = False
    clip: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_revision: str = ""
    bootstrap: int = 1000
    seed: int = 42
    trajectory: bool = False


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
    if attack.method == "dager_adapted":
        import math
        options = attack.dager
        if not math.isfinite(attack.seconds):
            raise ValueError("DAGER seconds must be finite")
        _require_choice("DAGER mode", options.mode, ("raw", "public_residual"))
        _require_choice("DAGER projections", options.projections, ("q", "k", "v", "qkv"))
        _require_choice("DAGER analysis dtype", options.analysis_dtype, ("float32", "float64"))
        _require_choice("DAGER token selection", options.token_selection, ("threshold", "topk"))
        for name in ("rank_rtol", "rank_atol", "zero_tolerance", "token_threshold"):
            value = getattr(options, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"DAGER {name} must be finite and nonnegative")
        if not 0 < options.zero_tolerance < 1 or options.token_threshold > 1:
            raise ValueError("DAGER tolerances must describe normalized distances")
        for name in ("max_candidates", "beam_width", "vocab_chunk_size", "prefix_batch_size"):
            value = getattr(options, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"DAGER {name} must be a positive integer")
        if type(options.rerank_candidates) is not int or options.rerank_candidates < 0:
            raise ValueError("DAGER rerank_candidates must be a nonnegative integer")
        if attack.text_method != "none" or attack.restarts != 1:
            raise ValueError("DAGER owns text search and requires text_method=none, restarts=1")
        if attack.init_source != "random" or attack.init_text_source != "random":
            raise ValueError("DAGER uses declared public inputs, not candidate initialization")
        _require_positive("attack.checkpoint_interval", attack.checkpoint_interval)
    _require_choice("attack init source", attack.init_source, INIT_SOURCES)
    _require_choice("attack init perturbation", attack.init_perturbation, INIT_PERTURBATIONS)
    if attack.init_source == "random":
        if attack.init_images or attack.init_perturbation != "none":
            raise ValueError("Random initialization takes no init_images or init_perturbation")
    elif attack.init_source == "public_image" and not attack.init_images:
        raise ValueError("attack.init_source=public_image requires attack.init_images")
    if attack.init_perturbation == "none" and attack.init_level != 0:
        raise ValueError("attack.init_level requires an init_perturbation")
    if attack.init_perturbation == "uniform_mix" and not 0 <= attack.init_level <= 1:
        raise ValueError("uniform_mix init_level must lie in [0, 1]")
    if attack.init_perturbation in {"gaussian", "blur"} and attack.init_level <= 0:
        raise ValueError(f"{attack.init_perturbation} init_level must be positive")
    _require_choice("attack image dtype", attack.image_dtype, CANDIDATE_IMAGE_DTYPES)
    _require_choice("attack init text source", attack.init_text_source, INIT_TEXT_SOURCES)
    _require_choice("attack init text perturbation", attack.init_text_perturbation,
                    INIT_TEXT_PERTURBATIONS)
    _require_positive("attack.init_text_scale", attack.init_text_scale)
    templates = attack.init_question or attack.init_target
    if attack.init_text_source == "random":
        if templates or attack.init_text_perturbation != "none":
            raise ValueError("Random text initialization takes no template or perturbation")
    elif attack.init_text_source == "public_text" and not templates:
        raise ValueError("attack.init_text_source=public_text requires init_question or init_target")
    elif attack.init_text_source == "private_reference" and templates:
        raise ValueError("private_reference text comes from the capture, not from templates")
    if attack.init_text_perturbation == "none" and attack.init_text_level != 0:
        raise ValueError("attack.init_text_level requires an init_text_perturbation")
    if attack.init_text_perturbation == "replace" and not 0 < attack.init_text_level <= 1:
        raise ValueError("replace init_text_level must lie in (0, 1]")


def validate_protocol_compatibility(cfg: Config) -> None:
    """Reject cross-component combinations with ambiguous research semantics."""
    training, attack = cfg.training, cfg.attack
    if training.task == "caption" and not caption_condition(training.knowledge):
        raise ValueError("Caption has a public task instruction, not a private question")
    from attacks.registry import SELF_CONTAINED_TEXT_METHODS
    if (attack.text_method == "none" and training.private_text
            and attack.method not in SELF_CONTAINED_TEXT_METHODS):
        raise ValueError("Private text needs an explicit reconstruction component")
    # Initialization applies only to fields the attacker optimizes.
    if attack.init_source != "random" and training.image_public:
        raise ValueError("Image initialization needs a private image; the image is public here")
    if attack.init_text_source != "random" and not training.private_text:
        raise ValueError("Text initialization needs private text; all text is public here")
    if attack.init_question and not training.question_private:
        raise ValueError("attack.init_question needs a private VQA question")
    if attack.init_target and training.target_public:
        raise ValueError("attack.init_target needs a private target")


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
