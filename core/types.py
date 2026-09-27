from dataclasses import dataclass, field
from typing import Literal

import torch

from core.config import ModelSpec, TrainingSpec


@dataclass
class Batch:
    images: torch.Tensor
    questions: torch.Tensor
    targets: torch.Tensor

    def slice(self, start: int, stop: int):
        return Batch(self.images[start:stop], self.questions[start:stop], self.targets[start:stop])


OBSERVATION_SCHEMA = 5


@dataclass
class Observation:
    """Allowlisted attacker input with explicitly authorized public fields."""

    model: ModelSpec
    training: TrainingSpec
    tensors: dict[str, torch.Tensor]
    model_fingerprint: str
    public_questions: list[str] = field(default_factory=list)
    public_targets: list[str] = field(default_factory=list)
    public_question_ids: list[list[int]] = field(default_factory=list)
    public_target_ids: list[list[int]] = field(default_factory=list)
    public_question_lengths: list[int] = field(default_factory=list)
    public_target_lengths: list[int] = field(default_factory=list)
    # Present exactly when the knowledge condition makes the image public.
    public_images: torch.Tensor | None = None
    schema_version: int = OBSERVATION_SCHEMA

    @property
    def sample_count(self):
        return self.training.sample_count

    def validate(self):
        from core.config import Config, validate
        validate(Config(model=self.model, training=self.training))
        if self.schema_version != OBSERVATION_SCHEMA:
            raise ValueError(f"Unsupported observation schema v{self.schema_version}; "
                             f"expected schema v{OBSERVATION_SCHEMA}")
        if not self.tensors or any(not torch.isfinite(x).all() for x in self.tensors.values()):
            raise ValueError("Missing or nonfinite observed update")
        training = self.training
        if training.task == "caption" and (self.public_questions or self.public_question_ids):
            raise ValueError("Caption observations cannot expose private question fields")
        for name, public, texts, ids, length in [
                ("question", training.question_public, self.public_questions,
                 self.public_question_ids, self.model.question_length),
                ("target", training.target_public, self.public_targets,
                 self.public_target_ids, self.model.target_length)]:
            if not public:
                if texts or ids:
                    raise ValueError(f"Private {name}s appeared in the public observation")
                continue
            if len(texts) != self.sample_count:
                raise ValueError(f"Missing declared public {name}s")
            if len(ids) != self.sample_count or any(len(row) != length for row in ids):
                raise ValueError(f"Missing or malformed public {name} token slots")
        if not training.image_public:
            if self.public_images is not None:
                raise ValueError("Private images appeared in the public observation")
        else:
            size = self.model.image_size
            images = self.public_images
            if images is None or tuple(images.shape) != (self.sample_count, 3, size, size):
                raise ValueError("Missing or malformed declared public images")
            if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
                raise ValueError("Public images must be finite RGB values in [0, 1]")

        lengths_known = self.training.token_lengths_known
        question_lengths_required = lengths_known and self.training.task == "vqa"
        self._validate_lengths("question", self.public_question_lengths,
                               self.model.question_length, question_lengths_required)
        self._validate_lengths("target", self.public_target_lengths,
                               self.model.target_length, lengths_known)

    def _validate_lengths(self, field, values, slot_length, required):
        if not required:
            if values:
                raise ValueError(f"Unauthorized public {field} token lengths")
            return
        if len(values) != self.sample_count:
            raise ValueError(f"Missing public {field} token lengths")
        if any(type(value) is not int or not 0 <= value < slot_length for value in values):
            raise ValueError(f"Malformed public {field} token lengths")


@dataclass
class Support:
    status: Literal["supported", "not_applicable", "not_implemented"]
    reason: str = ""


@dataclass
class Reconstruction:
    status: str
    reason: str = ""
    images: torch.Tensor | None = None
    questions: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    costs: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
