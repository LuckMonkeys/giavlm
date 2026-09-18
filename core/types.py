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


@dataclass
class Observation:
    """Allowlisted attacker input. No sample IDs, paths, private masks or lengths."""

    model: ModelSpec
    training: TrainingSpec
    tensors: dict[str, torch.Tensor]
    model_fingerprint: str
    public_questions: list[str] = field(default_factory=list)
    public_targets: list[str] = field(default_factory=list)
    public_question_ids: list[list[int]] = field(default_factory=list)
    public_target_ids: list[list[int]] = field(default_factory=list)
    schema_version: int = 3

    @property
    def sample_count(self):
        return self.training.sample_count

    def validate(self):
        from core.config import Config, validate
        validate(Config(model=self.model, training=self.training))
        if self.schema_version != 3:
            raise ValueError(
                f"Unsupported observation schema v{self.schema_version}; expected schema v3")
        if not self.tensors or any(not torch.isfinite(x).all() for x in self.tensors.values()):
            raise ValueError("Missing or nonfinite observed update")
        if self.training.knowledge == "private" and (self.public_questions or self.public_targets
                                                      or self.public_question_ids or self.public_target_ids):
            raise ValueError("Private text appeared in the public observation")
        if self.training.knowledge == "question_known" and (self.public_targets or self.public_target_ids):
            raise ValueError("Private targets appeared in the public observation")
        if self.training.task == "caption" and (self.public_questions or self.public_question_ids):
            raise ValueError("Caption observations cannot expose private question fields")
        if self.training.knowledge != "private" and self.training.task == "vqa":
            if len(self.public_questions) != self.sample_count:
                raise ValueError("Missing declared public questions")
            if len(self.public_question_ids) != self.sample_count or any(
                    len(row) != self.model.question_length for row in self.public_question_ids):
                raise ValueError("Missing or malformed public question token slots")
        if self.training.knowledge == "text_known" and len(self.public_targets) != self.sample_count:
            raise ValueError("Missing declared public targets")
        if self.training.knowledge == "text_known" and (len(self.public_target_ids) != self.sample_count or any(
                len(row) != self.model.target_length for row in self.public_target_ids)):
            raise ValueError("Missing or malformed public target token slots")


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
