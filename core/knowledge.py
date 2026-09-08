"""Public adversary assumptions, never ground-truth data or private paths."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AdversaryKnowledge:
    name: str = "private"
    template_known: bool = False
    server: str = "honest_but_curious"

    def validate(self):
        if self.name not in {"private", "question_known", "text_known"}:
            raise ValueError(f"Unknown knowledge condition: {self.name}")
        if self.template_known or self.server != "honest_but_curious":
            raise NotImplementedError("Template and malicious-server protocols are not implemented")
        return self

    def validate_observation(self, observation):
        self.validate()
        observation.validate()
        if observation.training.knowledge != self.name:
            raise ValueError("Adversary knowledge differs from the captured protocol")
