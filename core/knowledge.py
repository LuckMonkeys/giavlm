"""Public adversary assumptions, never ground-truth data or private paths."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AdversaryKnowledge:
    name: str = "private"
    token_lengths_known: bool = False
    template_known: bool = False
    server: str = "honest_but_curious"

    def validate(self):
        from core.config import KNOWLEDGE_CONDITIONS
        if self.name not in KNOWLEDGE_CONDITIONS:
            raise ValueError(f"Unknown knowledge condition: {self.name}")
        if not isinstance(self.token_lengths_known, bool):
            raise ValueError("token_lengths_known must be boolean")
        if self.template_known or self.server != "honest_but_curious":
            raise NotImplementedError("Template and malicious-server protocols are not implemented")
        return self

    def validate_observation(self, observation):
        self.validate()
        observation.validate()
        if observation.training.knowledge != self.name:
            raise ValueError("Adversary knowledge differs from the captured protocol")
        if observation.training.token_lengths_known != self.token_lengths_known:
            raise ValueError("Adversary token-length knowledge differs from the captured protocol")
