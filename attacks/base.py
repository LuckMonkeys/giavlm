from abc import ABC, abstractmethod
from dataclasses import replace

from attacks.optim.engine import AttackRunner
from core.knowledge import AdversaryKnowledge
from core.types import Observation, Reconstruction


class BaseAttacker(ABC):
    """References are intentionally absent; evaluation owns private truth."""

    def __init__(self, adapter, spec):
        self.adapter, self.spec = adapter, spec

    @abstractmethod
    def attack(self, gradients, batch_info: Observation, knowledge: AdversaryKnowledge,
               *, directory=None, resume=False) -> Reconstruction:
        raise NotImplementedError


class OptimizationAttacker(BaseAttacker):
    def attack(self, gradients, batch_info, knowledge, *, directory=None, resume=False):
        observation = replace(batch_info, tensors=gradients)
        knowledge.validate_observation(observation)
        return AttackRunner(self.adapter, observation, self.spec).run(directory, resume)


class UnimplementedAttacker(BaseAttacker):
    reason = "This method has no validated VLM implementation in this benchmark."

    def attack(self, gradients, batch_info, knowledge, *, directory=None, resume=False):
        knowledge.validate_observation(batch_info)
        return Reconstruction("not_implemented", self.reason)
