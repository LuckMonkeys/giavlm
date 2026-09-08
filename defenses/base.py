from abc import ABC, abstractmethod
import torch


class BaseDefense(ABC):
    @abstractmethod
    def apply(self, updates, *, generator=None):
        raise NotImplementedError

    @staticmethod
    def validate(updates):
        if not updates or any(not torch.isfinite(t).all() for t in updates.values()):
            raise ValueError("Defense requires finite, nonempty named updates")
