from defenses.base import BaseDefense


class NoDefense(BaseDefense):
    def apply(self, updates, *, generator=None):
        self.validate(updates)
        return {name: tensor.clone() for name, tensor in updates.items()}
