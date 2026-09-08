from defenses.base import BaseDefense


class SignSGDDefense(BaseDefense):
    def apply(self, updates, *, generator=None):
        self.validate(updates)
        return {name: value.sign() for name, value in updates.items()}
