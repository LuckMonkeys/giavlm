from defenses.base import BaseDefense


class TokenObfuscationDefense(BaseDefense):
    def apply(self, updates, *, generator=None):
        raise NotImplementedError("Token obfuscation must be applied before loss construction")
