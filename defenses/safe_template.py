from defenses.base import BaseDefense


class SafeTemplateDefense(BaseDefense):
    def apply(self, updates, *, generator=None):
        raise NotImplementedError("Safe-template defense requires an explicit document protocol")
