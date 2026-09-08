from defenses.none import NoDefense
from defenses.clipping import ClippingDefense
from defenses.gaussian_dp import GaussianDPDefense
from defenses.topk_sparsify import TopKSparsifyDefense
from defenses.sign_sgd import SignSGDDefense


def create_defense(config):
    options = dict(config)
    name = options.pop("name")
    if name == "none":
        return NoDefense(**options)
    elif name == "clipping":
        return ClippingDefense(**options)
    elif name == "gaussian_dp":
        return GaussianDPDefense(**options)
    elif name == "topk_sparsify":
        return TopKSparsifyDefense(**options)
    elif name == "sign_sgd":
        return SignSGDDefense(**options)
    elif name in {"token_obfuscation", "safe_template"}:
        # Deliberately fails closed rather than degrading like an unimplemented
        # attack. A defense is part of the experimental condition: continuing
        # would emit rows labelled with a defense that was never applied.
        raise NotImplementedError(f"{name} requires a separate pre-training protocol")
    raise ValueError(f"Unknown defense: {name}")


class DefenseFactory:
    create_defense = staticmethod(create_defense)
