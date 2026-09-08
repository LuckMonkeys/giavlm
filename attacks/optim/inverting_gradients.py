from attacks.base import OptimizationAttacker
from attacks.priors import total_variation

GRADIENT_OBJECTIVE = "cosine"


def regularize(engine, value, images, fraction):
    return value + engine.spec.tv * total_variation(images)


class IGAttacker(OptimizationAttacker):
    """Global cosine, TV and signed image gradients."""
