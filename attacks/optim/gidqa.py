import math
from attacks.base import OptimizationAttacker
from attacks.priors import document_prior

GRADIENT_OBJECTIVE = "combined"


def regularize(engine, value, images, fraction):
    return value + engine.spec.tv * (1 + math.cos(math.pi * fraction)) / 2 * document_prior(images)


class GIDQAAttacker(OptimizationAttacker):
    """Template-free document prior adaptation, not the original GI-DQA protocol."""
