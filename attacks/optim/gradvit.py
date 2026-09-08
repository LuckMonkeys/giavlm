from attacks.base import OptimizationAttacker
from attacks.priors import patch_prior, total_variation

GRADIENT_OBJECTIVE = "layer_l2"


def regularize(engine, value, images, fraction):
    if fraction >= 0.5:
        engine.prior_evaluations += 1
        value = value * 0.5 + engine.spec.gradvit_prior_weight * engine.image_prior(images)
    patch = engine.adapter.spec.patch_size
    if hasattr(engine.adapter, "backend"):
        patch = engine.adapter.backend.config.vision_config.patch_size
    return value + engine.spec.patch_weight * patch_prior(images, patch) + engine.spec.tv * total_variation(images)


class GradViTAttacker(OptimizationAttacker):
    """Two-stage BN and patch prior adaptation, without registration ensemble."""
