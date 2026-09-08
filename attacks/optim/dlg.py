from attacks.base import OptimizationAttacker

GRADIENT_OBJECTIVE = "l2"


class DLGAttacker(OptimizationAttacker):
    """Squared-update matching with L-BFGS; joint VLM adaptation."""
