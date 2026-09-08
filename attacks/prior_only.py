from attacks.base import OptimizationAttacker


class PriorOnlyAttacker(OptimizationAttacker):
    """Uses image/public text priors but never reads observed update values."""
