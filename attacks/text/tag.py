"""TAG-style L2 plus L1 matching component for private VLM token slots."""
from attacks.objectives import matching_loss


def gradient_objective(predicted, observed):
    return matching_loss(predicted, observed, "tag")
