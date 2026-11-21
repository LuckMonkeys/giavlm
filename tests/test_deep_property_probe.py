import torch

from evaluation.deep_property_probe import distance_to_gradient


def test_two_distinct_prototypes_use_the_actual_gradient_coordinates():
    prototypes = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    gradient = torch.tensor([[2., 0., 0.], [-4., 0., 0.]], dtype=torch.float64)
    assert distance_to_gradient(prototypes, gradient) == [0., 1.]
    assert distance_to_gradient(prototypes, -3 * gradient) == [0., 1.]
    # Opposite vectors cannot distinguish a class by distance to a linear span.
    opposite = torch.stack([prototypes[0], -prototypes[0]])
    assert distance_to_gradient(opposite, gradient) == [0., 0.]
