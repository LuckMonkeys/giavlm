import torch

from giavlm.artifacts import write_tensors
from giavlm.attacks import AttackRunner
from giavlm.config import AttackSpec, ModelSpec, TrainingSpec
from giavlm.federated import capture
from giavlm.models import build_model
from giavlm.priors import BatchNormPrior, document_prior, patch_prior


def test_image_priors_preserve_autograd():
    image = torch.rand(1, 3, 8, 8, requires_grad=True)
    for value in [document_prior(image), patch_prior(image, 4)]:
        grad, = torch.autograd.grad(value, image, retain_graph=True)
        assert torch.isfinite(grad).all() and grad.norm() > 0


def test_gradvit_external_bn_path_and_schedule(tmp_path):
    from torchvision.models import resnet50
    # Random weights are exclusively a test fixture, never a downloaded/public prior substitute.
    checkpoint = tmp_path / "test-only-random-resnet.safetensors"
    network = resnet50(weights=None)
    write_tensors(checkpoint, network.state_dict())
    del network
    prior = BatchNormPrior(str(checkpoint), "cpu")
    images = torch.rand(1, 3, 8, 8, requires_grad=True)
    value = prior(images)
    grad, = torch.autograd.grad(value, images)
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert not any(p.requires_grad for p in prior.parameters())
    del prior
    adapter = build_model(ModelSpec(), TrainingSpec())
    batch = adapter.batch(images.detach(), ["what color is the object"], ["red"])
    obs = capture(adapter, batch, [], [])
    result = AttackRunner(adapter, obs, AttackSpec(method="gradvit_adapted", iterations=2,
                                                   checkpoint_interval=1,
                                                   gradvit_prior_checkpoint=str(checkpoint))).run()
    assert result.status == "completed"
    assert result.costs["prior_evaluations"] == 1
    assert result.provenance["image_prior"]["sha256"]
