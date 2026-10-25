from dataclasses import replace

import pytest
import torch

from attacks.analytic.dager_adapted import DAGERSearch, dager_support
from attacks.analytic.dager_pooled import PooledTextView
from core.fl import capture, simulate_update
from defenses.factory import create_defense
from tests.test_dager import fixture, run, specification


def pooled_fixture(algorithm="fedsgd", steps=1):
    adapter, _, _ = fixture(knowledge="image_known")
    training = replace(adapter.training_spec, batch_size=2 // steps, local_steps=steps,
                       algorithm=algorithm)
    adapter.set_training_spec(training)
    batch = adapter.batch(torch.rand(2, 3, 8, 8, dtype=torch.float64),
                          ["what color", "what"], ["red circle", "blue"])
    obs = capture(adapter, batch, [], [])
    return adapter, batch, obs


def test_pooled_view_resets_context_and_retains_all_public_inputs():
    adapter, batch, obs = pooled_fixture()
    view = PooledTextView(adapter, obs)
    forbidden = {adapter.eos, adapter.pad}
    rows = [[token for field in (batch.questions, batch.targets)
             for token in field[i].tolist() if token not in forbidden] for i in range(2)]
    prefix = rows[0] + rows[1]
    ids = view.token_ids(prefix)
    torch.testing.assert_close(ids["questions"], batch.questions)
    torch.testing.assert_close(ids["targets"], batch.targets)
    torch.testing.assert_close(view.prefix_features([rows[0] + rows[1][:1]]),
                               view.views[1].prefix_features([rows[1][:1]]))
    nll, _ = view.text_prior_nll([rows[0]], [4, 5])
    expected, _ = view.views[1].text_prior_nll([[]], [4, 5])
    torch.testing.assert_close(nll, expected)
    assert len(view.public_layer0) == sum(len(v.public_layer0) for v in view.views)
    assert len(view.slots) == sum(map(len, rows))


@pytest.mark.parametrize("algorithm,steps", [("fedsgd", 1), ("fedavg", 1), ("fedavg", 2)])
def test_pooled_upload_search_is_explicit_and_does_not_mutate_model(tmp_path, algorithm, steps):
    adapter, _, obs = pooled_fixture(algorithm, steps)
    before = {name: tensor.detach().clone() for name, tensor in adapter.trainable().items()}
    assert dager_support(adapter, obs).status == "not_applicable"
    spec = specification(pooled_updates=True, token_selection="topk", max_candidates=8,
                         beam_width=2, lm_prior_weight=.01)
    result = run(adapter, obs, spec, directory=tmp_path)
    assert result.status == "completed"
    assert len(result.questions) == len(result.targets) == 2
    assert result.provenance["pooled_search"]
    for name, tensor in adapter.trainable().items():
        torch.testing.assert_close(tensor, before[name])
    adam = replace(obs, training=replace(obs.training, local_optimizer="adamw"))
    assert dager_support(adapter, adam, spec.dager).status == "not_applicable"


def test_clipping_invariance_and_explicit_defense_mode(tmp_path):
    adapter, _, obs = fixture(knowledge="image_known")
    norm = sum(g.square().sum() for g in obs.tensors.values()).sqrt().item()
    defense = {"name": "clipping", "max_norm": norm / 2}
    clipped = replace(obs, tensors=create_defense(defense).apply(obs.tensors))
    spec = specification(defense_mode="unaware")
    first = DAGERSearch(adapter, obs, spec, None)
    second = DAGERSearch(adapter, clipped, spec, None)
    first.prepare()
    second.prepare()
    for a, b in zip(first.spans, second.spans, strict=True):
        # SVD of nearly rank-deficient blocks amplifies scalar-rounding error.
        torch.testing.assert_close(a.basis.T @ a.basis, b.basis.T @ b.basis, atol=1e-8, rtol=1e-8)
    assert run(adapter, clipped, specification(), upload_metadata={"defense": defense}).status == "not_applicable"
    result = run(adapter, clipped, spec, directory=tmp_path, upload_metadata={"defense": defense})
    assert result.status == "completed" and result.provenance["upload_defense"] == defense


def test_noise_is_seeded_per_tensor_and_does_not_modify_input():
    values = {"a": torch.tensor([0., 2., -1.]), "zero": torch.zeros(4)}
    defense = create_defense({"name": "gaussian_noise", "noise_ratio": .1})
    actual = defense.apply(values, generator=torch.Generator().manual_seed(42))
    expected = torch.randn(3, generator=torch.Generator().manual_seed(42)) * .2 + values["a"]
    torch.testing.assert_close(actual["a"], expected)
    assert actual["zero"].count_nonzero() == 0
    assert values["a"].tolist() == [0., 2., -1.]
    with pytest.raises(ValueError):
        create_defense({"name": "gaussian_noise", "noise_ratio": float("nan")})


def test_one_step_delta_has_same_subspace_as_gradient():
    adapter, batch, obs = pooled_fixture()
    delta = simulate_update(adapter, batch, replace(obs.training, algorithm="fedavg"))
    for name in obs.tensors:
        torch.testing.assert_close(delta[name], -obs.training.lr * obs.tensors[name], atol=1e-12, rtol=1e-8)
