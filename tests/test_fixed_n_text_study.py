from dataclasses import replace

import pytest
import torch

from core.artifacts import write_json
from evaluation.fixed_n_text_study import (
    CONDITIONS, attack_spec, check_trace, evaluate_all, native_smoke, select_fixed_cohort,
    validate_groups,
)
from tests.test_fp32_multistep_study import fp32_fixture


def test_fixed_four_selection_preserves_original_order_and_client_groups():
    frozen = {"dataset": "slake", "selected": [
        {"sample_id": str(i), "length_bin": b}
        for i, b in enumerate(["33-64"] * 10 + ["17-32"] * 10 + ["1-16"] * 10)]}
    manifest = [{"sample_id": str(i), "client": 0} for i in range(30)]
    selected, rows = select_fixed_cohort(frozen, manifest)
    assert [r["sample_id"] for r in selected] == [str(i) for i in range(30) if i not in (19, 29)]
    assert len(rows) == 28
    for _, batch, steps in CONDITIONS:
        assert batch * steps == 4
    manifest[1]["client"] = 1
    with pytest.raises(ValueError, match="different clients"):
        select_fixed_cohort(frozen, manifest)
    with pytest.raises(ValueError, match="four distinct"):
        validate_groups(rows[:3])
    with pytest.raises(ValueError, match="four distinct"):
        validate_groups([rows[0]] * 4)


@pytest.mark.parametrize("image", ["gt", "random"])
def test_fixed_four_keeps_sgd_search_and_equal_sample_budgets(image):
    spec = attack_spec(image)
    assert spec.max_evaluations == 800000 and spec.seconds == 14400
    assert spec.dager.pooled_updates and spec.dager.rerank_candidates == 0
    assert spec.dager.lm_prior_weight == .01 and not spec.dager.adamw_delta_proposals


@pytest.mark.parametrize("batch,steps", [(1, 4), (2, 2), (4, 1)])
def test_all_three_partitions_match_native_mixed_sgd_and_restore_model(batch, steps):
    torch.manual_seed(175)
    adapter, _, training, names = fp32_fixture()
    adapter.bfloat16()
    adapter.spec = replace(adapter.spec, dtype="bfloat16")
    for p in adapter.trainable().values():
        p.data = p.data.float()
    before = {name: p.detach().clone() for name, p in adapter.named_parameters()}
    report = native_smoke(adapter, replace(training, batch_size=batch, local_steps=steps), names)
    assert report["native_sgd_exact"] and len(report["trace"]) == steps
    for name, p in adapter.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)
    assert all(p.grad is None for p in adapter.parameters())
    # A stale third/fourth forward must fail even if the second step changed.
    if steps == 4:
        trace = report["trace"]
        trace[2]["selected_A_distance_from_initial"] = 0
        with pytest.raises(ValueError, match="Every later step"):
            check_trace({"x": torch.ones(1)}, trace,
                        replace(training, batch_size=batch, local_steps=steps))


def test_fixed_four_evaluation_waits_for_every_committed_reconstruction(tmp_path, monkeypatch):
    monkeypatch.setattr("evaluation.fixed_n_text_study.verify_plan", lambda *args: {})
    write_json(tmp_path / "plan.json", {"jobs": [{"id": "x"}]})
    def forbidden(*args, **kwargs):
        raise AssertionError("Private references read before all reconstructions")
    monkeypatch.setattr("evaluation.federated_text_pilot.evaluate", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_all(tmp_path)
