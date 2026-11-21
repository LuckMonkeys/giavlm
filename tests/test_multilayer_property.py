import numpy as np

from evaluation.multilayer_property import LAYERS, complementarity, freeze_rule, fuse
from evaluation.multilayer_property_data import abdomen_cases
from evaluation.multilayer_property_gpu import abdomen_family, public_negatives
from evaluation.multilayer_property_runner import select_gpu


def test_fusion_margin_direction_and_vote_tie_follow_public_single_layer():
    scores = {f"gradient_{layer}": [[0., 1.], [1., 0.]] for layer in LAYERS}
    scores["gradient_8"] = [[1., 0.], [0., 1.]]
    rule = {"layers": [8, 24], "single_layer": 24, "rms_scales": {"8": 1., "24": 1.}}
    result = fuse(scores, rule)
    assert np.argmin(result["majority"], axis=1).tolist() == [0, 1]
    assert np.argmin(result["single_selected"], axis=1).tolist() == [0, 1]


def test_public_rms_fusion_is_invariant_to_layer_specific_positive_scaling():
    scores = {f"gradient_{layer}": [[0., 1.], [2., 0.]] for layer in LAYERS}
    public = {"fold": {"scores": scores}}
    metrics = {"fold": {f"gradient_{layer}": {"balanced_accuracy": .9,
                "class_auroc": {"head": 1.}} for layer in LAYERS}}
    rule = freeze_rule(public, metrics)
    first = fuse(scores, rule)["mean_standardized"]
    scores["gradient_8"] = (np.asarray(scores["gradient_8"]) * 100).tolist()
    rule = freeze_rule(public, metrics)
    np.testing.assert_allclose(first, fuse(scores, rule)["mean_standardized"])


def test_complementarity_counts_distinct_error_rescues():
    scores = {"gradient_8": [[0., 1.], [0., 1.]], "gradient_24": [[1., 0.], [1., 0.]]}
    result = complementarity(["head", "non-head"], scores, [8, 24], ["head", "non-head"])
    assert result["any_layer_correct"] == 2
    assert result["pairs"][0]["first_only_correct"] == 1
    assert result["pairs"][0]["second_only_correct"] == 1


def test_abdomen_generation_is_nonmedical_and_content_group_disjoint():
    rows = abdomen_cases()
    assert len(rows) == 48
    for split in ["fit", "dev"]:
        for style in ["photo", "line", "render"]:
            assert sum(row["split"] == split and row["style"] == style for row in rows) == 8
    fit = {row["group"] for row in rows if row["split"] == "fit"}
    dev = {row["group"] for row in rows if row["split"] == "dev"}
    assert fit.isdisjoint(dev)
    assert all("Fully clothed" in row["prompt"] for row in rows)
    assert not any(term in row["prompt"].lower() for row in rows for term in ["medical", "mri", "x-ray", "ct scan"])


def test_local_gpu_selection_excludes_busy_cards_and_honors_requested_index():
    import pytest
    rows = [{"index": 0, "uuid": "a", "used_mib": 0, "free_mib": 80000, "utilization": 100},
            {"index": 1, "uuid": "b", "used_mib": 0, "free_mib": 80000, "utilization": 0},
            {"index": 2, "uuid": "c", "used_mib": 20000, "free_mib": 60000, "utilization": 0}]
    assert select_gpu(rows)["index"] == 1
    with pytest.raises(RuntimeError, match="No idle GPU"):
        select_gpu(rows, 0)
    with pytest.raises(RuntimeError, match="No idle GPU"):
        select_gpu(rows, 2)


def test_abdomen_public_negatives_include_heads_and_limbs_and_no_shared_split_groups():
    rows = [{"group": f"{label}-{index:02d}", "split": "fit" if index < 8 else "dev", "style": style}
            for label in ["head", "non-head"] for index in range(16) for style in ["photo", "line", "render"]]
    selected = public_negatives(rows)
    assert len(selected) == 48
    for split in ["fit", "dev"]:
        assert sum(row["split"] == split and row["group"].startswith("head-") for row in selected) == 12
        assert sum(row["split"] == split and row["group"].startswith("non-head-") for row in selected) == 12


def test_abdomen_repeated_prompt_families_stay_together_across_seeds_and_styles():
    rows = abdomen_cases()
    for row in rows:
        counterpart = next(other for other in rows if other["id"] ==
                           f"abdomen-{(int(row['id'].split('-')[1]) + 8) % 16:02d}-{row['style']}")
        assert row["prompt"] == counterpart["prompt"]
        assert abdomen_family(row) == abdomen_family(counterpart)
    fit = {abdomen_family(row)[0] for row in rows if abdomen_family(row)[1] == "fit"}
    dev = {abdomen_family(row)[0] for row in rows if abdomen_family(row)[1] == "dev"}
    assert len(fit) == len(dev) == 4 and fit.isdisjoint(dev)
