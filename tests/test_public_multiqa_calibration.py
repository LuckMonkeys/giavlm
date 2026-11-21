import numpy as np
import re

from evaluation.public_multiqa_calibration import (
    QA, balanced_weights, calibration_split, fit_linear, fit_rules, linear_margin, original_score_form,
)


def test_content_family_repeats_and_class_imbalance_do_not_change_total_weight():
    weights = balanced_weights([True, True, True, True, False], ["a", "a", "a", "b", "c"])
    np.testing.assert_allclose([weights[:3].sum(), weights[3], weights[4]], [.25, .25, .5])


def test_ridge_unchanged_by_repeated_identical_qa_or_affine_feature_units():
    x = np.random.default_rng(42).normal(size=(12, 4))
    labels = np.arange(12) % 2 == 0
    groups = [f"g{i}" for i in range(12)]
    original = fit_linear(x, labels, groups)
    repeated = fit_linear(np.repeat(x, 4, axis=0), np.repeat(labels, 4), np.repeat(groups, 4))
    np.testing.assert_allclose(linear_margin(x, original), linear_margin(x, repeated), atol=1e-12)
    changed = fit_linear(x * [2., 10., .3, 4.] + 17., labels, groups)
    np.testing.assert_allclose(linear_margin(x, original),
                               linear_margin(x * [2., 10., .3, 4.] + 17., changed), atol=1e-12)


def test_audit_qa_never_enters_calibration_rules():
    bank = [{"group": f"family-{i:02d}", "calibration_split": "fit", "labels": {"head": label}}
            for i, label in enumerate(["head", "non-head", "head", "non-head"])]
    records = []
    for i in range(4):
        for qa in QA:
            value = float(i % 2)
            records.append({"image": i, "qa": qa["id"], "scores": {"head": {
                str(layer): [value + layer * .001, 1 - value] for layer in (1, 8, 16, 24, 32)}}})
    # fit_rules also expects Abdomen; reuse a separate consistent property task.
    for row in bank:
        row["labels"]["abdomen"] = "abdomen" if row["labels"]["head"] == "head" else "non-abdomen"
    for row in records:
        row["scores"]["abdomen"] = dict(row["scores"]["head"])
    before = fit_rules(records, bank, {"qa": QA, "ridge": .1})
    for row in records:
        if row["qa"] in {qa["id"] for qa in QA if qa["split"] == "audit"}:
            row["scores"] = {prop: {str(layer): [999., -999.] for layer in (1, 8, 16, 24, 32)}
                             for prop in ("head", "abdomen")}
    assert before == fit_rules(records, bank, {"qa": QA, "ridge": .1})


def test_shared_styles_and_abdomen_seeds_follow_one_content_split():
    assert calibration_split("head-08") == "fit"
    assert calibration_split("head-09") == "audit"
    assert calibration_split("abdomen-prompt-family-04") == "fit"
    assert calibration_split("abdomen-prompt-family-05") == "audit"
    forbidden = ("head", "abdomen", "ct", "mri", "x-ray")
    assert all(re.search(r"\b" + re.escape(term) + r"\b", qa["q"] + " " + qa["a"], re.I) is None
               for qa in QA for term in forbidden)


def test_raw_distance_and_original_signed_margin_forms_preserve_decisions_and_ranks():
    from evaluation.image_property_scoring import classification_stats
    raw = [[.1, .8], [.7, .2], [.4, .4], [.2, .6]]
    canonical = [original_score_form(row, "original_32") for row in raw]
    np.testing.assert_allclose(canonical[0], [-.7, .7], atol=1e-15)
    assert classification_stats(["h", "n", "h", "n"], raw, ["h", "n"]) == classification_stats(
        ["h", "n", "h", "n"], canonical, ["h", "n"])
    assert original_score_form([-.25, .25], "original_fusion") == [-.25, .25]
