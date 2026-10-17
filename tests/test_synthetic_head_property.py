from collections import Counter

from evaluation.synthetic_head_generation import cases
from evaluation.synthetic_head_report import binary_details


def test_bank_does_not_encode_property_in_modality_counts():
    rows = cases()
    assert Counter((r["property_label"], r["modality"]) for r in rows) == {
        (label, modality): 4 for label in ["head", "non-head"] for modality in ["CT", "MRI", "X-Ray"]}
    for modality in ["CT", "MRI", "X-Ray"]:
        assert {r["seed"] for r in rows if r["modality"] == modality and r["property_label"] == "head"} == {
            r["seed"] for r in rows if r["modality"] == modality and r["property_label"] == "non-head"}


def test_missing_reconstruction_counts_against_full_cohort_accuracy():
    result = binary_details(["head", "non-head"], [None, [1., 0.]], ["head", "non-head"])
    assert result["n"] == 2
    assert result["accuracy"] == result["coverage"] == 0.5
    assert result["recall"]["head"] == 0
    assert result["precision"] is None
    assert result["class_auroc"]["head"] is None


def test_single_class_modality_is_not_reported_as_binary_balanced_accuracy():
    result = binary_details(["non-head"]*4, [[1., 0.]]*4, ["head", "non-head"])
    assert result["accuracy"] == 1
    assert result["balanced_accuracy"] is None
    assert result["class_auroc"]["head"] is None
    assert result["accuracy_wilson_ci95"][0] < 1
