from collections import Counter

from evaluation.prototype_medical_synthetic import cases


def test_synthetic_bank_has_equal_class_modality_counts_and_isolated_content_splits():
    rows = cases()
    assert len(rows) == len({r["id"] for r in rows}) == len({r["seed"] for r in rows}) == 32
    assert Counter((r["split"], r["label"], r["modality"]) for r in rows) == {
        (s, c, m): n
        for s, n in (("fit", 4), ("calibration", 2), ("audit", 2))
        for c in ("head", "non-head")
        for m in ("CT", "MRI")
    }
    groups = {
        s: {r["group"] for r in rows if r["split"] == s} for s in ("fit", "calibration", "audit")
    }
    assert not (
        groups["fit"] & groups["calibration"]
        or groups["fit"] & groups["audit"]
        or groups["calibration"] & groups["audit"]
    )
    assert all("No external face" in r["prompt"] for r in rows)
