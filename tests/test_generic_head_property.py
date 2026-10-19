from collections import Counter
import re

import pytest

from core.artifacts import file_hash
from evaluation.generic_head_generation import cases
from evaluation.generic_head_property import verify_hashes
from evaluation.generic_head_report import clustered_interval


def test_public_banks_have_matched_seeds_and_no_medical_prompts():
    rows = cases()
    assert len(rows) == 72
    assert len({(r["case"], r["seed"]) for r in rows}) == 72
    bank_seeds = []
    for b in range(3):
        bank = [r for r in rows if r["bank"] == b]
        assert Counter(r["property_label"] for r in bank) == {"head": 12, "non-head": 12}
        assert set(Counter(r["region"] for r in bank).values()) == {4}
        positive = {r["seed"] for r in bank if r["property_label"] == "head"}
        negative = {r["seed"] for r in bank if r["property_label"] == "non-head"}
        assert positive == negative
        bank_seeds.append(positive)
    assert len(set.union(*bank_seeds)) == 12
    for row in rows:
        assert not re.search(r"\b(?:CT|MRI|X-Ray)\b|医学|医疗|扫描|断层|放射|磁共振", row["prompt"], re.I)


def test_reused_input_mutation_stops_scoring(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"score": 1}')
    hashes = {str(path): file_hash(path)}
    verify_hashes(hashes)
    path.write_text('{"score": 2}')
    with pytest.raises(ValueError, match="Frozen input changed"):
        verify_hashes(hashes)


def test_duplicate_banks_do_not_artificially_narrow_target_interval():
    labels = ["head"] * 4 + ["non-head"] * 4
    credit = [1, 0, 1, 1, 0, 1, 0, 1]
    one = clustered_interval([credit], labels)
    three = clustered_interval([credit] * 3, labels)
    assert one["ci95"] == three["ci95"]
    assert three["independent_images"] == 8
    paired = clustered_interval([credit] * 3, labels, credit)
    assert paired["ci95"] == [0, 0]
