import pytest

from evaluation.fixed_qa_property import ANSWER, QUESTION, align_runs, substitute_qa


def test_fixed_qa_replaces_only_text_without_mutating_original_row():
    original = {"image": "opaque.png", "image_id": "private-group", "question": "original q",
                "target": "original a", "task": "vqa"}
    fixed = substitute_qa(original)
    assert fixed["question"] == QUESTION and fixed["target"] == ANSWER
    assert {key: value for key, value in fixed.items() if key not in ["question", "target"]} == {
        key: value for key, value in original.items() if key not in ["question", "target"]}
    assert original["question"] == "original q" and original["target"] == "original a"


def test_paired_evaluation_matches_images_not_random_upload_position():
    rows = [{"id": "second", "score": 2}, {"id": "first", "score": 1}]
    matched = align_runs(rows, {"second": 1, "first": 0}, [0, 1])
    assert [row["score"] for row in matched] == [1, 2]
    with pytest.raises(ValueError, match="one-to-one"):
        align_runs(rows, {"second": 0, "first": 0}, [0, 1])
