import pytest

from utils import experiment_index


RECORD = """---
id: tiny-run
title: Tiny run
date: 2026-10-06
track: smoke
importance: key
knowledge: [private]
n: 1
evidence_level: smoke
outcome: inconclusive
uses_private_reference: false
contains_private_tokens: false
commits: [73ebac2]
artifacts: [outputs/tiny/, {path: outputs/tiny/result.json, note: committed result}]
related: []
---

## 要点

- **Pipeline runs**：all stages finish.
- A point without a bold lead.

## 研究问题

- **Pipeline**：Does every stage finish?

## 结果

### Pipeline

**回答**：Yes.

Body linking [[tiny-run]].
"""
TRACKS = """smoke:
  title: Smoke tests
  summary: The pipeline runs.
  start_here: [tiny-run]
  pages: [tiny-run]
"""


def write(directory, record=RECORD, tracks=TRACKS, name="2026-10-06-tiny-run.md"):
    (directory / name).write_text(record)
    (directory / "tracks.yaml").write_text(tracks)


def test_repository_experiment_records_are_valid_and_index_is_current():
    experiment_index.main(["--check"])


def test_index_lists_records_by_track_with_summary(tmp_path):
    write(tmp_path)
    experiment_index.main(["--directory", str(tmp_path)])
    index = (tmp_path / "README.md").read_text()
    assert "## Smoke tests" in index
    assert "The pipeline runs." in index
    assert "**先读：** [Tiny run](2026-10-06-tiny-run.md)" in index
    assert "| **key** | 2026-10-06 | [Tiny run](2026-10-06-tiny-run.md)" in index
    experiment_index.main(["--directory", str(tmp_path), "--check"])


def test_key_points_feed_the_findings_column(tmp_path):
    write(tmp_path)
    record = experiment_index.load_records(tmp_path)[0]
    assert experiment_index.key_points(record["body"], True) == [
        "Pipeline runs", "A point without a bold lead."]
    experiment_index.main(["--directory", str(tmp_path)])
    assert "• Pipeline runs<br>• A point without a bold lead." in (
        tmp_path / "README.md").read_text()


@pytest.mark.parametrize(("old", "new", "message"), [
    ("knowledge: [private]", "knowledge: [everything_known]", "knowledge"),
    ("outcome: inconclusive", "outcome: great", "outcome"),
    ("importance: key", "importance: urgent", "importance"),
    ("n: 1", "n: one", "n must be int"),
    ("commits: [73ebac2]", "commits: []", "uncommitted"),
    ("related: []", "related: [missing-run]", "missing-run"),
    ("## 要点", "## Summary", "要点"),
    ("artifacts: [outputs/tiny/,", "artifacts: [outputs/capture/private/x,", "private"),
    ("n: 1", "n: 1\nsecret: 1", "unknown field"),
    ("track: smoke", "track: other", "tracks.yaml"),
    ("uses_private_reference: false", "uses_private_reference: true", "private_reference_note"),
    ("uses_private_reference: false",
     "uses_private_reference: false\nprivate_reference_note: 从真实图像出发", "private_reference_note"),
    ("title: Tiny run", "title: 一个很长的标题，在文档侧栏里放不下一行", "title is wider"),
    ("- A point without a bold lead.", "- A point without a bold lead.\n" + "- More.\n" * 4,
     "more than 5 bullets"),
    ("- **Pipeline**：Does", "- Does", "bold topic"),
    ("### Pipeline", "### Other", "repeat the question topics"),
    ("**回答**：Yes.", "Yes.", "must open with"),
])
def test_invalid_records_are_rejected(tmp_path, old, new, message):
    write(tmp_path, RECORD.replace(old, new))
    with pytest.raises(ValueError, match=message):
        experiment_index.load_catalog(tmp_path)


def test_archived_record_needs_a_reason_and_cannot_be_a_start(tmp_path):
    write(tmp_path, RECORD.replace("importance: key", "importance: archive"))
    with pytest.raises(ValueError, match="archive"):
        experiment_index.load_catalog(tmp_path)
    archived = RECORD.replace("importance: key", "importance: archive\narchive_note: Old.")
    write(tmp_path, archived)
    with pytest.raises(ValueError, match="start_here"):
        experiment_index.load_catalog(tmp_path)


def test_track_pages_must_list_every_record_once(tmp_path):
    write(tmp_path, tracks=TRACKS.replace("  pages: [tiny-run]\n", "  pages: [other-run]\n"))
    with pytest.raises(ValueError, match="pages"):
        experiment_index.load_catalog(tmp_path)


def test_file_name_must_match_date_and_id(tmp_path):
    write(tmp_path, name="tiny-run.md")
    with pytest.raises(ValueError, match="file name"):
        experiment_index.load_catalog(tmp_path)


def test_stale_index_fails_check(tmp_path):
    write(tmp_path)
    (tmp_path / "README.md").write_text("old")
    with pytest.raises(SystemExit, match="stale"):
        experiment_index.main(["--directory", str(tmp_path), "--check"])
