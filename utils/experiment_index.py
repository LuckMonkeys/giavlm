"""Validate experiment records in docs/experiments and regenerate their index."""
import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys

import yaml

from core.config import KNOWLEDGE_CONDITIONS


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "docs" / "experiments"
# Hand-written documents whose [[id]] links must resolve.
LINKED_DOCS = (ROOT / "docs" / "PROJECT_HANDOFF.md", ROOT / "docs" / "index.md")
INDEX_NAME = "README.md"
TRACKS_NAME = "tracks.yaml"
IMPORTANCE = ("key", "supporting", "archive")
EVIDENCE_LEVELS = ("smoke", "diagnostic", "development", "validation", "benchmark")
OUTCOMES = ("positive", "negative", "mixed", "inconclusive", "invalid")
REQUIRED = {
    "id": str, "title": str, "date": date, "track": str, "importance": str,
    "knowledge": list, "n": int, "evidence_level": str, "outcome": str,
    "uses_private_reference": bool, "contains_private_tokens": bool,
    "commits": list, "artifacts": list,
}
OPTIONAL = {
    "end_date": date, "data": str, "model": str, "tuning": str, "round": int,
    "fed": str, "token_lengths_known": bool, "uncommitted": bool,
    "private_reference_note": str,
    "superseded_by": str, "archive_note": str, "code": list, "related": list,
}
CHOICES = {"importance": IMPORTANCE, "evidence_level": EVIDENCE_LEVELS, "outcome": OUTCOMES}
ID_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{7,40}")
LINK_PATTERN = re.compile(r"\[\[([^\]]+)\]\]")
CODE_PATTERN = re.compile(r"`[^`\n]*`")
KEY_POINTS_HEADING = "## 要点"
BOLD_LEAD = re.compile(r"^\*\*(.+?)\*\*")


def experiment_links(text):
    """Experiment ids referenced as [[id]] outside inline code."""
    return set(LINK_PATTERN.findall(CODE_PATTERN.sub("", text)))


def split_record(text, name):
    if not text.startswith("---\n"):
        raise ValueError(f"{name}: missing YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{name}: unterminated YAML frontmatter")
    meta = yaml.safe_load(text[4:end])
    if not isinstance(meta, dict):
        raise ValueError(f"{name}: frontmatter must be a mapping")
    return meta, text[end + 5:]


def _artifact_path(entry, name):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("path"), str) \
            and set(entry) <= {"path", "note"} and isinstance(entry.get("note", ""), str):
        return entry["path"]
    raise ValueError(f"{name}: artifacts entries are paths or {{path, note}} mappings")


def validate_record(meta, body, name):
    errors = []
    for key, kind in REQUIRED.items():
        if key not in meta:
            errors.append(f"missing {key}")
        elif type(meta[key]) is not kind:
            errors.append(f"{key} must be {kind.__name__}")
    for key, value in meta.items():
        if key not in REQUIRED and key not in OPTIONAL:
            errors.append(f"unknown field {key}")
        elif key in OPTIONAL and type(value) is not OPTIONAL[key]:
            errors.append(f"{key} must be {OPTIONAL[key].__name__}")
    if errors:
        raise ValueError(f"{name}: " + "; ".join(errors))
    for key, choices in CHOICES.items():
        if meta[key] not in choices:
            errors.append(f"{key} must be one of {', '.join(choices)}")
    if not ID_PATTERN.fullmatch(meta["id"]):
        errors.append("id must be lowercase kebab-case")
    if name != f"{meta['date'].isoformat()}-{meta['id']}.md":
        errors.append("file name must be <date>-<id>.md")
    if meta.get("end_date") and meta["end_date"] < meta["date"]:
        errors.append("end_date precedes date")
    unknown = [value for value in meta["knowledge"] if value not in KNOWLEDGE_CONDITIONS]
    if unknown or not meta["knowledge"]:
        errors.append(f"knowledge must list conditions from core.config: {unknown}")
    if not key_points(body):
        errors.append(f"body needs a '{KEY_POINTS_HEADING}' section with bullets")
    if meta["n"] < 1:
        errors.append("n must be positive")
    if not all(isinstance(c, str) and COMMIT_PATTERN.fullmatch(c) for c in meta["commits"]):
        errors.append("commits must be hex SHAs")
    if not meta["commits"] and not meta.get("uncommitted"):
        errors.append("an experiment without commits must set uncommitted: true")
    if meta["uses_private_reference"] != bool(meta.get("private_reference_note", "").strip()):
        errors.append("uses_private_reference needs private_reference_note saying what was used")
    if meta["importance"] == "archive" and not (meta.get("superseded_by")
                                                or meta.get("archive_note")):
        errors.append("archived experiments need superseded_by or archive_note")
    for key in ("code", "related"):
        if not all(isinstance(value, str) for value in meta.get(key, [])):
            errors.append(f"{key} must be a list of strings")
    try:
        paths = [_artifact_path(entry, name) for entry in meta["artifacts"]]
    except ValueError as error:
        errors.append(str(error).split(": ", 1)[1])
        paths = []
    if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
        errors.append("artifact paths must be repository-relative")
    if any("capture/private" in path or "captures/private" in path for path in paths):
        errors.append("artifact paths must not point into private captures")
    if not body.strip():
        errors.append("body is empty")
    if errors:
        raise ValueError(f"{name}: " + "; ".join(errors))
    return paths


def key_points(body, lead_only=False):
    """Top-level bullets of the 要点 section; with lead_only, just their bold leads."""
    points, inside = [], False
    for line in body.splitlines():
        if line.startswith("## "):
            inside = line.strip() == KEY_POINTS_HEADING
        elif inside and line.startswith("- "):
            point = line[2:].strip()
            lead = BOLD_LEAD.match(point)
            points.append(lead.group(1) if lead_only and lead else point)
    return points


def summary(track):
    value = track["summary"]
    return [value] if isinstance(value, str) else list(value)


def load_tracks(directory):
    path = directory / TRACKS_NAME
    tracks = yaml.safe_load(path.read_text()) if path.exists() else None
    if not isinstance(tracks, dict):
        raise ValueError(f"{TRACKS_NAME}: must map track names to definitions")
    for name, track in tracks.items():
        if not isinstance(track, dict) \
                or set(track) != {"title", "summary", "start_here", "pages"} \
                or not isinstance(track["title"], str) \
                or not isinstance(track["summary"], (str, list)) \
                or not all(isinstance(track[key], list) and track[key]
                           for key in ("start_here", "pages")):
            raise ValueError(f"{TRACKS_NAME}: {name} needs title, summary, start_here and pages")
    return tracks


def load_catalog(directory=EXPERIMENTS):
    """Records, tracks and an id -> record map."""
    tracks = load_tracks(directory)
    records = []
    for path in sorted(directory.glob("*.md")):
        if path.name == INDEX_NAME:
            continue
        meta, body = split_record(path.read_text(), path.name)
        paths = validate_record(meta, body, path.name)
        records.append({"meta": meta, "body": body, "file": path.name, "artifact_paths": paths})
    names = {}
    for record in records:
        if record["meta"]["id"] in names:
            raise ValueError(f"duplicate experiment id: {record['meta']['id']}")
        names[record["meta"]["id"]] = record
    ids = set(names)
    for record in records:
        meta = record["meta"]
        if meta["track"] not in tracks:
            raise ValueError(f"{record['file']}: track {meta['track']} is not in {TRACKS_NAME}")
        canonical = set(meta.get("related", []))
        canonical |= {meta["superseded_by"]} if meta.get("superseded_by") else set()
        missing = sorted(canonical - ids) + sorted(experiment_links(record["body"]) - set(names))
        if missing:
            raise ValueError(f"{record['file']}: unknown experiment references {missing}")
    for name, track in tracks.items():
        expected = sorted(r["meta"]["id"] for r in records if r["meta"]["track"] == name)
        if sorted(track["pages"]) != expected:
            raise ValueError(f"{TRACKS_NAME}: {name} pages must list each experiment of the "
                             f"track exactly once, in reading order: {expected}")
        for value in track["start_here"]:
            record = names.get(value)
            if record is None or record["meta"]["track"] != name \
                    or record["meta"]["importance"] == "archive":
                raise ValueError(f"{TRACKS_NAME}: {name} start_here {value} must be a "
                                 "non-archived experiment of that track")
    return {"records": records, "tracks": tracks, "names": names}


def load_records(directory=EXPERIMENTS):
    return load_catalog(directory)["records"]


def check_links(path, catalog):
    """Reject `[[id]]` references in another document that name no experiment."""
    missing = sorted(experiment_links(path.read_text()) - set(catalog["names"]))
    if missing:
        raise ValueError(f"{path.name}: unknown experiment references {missing}")


FLAGS = ("使用私有数据", "uncommitted", "private-tokens")


def date_range(meta):
    start, end = meta["date"].isoformat(), meta.get("end_date")
    return start if end is None or end == meta["date"] else f"{start} – {end.isoformat()}"


def condition(meta):
    parts = [meta.get("tuning"), f"round {meta['round']}" if "round" in meta else None,
             " / ".join(meta["knowledge"]),
             "lengths known" if meta.get("token_lengths_known") else None]
    return " · ".join(part for part in parts if part)


def flags(meta):
    values = [("使用私有数据", meta["uses_private_reference"]),
              ("uncommitted", meta.get("uncommitted", False)),
              ("private-tokens", meta["contains_private_tokens"])]
    return [name for name, present in values if present]


def _cell(text):
    return " ".join(str(text).split()).replace("|", "\\|")


def track_records(catalog, name):
    """A track's records in the reading order declared by its `pages` in tracks.yaml."""
    return [catalog["names"][value] for value in catalog["tracks"][name]["pages"]]


def _filter_groups(catalog):
    """Quick-filter buttons per table column, limited to values that occur."""
    metas = [record["meta"] for record in catalog["records"]]
    columns = {
        "Importance": (IMPORTANCE, {m["importance"] for m in metas}),
        "Level": (EVIDENCE_LEVELS, {m["evidence_level"] for m in metas}),
        "Outcome": (OUTCOMES, {m["outcome"] for m in metas}),
        "Flags": (FLAGS, {flag for m in metas for flag in flags(m)}),
    }
    return {column: [value for value in values if value in present]
            for column, (values, present) in columns.items()}


def render_index(catalog):
    groups = json.dumps(_filter_groups(catalog), ensure_ascii=False)
    lines = [
        "# 实验索引",
        "",
        "<!-- Generated by `python -m utils.experiment_index` from the experiment files and",
        "tracks.yaml; edit those, not this page. -->",
        "",
        "每个研究问题对应一页。同一批样本上的消融、对照和事后审计，写在该页的小节里。",
        "",
        "- **key**：承载当前结论的实验，建议优先阅读。",
        "- **archive**：已被取代或有明显局限，只为溯源保留。",
        "- **使用私有数据**：用到了攻击者拿不到的真实图像或文本，结论只能用于诊断。",
        "- **uncommitted**：代码或结果还没有提交到 git。",
        "- **private-tokens**：页面引用了私有 token 片段，只能在本地查看。",
        "",
        f"<div class=\"table-filter\" data-groups='{groups}'></div>",
        "",
    ]
    names = catalog["names"]
    for name, track in catalog["tracks"].items():
        start = "、".join(f"[{names[value]['meta']['title']}]({names[value]['file']})"
                         for value in track["start_here"])
        lines += [f"## {track['title']}", ""]
        lines += [f"- {' '.join(item.split())}" for item in summary(track)]
        lines += ["", f"**先读：** {start}", "",
                  "| Importance | Dates | Experiment | Condition | n | Level | Outcome | Flags "
                  "| Findings |",
                  "|---|---|---|---|---:|---|---|---|---|"]
        for record in track_records(catalog, name):
            meta = record["meta"]
            importance = meta["importance"]
            importance = f"**{importance}**" if importance == "key" else importance
            lines.append("| " + " | ".join([
                importance, date_range(meta), f"[{_cell(meta['title'])}]({record['file']})",
                _cell(condition(meta)), str(meta["n"]), meta["evidence_level"], meta["outcome"],
                ", ".join(flags(meta)),
                "<br>".join(f"• {_cell(item)}" for item in key_points(record["body"], True)),
            ]) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="validate and fail if the index is stale instead of writing it")
    parser.add_argument("--check-artifacts", action="store_true",
                        help="also require every listed artifact path to exist locally")
    parser.add_argument("--directory", type=Path, default=EXPERIMENTS)
    args = parser.parse_args(argv)
    catalog = load_catalog(args.directory)
    if args.directory == EXPERIMENTS:
        for path in LINKED_DOCS:
            check_links(path, catalog)
    if args.check_artifacts:
        missing = [f"{r['file']}: {path}" for r in catalog["records"]
                   for path in r["artifact_paths"] if not (ROOT / path).exists()]
        if missing:
            raise SystemExit("missing artifacts:\n" + "\n".join(missing))
    index = args.directory / INDEX_NAME
    text = render_index(catalog)
    if args.check:
        if not index.exists() or index.read_text() != text:
            raise SystemExit(f"{index} is stale; run python -m utils.experiment_index")
    else:
        index.write_text(text)
    print(f"{len(catalog['records'])} experiment records valid", file=sys.stderr)


if __name__ == "__main__":
    main()
