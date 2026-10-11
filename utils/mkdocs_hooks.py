"""MkDocs hooks: experiment navigation, condition cards and in-page links."""
from collections import Counter
from pathlib import Path
import posixpath
import re
import sys

from mkdocs.exceptions import PluginError
from mkdocs.plugins import event_priority

# MkDocs loads hooks by file path, so the repository root is not importable by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import experiment_index  # noqa: E402


KEY_MARKER = "<!-- key-experiments -->"
TRACK_MARKER = "<!-- track-overview -->"
INLINE_CODE = re.compile(r"(`[^`\n]*`)")
MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
CATALOG = {}


def on_config(config):
    try:
        catalog = experiment_index.load_catalog()
        for path in experiment_index.LINKED_DOCS:
            experiment_index.check_links(path, catalog)
    except ValueError as error:
        raise PluginError(f"invalid experiment records: {error}") from error
    CATALOG.clear()
    CATALOG.update(catalog)
    for entry in config["nav"] or []:
        if isinstance(entry, dict) and "Experiments" in entry:
            entry["Experiments"] = _experiment_nav(catalog)
    return config


def _nav_title(meta):
    return f"★ {meta['title']}" if meta["importance"] == "key" else meta["title"]


def _experiment_nav(catalog):
    nav = ["experiments/README.md"]
    for name, track in catalog["tracks"].items():
        records = experiment_index.track_records(catalog, name)
        pages = [{_nav_title(r["meta"]): f"experiments/{r['file']}"} for r in records
                 if r["meta"]["importance"] != "archive"]
        archive = [{r["meta"]["title"]: f"experiments/{r['file']}"} for r in records
                   if r["meta"]["importance"] == "archive"]
        nav.append({track["title"]: pages + ([{"Archive": archive}] if archive else [])})
    return nav


def _link(record, page):
    target = f"experiments/{record['file']}"
    return posixpath.relpath(target, posixpath.dirname(page.file.src_uri) or ".")


def _resolve_links(markdown, page):
    def replace(match):
        record = CATALOG["names"].get(match.group(1))
        if record is None:
            raise PluginError(f"{page.file.src_uri}: unknown experiment [[{match.group(1)}]]")
        return f"[{record['meta']['title']}]({_link(record, page)})"
    def source(match):
        # Links to repository files outside docs/ work in an editor but not on the site.
        text, target = match.groups()
        if "://" in target or target.startswith(("#", "/", "mailto:")):
            return match.group(0)
        path = posixpath.normpath(posixpath.join(posixpath.dirname(page.file.src_uri),
                                                 target.split("#", 1)[0]))
        if not path.startswith("../"):
            return match.group(0)
        path = path.removeprefix("../")
        return f"`{path}`" if text.strip("`") in (path, posixpath.basename(path)) \
            else f"{text} (`{path}`)"
    parts = INLINE_CODE.split(markdown)
    return "".join(part if index % 2 else MARKDOWN_LINK.sub(
        source, experiment_index.LINK_PATTERN.sub(replace, part))
        for index, part in enumerate(parts))


def _badges(meta):
    badges = []
    if meta["uses_private_reference"]:
        note = " ".join(meta["private_reference_note"].split()).rstrip("。")
        badges.append(("private-data",
                       f"使用私有数据：{note}。攻击者拿不到这些数据，结论只用于诊断。"))
    if meta.get("uncommitted"):
        badges.append(("uncommitted", "未提交 git"))
    return " ".join(f'<span class="badge badge--{kind}">{text}</span>' for kind, text in badges)


def _card(record, page):
    """Title, badges and the condition row; everything else stays in the page body."""
    meta = record["meta"]
    setting = " · ".join(str(meta[key]) for key in ("data", "model", "fed") if meta.get(key))
    lines = [f"# {meta['title']}", ""]
    badges = _badges(meta)
    lines += [badges, ""] if badges else []
    lines += ["| Dates | Condition | Setting | n | Level | Outcome | Source |",
              "|---|---|---|---:|---|---|---|",
              f"| {experiment_index.date_range(meta)} | {experiment_index.condition(meta)} | "
              f"{setting} | {meta['n']} | {meta['evidence_level']} | **{meta['outcome']}** | "
              f"`{record['file']}` |",
              ""]
    if meta["importance"] == "archive":
        note = " ".join(meta.get("archive_note", "").split())
        see = f" 已被 [[{meta['superseded_by']}]] 取代。" if meta.get("superseded_by") else ""
        lines += ['!!! note "为什么归档"', f"    {note}{see}".rstrip(), ""]
    return "\n".join(lines) + "\n"


def _provenance(record):
    meta = record["meta"]
    lines = ["", '??? abstract "溯源：commit、代码、产物"']
    commits = "、".join(f"`{commit}`" for commit in meta["commits"]) or "无（未提交）"
    lines += [f"    - **Commit：** {commits}"]
    if meta.get("code"):
        lines += ["    - **代码：** " + "、".join(f"`{path}`" for path in meta["code"])]
    lines += ["    - **产物：**"]
    for entry in meta["artifacts"]:
        path, note = (entry, "") if isinstance(entry, str) else (entry["path"], entry.get("note"))
        lines += [f"        - `{path}`" + (f"：{note}" if note else "")]
    if meta.get("related"):
        lines += ["    - **相关实验：** " + "、".join(f"[[{value}]]" for value in meta["related"])]
    return "\n".join(lines) + "\n"


def _key_experiments(page):
    lines = []
    for name in CATALOG["tracks"]:
        for record in experiment_index.track_records(CATALOG, name):
            meta = record["meta"]
            if meta["importance"] == "key":
                lines.append(f"- **[{meta['title']}]({_link(record, page)})** "
                             f"*({meta['evidence_level']}, n={meta['n']})*")
                lines += [f"    - {item}"
                          for item in experiment_index.key_points(record["body"], True)]
    return "\n".join(lines)


def _track_overview(page):
    lines = []
    for name, track in CATALOG["tracks"].items():
        records = [r for r in CATALOG["records"] if r["meta"]["track"] == name]
        counts = Counter(r["meta"]["importance"] for r in records)
        tally = ", ".join(f"{counts[level]} {level}"
                          for level in experiment_index.IMPORTANCE if counts[level])
        start = "、".join(f"[[{value}]]" for value in track["start_here"])
        lines += [f"### {track['title']}", ""]
        lines += [f"- {' '.join(item.split())}" for item in experiment_index.summary(track)]
        lines += ["", f"*先读：* {start} · *共 {len(records)} 页：{tally}*", ""]
    return "\n".join(lines)


@event_priority(100)
def on_page_markdown(markdown, page, config, files):
    record = CATALOG["names"].get(page.meta.get("id"))
    if record is not None and page.file.src_uri == f"experiments/{record['file']}":
        markdown = _card(record, page) + markdown + _provenance(record)
    if KEY_MARKER in markdown:
        markdown = markdown.replace(KEY_MARKER, _key_experiments(page))
    if TRACK_MARKER in markdown:
        markdown = markdown.replace(TRACK_MARKER, _track_overview(page))
    return _resolve_links(markdown, page)
