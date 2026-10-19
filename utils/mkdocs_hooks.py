"""MkDocs hooks: experiment navigation, condition cards, home page cards and in-page links."""
from pathlib import Path
import html
import logging
import posixpath
import re
import sys

from mkdocs.exceptions import PluginError
from mkdocs.plugins import event_priority

# MkDocs loads hooks by file path, so the repository root is not importable by default.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import docs_editor, experiment_index  # noqa: E402

LOG = logging.getLogger("mkdocs.hooks.giavlm")


STATS_MARKER = "<!-- home-stats -->"
TRACK_MARKER = "<!-- track-overview -->"
DEFAULT_TRACK_ICON = "material-flask-outline"
INLINE_CODE = re.compile(r"(`[^`\n]*`)")
MARKDOWN_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
CATALOG = {}
PLANS_DIR = "plans"


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
    _expand_plans(config["nav"] or [], Path(config["docs_dir"]) / PLANS_DIR)
    return config


def _expand_plans(nav, folder):
    """Replace the `Plans: plans/` placeholder with every plan page, newest first."""
    for entry in nav:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if value == f"{PLANS_DIR}/":
                entry[key] = [{_page_title(path): f"{PLANS_DIR}/{path.name}"}
                              for path in sorted(folder.glob("*.md"), reverse=True)]
            elif isinstance(value, list):
                _expand_plans(value, folder)


def _page_title(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def _experiment_nav(catalog):
    nav = ["experiments/README.md"]
    for name, track in catalog["tracks"].items():
        records = experiment_index.track_records(catalog, name)
        # The track's start pages come first; that order is the only emphasis in the sidebar.
        start = [r for r in records if r["meta"]["id"] in track["start_here"]]
        records = start + [r for r in records if r not in start]
        pages = [{r["meta"]["title"]: f"experiments/{r['file']}"} for r in records
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
    """Title, badges and a condition card; everything else stays in the page body."""
    meta = record["meta"]
    setting = " · ".join(str(meta[key]) for key in ("data", "model", "fed") if meta.get(key))
    outcome = (f'<span class="exp-outcome exp-outcome--{meta["outcome"]}">'
               f'{meta["outcome"]}</span>')
    # Label/value pairs, two per row; Source takes a whole row.
    items = [("Outcome", outcome, ""),
             ("Level", meta["evidence_level"], ""),
             ("n", meta["n"], ""),
             ("Dates", experiment_index.date_range(meta), ""),
             ("Condition", experiment_index.condition(meta), ""),
             ("Setting", setting, ""),
             ("Source", f"`docs/experiments/{record['file']}`", " exp-meta__value--wide")]
    lines = [f"# {meta['title']}", ""]
    badges = _badges(meta)
    lines += [badges, ""] if badges else []
    # md_in_html renders each value as inline Markdown, so abbreviations still get tooltips.
    lines += ['<div class="exp-meta" markdown>']
    for label, value, extra in items:
        lines += [f'<div class="exp-meta__label">{label}</div>',
                  f'<div class="exp-meta__value{extra}" markdown="span">{value}</div>']
    lines += ["</div>", ""]
    if meta["importance"] == "archive":
        note = " ".join(meta.get("archive_note", "").split())
        see = f" 已被 [[{meta['superseded_by']}]] 取代。" if meta.get("superseded_by") else ""
        lines += ['!!! note "为什么归档"', f"    {note}{see}".rstrip(), ""]
    return "\n".join(lines) + "\n"


def _provenance(record):
    meta = record["meta"]
    lines = ["", '??? abstract "溯源：源文件、commit、代码、产物"']
    lines += [f"    - **源文件：** `docs/experiments/{record['file']}`"]
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


def _home_stats():
    records = CATALOG["records"]
    latest = max(r["meta"].get("end_date") or r["meta"]["date"] for r in records)
    stats = [(len(records), "实验记录"), (len(CATALOG["tracks"]), "研究方向"),
             (sum(r["meta"]["importance"] == "key" for r in records), "承载当前结论"),
             (latest.isoformat(), "最近更新")]
    tiles = "".join(f'<div class="home-stat"><span class="home-stat__value">{value}</span>'
                    f'<span class="home-stat__label">{label}</span></div>' for value, label in stats)
    return f'<div class="home-stats">{tiles}</div>'


def _track_overview(page):
    """One Material grid card per track: headline, start pages and page count."""
    lines = ['<div class="grid cards track-cards" markdown>', ""]
    for name, track in CATALOG["tracks"].items():
        count = sum(r["meta"]["track"] == name for r in CATALOG["records"])
        headline = " ".join(track.get("headline", experiment_index.summary(track)[0]).split())
        start = "、".join(f"[[{value}]]" for value in track["start_here"])
        lines += [f"-   :{track.get('icon', DEFAULT_TRACK_ICON)}:{{ .lg .middle }} "
                  f"**{track['title']}**", "", "    ---", "", f"    {headline}", "",
                  f"    :material-arrow-right: 先读 {start} · 共 {count} 页", ""]
    return "\n".join(lines + ["</div>"])


@event_priority(100)
def on_page_markdown(markdown, page, config, files):
    if page.parent is None:
        # A top-level page has no siblings, so the left sidebar would list only itself;
        # the home page also drops its short contents list so the cards get the full width.
        hidden = ["navigation", "toc"] if page.file.src_uri == "index.md" else ["navigation"]
        page.meta.setdefault("hide", hidden)
    record = CATALOG["names"].get(page.meta.get("id"))
    if record is not None and page.file.src_uri == f"experiments/{record['file']}":
        markdown = _card(record, page) + markdown + _provenance(record)
    if STATS_MARKER in markdown:
        markdown = markdown.replace(STATS_MARKER, _home_stats())
    if TRACK_MARKER in markdown:
        markdown = markdown.replace(TRACK_MARKER, _track_overview(page))
    if page.file.src_uri == INDEX_PAGE:
        markdown = _index_cards(markdown)
    return _resolve_links(markdown, page)


INDEX_PAGE = f"experiments/{experiment_index.INDEX_NAME}"
INDEX_TABLE = re.compile(r"^\| Importance \|.*?(?=^$)", re.M | re.S)


def _chip(kind, text):
    return f'<span class="chip chip--{kind}">{text}</span>'


def _entry(record):
    """One experiment of the index as a card; data-* attributes feed table-filter.js."""
    meta = record["meta"]
    marks = experiment_index.flags(meta)
    chips = " ".join([_chip(meta["importance"], meta["importance"]),
                      _chip("level", meta["evidence_level"]),
                      f'<span class="exp-outcome exp-outcome--{meta["outcome"]}">'
                      f'{meta["outcome"]}</span>'] + [_chip("flag", mark) for mark in marks])
    data = (f'data-importance="{meta["importance"]}" data-level="{meta["evidence_level"]}" '
            f'data-outcome="{meta["outcome"]}" data-flags="{",".join(marks)}"')
    lines = [f'<div class="exp-entry exp-entry--{meta["importance"]}" {data} markdown>', "",
             f'[{meta["title"]}]({record["file"]}){{ .exp-entry__title }} {chips}', "",
             f'{experiment_index.date_range(meta)} · {experiment_index.condition(meta)} · '
             f'n={meta["n"]}', "{ .exp-entry__meta }", ""]
    lines += [f"- {point}" for point in experiment_index.key_points(record["body"], True)]
    return "\n".join(lines + ["", "</div>", ""])


def _index_cards(markdown):
    """Swap each track's generated table for experiment cards (the source stays a table)."""
    tables = iter(CATALOG["tracks"])
    def replace(match):
        name = next(tables)
        return "\n".join(_entry(record) for record in experiment_index.track_records(CATALOG, name))
    return INDEX_TABLE.sub(replace, markdown)


def on_serve(server, config, builder):
    """Mount the in-browser editor (utils/docs_editor.py) on a loopback preview only."""
    host = str(server.server_address[0])
    if docs_editor.is_loopback(host):
        server.set_app(docs_editor.EditApp(server.serve_request, config["docs_dir"]))
        LOG.info("In-browser editing enabled at %s", docs_editor.PREFIX)
    else:
        LOG.warning("In-browser editing disabled: the preview listens on %s, not loopback", host)
    return server


def on_page_content(html_text, page, config, files):
    """Tell docs/javascripts/editor.js which source file the page comes from."""
    source = page.file.src_uri
    if source == INDEX_PAGE:
        source = f"experiments/{experiment_index.TRACKS_NAME}"
    return html_text + (f'<span class="giavlm-source" hidden '
                        f'data-source="{html.escape(source, quote=True)}"></span>')
