"""In-browser editing for the local docs preview (`mkdocs serve` on a loopback address).

`utils/mkdocs_hooks.py:on_serve` mounts `EditApp` in front of MkDocs' live-reload server,
so the editor shares the preview's origin and port; `mkdocs build` output has no editing.
A page's source is read and written through `/__edit__/source`; MkDocs notices the change,
rebuilds and reloads the page. Every write is validated first (experiment records with
the full `utils.experiment_index` rules, other pages for `[[id]]` links), refuses
generated or private paths, and fails with 409 when the file changed after it was
opened, so concurrent edits are never overwritten.
"""
import hashlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import parse_qs

from utils import experiment_index


PREFIX = "/__edit__/"
# Writes need this header: a cross-site form cannot set it without a CORS preflight,
# which this app never answers.
EDIT_HEADER = "HTTP_X_GIAVLM_EDIT"
MAX_BYTES = 2_000_000
EXPERIMENTS = PurePosixPath("experiments")
TRACKS = EXPERIMENTS / experiment_index.TRACKS_NAME
GENERATED = {EXPERIMENTS / experiment_index.INDEX_NAME:
             "实验索引由 python -m utils.experiment_index 生成；请编辑各实验页或 tracks.yaml"}
# Private local examples, snippet includes and slide sources are never served for editing.
BLOCKED = (PurePosixPath("includes"), EXPERIMENTS / "examples", PurePosixPath("ppt"))


class EditError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


def digest(data):
    return hashlib.sha256(data).hexdigest()


def is_loopback(address):
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback


class EditApp:
    """WSGI middleware: `/__edit__/*` is handled here, everything else goes to `app`."""

    def __init__(self, app, docs_dir):
        self.app = app
        self.docs = Path(docs_dir).resolve()

    def __call__(self, environ, start_response):
        route = environ.get("PATH_INFO", "")
        if not route.startswith(PREFIX):
            return self.app(environ, start_response)
        try:
            if not is_loopback(environ.get("REMOTE_ADDR", "")):
                raise EditError(403, "只允许本机访问编辑接口")
            body = self.route(route[len(PREFIX):], environ)
            status = 200
        except EditError as error:
            status, body = error.status, {"ok": False, "error": error.message}
        data = json.dumps(body, ensure_ascii=False).encode()
        start_response(f"{status} {'OK' if status == 200 else 'Error'}",
                       [("Content-Type", "application/json; charset=utf-8"),
                        ("Content-Length", str(len(data))), ("Cache-Control", "no-store")])
        return [data]

    def route(self, name, environ):
        method = environ.get("REQUEST_METHOD", "GET")
        if name == "ping" and method == "GET":
            return {"ok": True}
        if name == "source" and method == "GET":
            query = parse_qs(environ.get("QUERY_STRING", ""))
            return self.read(query.get("path", [""])[0])
        if name == "source" and method == "POST":
            if environ.get(EDIT_HEADER) != "1":
                raise EditError(403, "缺少编辑请求头")
            length = int(environ.get("CONTENT_LENGTH") or 0)
            if length > MAX_BYTES * 2:
                raise EditError(413, "内容过大")
            try:
                payload = json.loads(environ["wsgi.input"].read(length) or b"{}")
                return self.write(payload["path"], payload["text"], payload["sha"])
            except (ValueError, KeyError, TypeError) as error:
                raise EditError(400, f"请求格式错误：{error}") from error
        raise EditError(404, "未知的编辑接口")

    def blocked(self, path):
        rel = PurePosixPath(path.relative_to(self.docs).as_posix())
        return any(rel == blocked or blocked in rel.parents for blocked in BLOCKED)

    def resolve(self, name):
        rel = PurePosixPath(name)
        if not name or rel.is_absolute() or ".." in rel.parts:
            raise EditError(403, "路径必须在 docs/ 之内")
        if rel in GENERATED:
            raise EditError(403, GENERATED[rel])
        if self.blocked(self.docs / rel):
            raise EditError(403, "这个目录不能在网页中编辑")
        if rel.suffix != ".md" and rel != TRACKS:
            raise EditError(403, "只能编辑 Markdown 页面和 tracks.yaml")
        path = (self.docs / rel).resolve()
        if self.docs not in path.parents or not path.is_file():
            raise EditError(404, "文件不存在")
        return path, rel

    def read(self, name):
        path, rel = self.resolve(name)
        data = path.read_bytes()
        return {"ok": True, "path": f"docs/{rel}", "text": data.decode("utf-8"),
                "sha": digest(data)}

    def write(self, name, text, base_sha):
        path, rel = self.resolve(name)
        data = text.encode("utf-8")
        if len(data) > MAX_BYTES:
            raise EditError(413, "内容过大")
        current = path.read_bytes()
        if digest(current) != base_sha:
            raise EditError(409, "文件在打开编辑器之后已被修改。请先复制你的改动，重新打开后再合并。")
        if data == current:
            return {"ok": True, "sha": base_sha, "changed": False}
        record = rel == TRACKS or rel.parent == EXPERIMENTS
        catalog = self.validate(rel, text, record)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".edit")
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
        if record:
            index = self.docs / EXPERIMENTS / experiment_index.INDEX_NAME
            index.write_text(experiment_index.render_index(catalog))
        return {"ok": True, "sha": digest(data), "changed": True}

    def validate(self, rel, text, record):
        """The catalog as it would be after the write; raise 422 on any rule violation."""
        directory = self.docs / EXPERIMENTS
        try:
            if record:
                with tempfile.TemporaryDirectory() as scratch:
                    copy = Path(scratch)
                    for source in [*directory.glob("*.md"), directory / experiment_index.TRACKS_NAME]:
                        shutil.copy2(source, copy / source.name)
                    (copy / rel.name).write_text(text)
                    catalog = experiment_index.load_catalog(copy)
            else:
                catalog = experiment_index.load_catalog(directory)
        except ValueError as error:
            raise EditError(422, f"校验未通过：{error}") from error
        names = set(catalog["names"])
        pages = {path.relative_to(self.docs).as_posix(): path.read_text()
                 for path in self.docs.rglob("*.md") if not self.blocked(path)}
        pages[rel.as_posix()] = text
        broken = sorted(f"{page}: {', '.join(sorted(missing))}"
                        for page, body in pages.items()
                        if (missing := experiment_index.experiment_links(body) - names))
        if broken:
            raise EditError(422, "存在指向不存在实验的链接：" + "；".join(broken))
        return catalog
