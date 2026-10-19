import io
import json

import pytest

from utils import docs_editor, experiment_index
from test_experiment_index import RECORD, TRACKS


NAME = "2026-10-06-tiny-run.md"


@pytest.fixture
def docs(tmp_path):
    experiments = tmp_path / "experiments"
    (experiments / "examples").mkdir(parents=True)
    (experiments / NAME).write_text(RECORD)
    (experiments / "tracks.yaml").write_text(TRACKS)
    (experiments / "examples" / "private.md").write_text("private")
    (experiments / experiment_index.INDEX_NAME).write_text("stale")
    (tmp_path / "index.md").write_text("# Home\n\nSee [[tiny-run]].\n")
    return tmp_path


def call(docs, method, route, payload=None, address="127.0.0.1", header=True):
    data = json.dumps(payload or {}).encode()
    path, _, query = route.partition("?")
    environ = {"REQUEST_METHOD": method, "PATH_INFO": docs_editor.PREFIX + path,
               "QUERY_STRING": query, "REMOTE_ADDR": address,
               "CONTENT_LENGTH": str(len(data)), "wsgi.input": io.BytesIO(data)}
    if header:
        environ[docs_editor.EDIT_HEADER] = "1"
    status = []
    app = docs_editor.EditApp(lambda environ, start: [b"page"], docs)
    body = app(environ, lambda line, headers: status.append(int(line.split()[0])))
    return status[0], json.loads(b"".join(body))


def source(docs, name=f"experiments/{NAME}"):
    return call(docs, "GET", f"source?path={name}")[1]


def test_other_paths_reach_the_site_and_only_loopback_may_edit(docs):
    app = docs_editor.EditApp(lambda environ, start: [b"page"], docs)
    assert app({"PATH_INFO": "/experiments/"}, None) == [b"page"]
    assert call(docs, "GET", "ping")[0] == 200
    assert call(docs, "GET", "ping", address="::ffff:127.0.0.1")[0] == 200
    assert call(docs, "GET", "ping", address="192.168.1.20")[0] == 403


def test_reads_source_with_digest(docs):
    data = source(docs)
    assert data["text"] == RECORD and data["path"] == f"docs/experiments/{NAME}"
    assert data["sha"] == docs_editor.digest(RECORD.encode())


@pytest.mark.parametrize("name", ["../secret.md", "experiments/README.md",
                                  "experiments/examples/private.md", "experiments/x.json",
                                  "missing.md"])
def test_refuses_generated_private_and_outside_paths(docs, name):
    status, data = call(docs, "GET", f"source?path={name}")
    assert status in (403, 404) and not data["ok"]


def test_write_needs_header_and_current_digest(docs):
    sha = source(docs)["sha"]
    new = RECORD.replace("all stages finish", "every stage finishes")
    payload = {"path": f"experiments/{NAME}", "text": new, "sha": sha}
    assert call(docs, "POST", "source", payload, header=False)[0] == 403
    assert call(docs, "POST", "source", {**payload, "sha": "0" * 64})[0] == 409
    assert (docs / "experiments" / NAME).read_text() == RECORD


def test_invalid_record_is_rejected_without_writing(docs):
    payload = {"path": f"experiments/{NAME}", "text": RECORD.replace("## 要点", "## Summary"),
               "sha": source(docs)["sha"]}
    status, data = call(docs, "POST", "source", payload)
    assert status == 422 and "要点" in data["error"]
    assert (docs / "experiments" / NAME).read_text() == RECORD


def test_renaming_an_id_that_other_pages_link_is_rejected(docs):
    payload = {"path": f"experiments/{NAME}", "text": RECORD.replace("id: tiny-run", "id: other"),
               "sha": source(docs)["sha"]}
    assert call(docs, "POST", "source", payload)[0] == 422


def test_valid_write_saves_and_regenerates_the_index(docs):
    new = RECORD.replace("all stages finish", "every stage finishes")
    status, data = call(docs, "POST", "source",
                        {"path": f"experiments/{NAME}", "text": new, "sha": source(docs)["sha"]})
    assert status == 200 and data["changed"]
    assert (docs / "experiments" / NAME).read_text() == new
    index = (docs / "experiments" / experiment_index.INDEX_NAME).read_text()
    assert "## Smoke tests" in index
    assert data["sha"] == source(docs)["sha"]
    assert not list((docs / "experiments").glob(".*.edit"))


def test_plain_pages_only_check_links(docs):
    sha = source(docs, "index.md")["sha"]
    bad = {"path": "index.md", "text": "See [[nowhere]].\n", "sha": sha}
    assert call(docs, "POST", "source", bad)[0] == 422
    good = {"path": "index.md", "text": "# Home\n\nUpdated.\n", "sha": sha}
    assert call(docs, "POST", "source", good)[0] == 200
    assert (docs / "index.md").read_text() == "# Home\n\nUpdated.\n"
