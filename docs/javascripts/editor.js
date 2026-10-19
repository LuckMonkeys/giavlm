// In-browser editing for the local preview; the server side is utils/docs_editor.py.
// The edit button appears only when /__edit__/ping answers, i.e. under `mkdocs serve`
// on a loopback address. Saving writes the page's Markdown source; MkDocs rebuilds and
// reloads the page, and the editor reopens at the same position.
(() => {
  const API = "/__edit__/";
  const REOPEN = "giavlm-editor-reopen";
  const PENCIL = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M20.71 7.04c.39-.39.39-1.04 0-1.41l-2.34-2.34c-.37-.39-1.02-.39-1.41 0l-1.84 1.83 3.75 3.75M3 17.25V21h3.75L17.81 9.93l-3.75-3.75z"/></svg>';
  let available = null;
  let ui = null;

  const ping = () => (available ??= fetch(API + "ping", { cache: "no-store" })
    .then((response) => response.ok).catch(() => false));
  const dirty = () => Boolean(ui && !ui.root.hidden && ui.text.value !== ui.original);

  function build() {
    const root = document.createElement("div");
    root.className = "giavlm-editor";
    root.hidden = true;
    root.innerHTML = `
      <div class="giavlm-editor__bar">
        <span class="giavlm-editor__path"></span>
        <span class="giavlm-editor__status" role="status"></span>
        <button type="button" class="giavlm-editor__save">保存</button>
        <button type="button" class="giavlm-editor__close">关闭</button>
      </div>
      <textarea class="giavlm-editor__text" spellcheck="false"></textarea>
      <div class="giavlm-editor__hint">Ctrl+S 保存 · Esc 关闭 · Tab 缩进四格 · 保存前会校验实验记录和链接</div>`;
    document.body.append(root);
    ui = {
      root, text: root.querySelector("textarea"), path: root.querySelector(".giavlm-editor__path"),
      status: root.querySelector(".giavlm-editor__status"), source: null, sha: null, original: "",
    };
    root.querySelector(".giavlm-editor__save").addEventListener("click", save);
    root.querySelector(".giavlm-editor__close").addEventListener("click", close);
    ui.text.addEventListener("input", () => setStatus(dirty() ? "未保存" : "", "pending"));
    ui.text.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        save();
      } else if (event.key === "Escape") {
        event.preventDefault();
        close();
      } else if (event.key === "Tab" && !event.shiftKey) {
        event.preventDefault();
        ui.text.setRangeText("    ", ui.text.selectionStart, ui.text.selectionEnd, "end");
        ui.text.dispatchEvent(new Event("input"));
      }
    });
    window.addEventListener("beforeunload", (event) => {
      if (dirty()) event.preventDefault();
    });
    return ui;
  }

  function setStatus(message, kind = "") {
    ui.status.textContent = message;
    ui.status.dataset.kind = kind;
  }

  async function open(source, restore) {
    if (!ui) build();
    ui.root.hidden = false;
    document.body.classList.add("giavlm-editing");
    ui.path.textContent = `docs/${source}`;
    ui.text.value = "";
    ui.text.disabled = true;
    setStatus("正在读取…");
    try {
      const response = await fetch(`${API}source?path=${encodeURIComponent(source)}`,
        { cache: "no-store" });
      const data = await response.json();
      if (!data.ok) return setStatus(data.error, "error");
      Object.assign(ui, { source, sha: data.sha, original: data.text });
      ui.text.value = data.text;
      ui.text.disabled = false;
      setStatus("");
      ui.text.focus();
      if (restore) {
        ui.text.setSelectionRange(restore.cursor, restore.cursor);
        ui.text.scrollTop = restore.scroll;
      }
    } catch (error) {
      setStatus(`读取失败：${error}`, "error");
    }
  }

  function close() {
    if (dirty() && !confirm("放弃未保存的改动？")) return;
    ui.root.hidden = true;
    document.body.classList.remove("giavlm-editing");
  }

  async function save() {
    if (!ui.source || ui.busy) return;
    if (ui.text.value === ui.original) return setStatus("没有改动");
    ui.busy = true;
    setStatus("正在校验并保存…", "pending");
    try {
      const response = await fetch(`${API}source`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-GIAVLM-Edit": "1" },
        body: JSON.stringify({ path: ui.source, text: ui.text.value, sha: ui.sha }),
      });
      const data = await response.json();
      if (!data.ok) return setStatus(data.error, "error");
      Object.assign(ui, { sha: data.sha, original: ui.text.value });
      setStatus("已保存，正在重新构建页面…", "ok");
      try {
        sessionStorage.setItem(REOPEN, JSON.stringify({
          source: ui.source, at: Date.now(),
          cursor: ui.text.selectionStart, scroll: ui.text.scrollTop,
        }));
      } catch {}
    } catch (error) {
      setStatus(`保存失败：${error}`, "error");
    } finally {
      ui.busy = false;
    }
  }

  document$.subscribe(async () => {
    const marker = document.querySelector(".md-content .giavlm-source");
    if (ui && !ui.root.hidden && !dirty() && ui.source !== marker?.dataset.source) close();
    if (!marker || !(await ping())) return;
    const source = marker.dataset.source;
    const article = document.querySelector(".md-content__inner");
    if (!article || article.querySelector(".giavlm-edit-button")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "md-content__button md-icon giavlm-edit-button";
    button.title = source.endsWith(".yaml") ? `编辑 docs/${source}` : "编辑本页 Markdown";
    button.innerHTML = PENCIL;
    button.addEventListener("click", () => open(source));
    article.prepend(button);
    let saved = null;
    try {
      saved = JSON.parse(sessionStorage.getItem(REOPEN) || "null");
      sessionStorage.removeItem(REOPEN);
    } catch {}
    if (saved && saved.source === source && Date.now() - saved.at < 60000) open(source, saved);
  });
})();
