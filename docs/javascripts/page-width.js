// Header button that toggles the page between Material's default width and full width,
// remembered per browser.
(() => {
  const KEY = "giavlm-full-width";
  const read = () => {
    try {
      return localStorage.getItem(KEY) === "1";
    } catch {
      return false;
    }
  };
  // overrides/main.html applies the saved width in <head>; this only handles toggling.
  const apply = (full) => {
    document.documentElement.style.setProperty("--page-width", full ? "none" : "61rem");
  };

  document$.subscribe(() => {
    const palette = document.querySelector('.md-header__option[data-md-component="palette"]');
    if (!palette || document.querySelector(".page-width")) return;
    let full = read();
    const box = document.createElement("div");
    box.className = "md-header__option page-width";
    const button = document.createElement("button");
    button.type = "button";
    const update = () => {
      apply(full);
      button.textContent = full ? "默认宽度" : "全宽";
      button.title = full ? "恢复默认页面宽度" : "页面铺满屏幕宽度";
      button.setAttribute("aria-pressed", String(full));
    };
    button.addEventListener("click", () => {
      full = !full;
      try { localStorage.setItem(KEY, full ? "1" : "0"); } catch {}
      update();
    });
    box.append(button);
    update();
    palette.before(box);
  });
})();
