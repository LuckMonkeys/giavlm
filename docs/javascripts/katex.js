// Renders the math that pymdownx.arithmatex (generic mode) wraps in `.arithmatex`
// as \(...\) or \[...\]. Runs on every page load, including instant navigation.
document$.subscribe(() => {
  if (typeof katex === "undefined") return;
  for (const el of document.querySelectorAll(".arithmatex")) {
    const tex = el.textContent.trim();
    const display = tex.startsWith("\\[");
    katex.render(tex.slice(2, -2), el, {
      displayMode: display,
      throwOnError: false,
      macros: {},
    });
  }
});
