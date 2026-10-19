// Quick filters for the experiment index: <div class="table-filter" data-groups='{column: [values]}'>.
// utils/mkdocs_hooks.py renders each experiment as <div class="exp-entry" data-importance=...
// data-level=... data-outcome=... data-flags="a,b">; a group named "Level" reads data-level.
// Buttons in one group are alternatives (OR); groups and the text box combine with AND.
document$.subscribe(() => {
  const host = document.querySelector(".md-typeset .table-filter");
  if (!host || host.childElementCount) return;
  const groups = JSON.parse(host.dataset.groups || "{}");
  const active = Object.fromEntries(Object.keys(groups).map((group) => [group, new Set()]));
  const entries = [...document.querySelectorAll(".md-typeset .exp-entry")];

  for (const [group, values] of Object.entries(groups)) {
    const row = document.createElement("div");
    row.className = "table-filter__group";
    row.innerHTML = `<span class="table-filter__label">${group}</span>`;
    for (const value of values) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = value;
      button.addEventListener("click", () => {
        const set = active[group];
        set.has(value) ? set.delete(value) : set.add(value);
        button.classList.toggle("is-active", set.has(value));
        apply();
      });
      row.appendChild(button);
    }
    host.appendChild(row);
  }
  const input = document.createElement("input");
  input.type = "search";
  input.placeholder = "搜索文字，例如 F-CL 或 DAGER";
  input.addEventListener("input", apply);
  const status = document.createElement("span");
  status.className = "table-filter__status";
  const bottom = document.createElement("div");
  bottom.className = "table-filter__group";
  bottom.append(input, status);
  host.appendChild(bottom);

  function matches(entry, group, values) {
    const field = entry.dataset[group.toLowerCase()] || "";
    const present = field.split(",").filter(Boolean);
    return [...values].some((value) => present.includes(value));
  }

  function apply() {
    const terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    for (const entry of entries) {
      const ok = Object.entries(active).every(([group, values]) =>
        !values.size || matches(entry, group, values))
        && terms.every((term) => entry.textContent.toLowerCase().includes(term));
      entry.hidden = !ok;
      shown += ok;
    }
    status.textContent = `显示 ${shown} / ${entries.length} 个实验`;
  }
  apply();
});
