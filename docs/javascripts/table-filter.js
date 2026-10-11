// Quick filters for the experiment index: <div class="table-filter" data-groups='{column: [values]}'>.
// Buttons in one group are alternatives (OR); groups and the text box combine with AND.
document$.subscribe(() => {
  const host = document.querySelector(".md-typeset .table-filter");
  if (!host || host.childElementCount) return;
  const groups = JSON.parse(host.dataset.groups || "{}");
  const active = Object.fromEntries(Object.keys(groups).map((column) => [column, new Set()]));
  const tables = [...document.querySelectorAll(".md-typeset table")].map((table) => {
    const headers = [...table.querySelectorAll("thead th")].map((th) => th.textContent.trim());
    return { table, headers, rows: [...table.querySelectorAll("tbody tr")] };
  });
  const total = tables.reduce((sum, t) => sum + t.rows.length, 0);

  for (const [column, values] of Object.entries(groups)) {
    const row = document.createElement("div");
    row.className = "table-filter__group";
    row.innerHTML = `<span class="table-filter__label">${column}</span>`;
    for (const value of values) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = value;
      button.addEventListener("click", () => {
        const set = active[column];
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

  function cellMatches(cell, column, values) {
    const text = cell.textContent.trim().toLowerCase();
    return [...values].some((value) =>
      column === "Flags" ? text.split(/[,、]\s*/).includes(value) : text === value);
  }

  function apply() {
    const terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    let shown = 0;
    for (const { table, headers, rows } of tables) {
      let visible = 0;
      for (const row of rows) {
        const cells = row.querySelectorAll("td");
        const ok = Object.entries(active).every(([column, values]) => {
          const index = headers.indexOf(column);
          return !values.size || (index >= 0 && cellMatches(cells[index], column, values));
        }) && terms.every((term) => row.textContent.toLowerCase().includes(term));
        row.style.display = ok ? "" : "none";
        visible += ok;
      }
      (table.closest(".md-typeset__scrollwrap") || table).style.display = visible ? "" : "none";
      shown += visible;
    }
    status.textContent = `显示 ${shown} / ${total} 个实验`;
  }
  apply();
});
