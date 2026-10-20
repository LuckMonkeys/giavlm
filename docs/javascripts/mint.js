// Part of the optional Mintlify-like skin (stylesheets/mint.css): shows the page's
// navigation group, e.g. "属性推断", as a small label above the title.
document$.subscribe(() => {
  const title = document.querySelector(".md-content__inner > h1");
  if (!title || title.previousElementSibling?.classList.contains("mint-eyebrow")) return;
  const sections = document.querySelectorAll(
    ".md-nav--primary .md-nav__item--active.md-nav__item--section > .md-nav__link .md-ellipsis");
  const name = sections.length ? sections[sections.length - 1].textContent.trim() : "";
  if (!name || name === title.textContent.replace("¶", "").trim()) return;
  const label = document.createElement("p");
  label.className = "mint-eyebrow";
  label.textContent = name;
  title.before(label);
});
