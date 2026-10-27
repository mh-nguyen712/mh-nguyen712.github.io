// Light/dark toggle (dark is the default; the choice is remembered).
document.querySelector(".theme-toggle")?.addEventListener("click", () => {
  const root = document.documentElement;
  const light = root.dataset.theme !== "light";
  if (light) root.dataset.theme = "light";
  else delete root.dataset.theme;
  try {
    localStorage.setItem("theme", light ? "light" : "dark");
  } catch (e) {}
});

// Abstract / BibTeX toggles on publications.
document.querySelectorAll("[data-toggle]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = document.getElementById(btn.dataset.toggle);
    if (!target) return;
    // Close the sibling panel (abstract vs bibtex) so only one is open at a time.
    target.parentElement.querySelectorAll(".pub-extra").forEach((el) => {
      if (el !== target) el.hidden = true;
    });
    target.hidden = !target.hidden;
    btn.parentElement.querySelectorAll("[data-toggle]").forEach((b) => b.classList.toggle("active", b === btn && !target.hidden));
  });
});
