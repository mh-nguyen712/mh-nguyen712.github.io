const root = document.documentElement;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Light/dark toggle (dark is the default; the choice is remembered). Where the browser supports view transitions, the
// new theme spreads out from the button.
const toggle = document.querySelector(".theme-toggle");
toggle?.addEventListener("click", () => {
  const flip = () => {
    const light = root.dataset.theme !== "light";
    if (light) root.dataset.theme = "light";
    else delete root.dataset.theme;
    try {
      localStorage.setItem("theme", light ? "light" : "dark");
    } catch (e) {}
  };
  if (reduceMotion || !document.startViewTransition) return flip();
  const box = toggle.getBoundingClientRect();
  const x = box.left + box.width / 2;
  const y = box.top + box.height / 2;
  root.style.setProperty("--reveal-x", `${x}px`);
  root.style.setProperty("--reveal-y", `${y}px`);
  root.style.setProperty("--reveal-r", `${Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y))}px`);
  root.classList.add("theme-switch");
  document.startViewTransition(flip).finished.finally(() => root.classList.remove("theme-switch"));
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

// Reading progress along the bottom edge of the header.
const progress = document.querySelector(".scroll-progress");
if (progress) {
  const update = () => {
    const max = root.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`;
  };
  addEventListener("scroll", update, { passive: true });
  addEventListener("resize", update);
  update();
}

// Cards and publications: a soft light follows the cursor (see --mx / --my in main.scss).
document.addEventListener(
  "pointermove",
  (e) => {
    const el = e.target.closest?.(".card, ol.bibliography > li");
    if (!el) return;
    const box = el.getBoundingClientRect();
    el.style.setProperty("--mx", `${e.clientX - box.left}px`);
    el.style.setProperty("--my", `${e.clientY - box.top}px`);
  },
  { passive: true }
);

if (!reduceMotion) {
  // Blocks below the first screen fade up as they scroll into view. What is already on screen is left alone, so
  // nothing flickers on load, and without JavaScript everything is simply visible.
  const observer = new IntersectionObserver(
    (entries) => {
      let shown = 0;
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const el = entry.target;
        observer.unobserve(el);
        el.style.transitionDelay = `${Math.min(shown++, 6) * 70}ms`; // stagger the ones that arrive together
        el.classList.add("is-visible");
        const done = (e) => {
          if (e.target !== el) return;
          el.removeEventListener("transitionend", done);
          el.classList.remove("reveal", "is-visible"); // hand the element back to its own hover transitions
          el.style.transitionDelay = "";
        };
        el.addEventListener("transitionend", done);
      }
    },
    { rootMargin: "0px 0px -6% 0px" }
  );
  const blocks = ".section-title, h2.bibliography, .news > li, .talks > li, ol.bibliography > li, .card, .entries > li, .compare";
  document.querySelectorAll(blocks).forEach((el) => {
    if (el.getBoundingClientRect().top < innerHeight) return;
    el.classList.add("reveal");
    observer.observe(el);
  });

  // The name on the home page resolves out of random glyphs, left to right (once per visit).
  const name = document.querySelector("[data-decode]");
  let played = false;
  try {
    played = sessionStorage.getItem("decoded") === "1";
    sessionStorage.setItem("decoded", "1");
  } catch (e) {}
  if (name && !played) {
    const text = name.textContent;
    const glyphs = "abcdefghijklmnopqrstuvwxyz0123456789#%&+*<>/";
    const start = performance.now();
    const tick = (now) => {
      const solved = Math.floor(((now - start) / 1100) * text.length);
      if (solved >= text.length) return (name.textContent = text);
      const noise = Array.from(text.slice(solved), (ch) => (ch === " " ? " " : glyphs[Math.floor(Math.random() * glyphs.length)]));
      name.textContent = text.slice(0, solved) + noise.join("");
      setTimeout(() => requestAnimationFrame(tick), 35);
    };
    requestAnimationFrame(tick);
  }

  // A faint light that follows the mouse across the page.
  if (window.matchMedia("(pointer: fine)").matches) {
    const glow = document.createElement("div");
    glow.className = "cursor-glow";
    glow.setAttribute("aria-hidden", "true");
    document.body.append(glow);
    addEventListener(
      "pointermove",
      (e) => {
        glow.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0)`;
        glow.classList.add("is-on");
      },
      { passive: true }
    );
    root.addEventListener("pointerleave", () => glow.classList.remove("is-on"));
  }
}
