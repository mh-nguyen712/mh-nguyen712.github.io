// Before/after image viewer (see _includes/compare.html).
// Drag the divider to compare; wheel / pinch / buttons to zoom; drag to pan when zoomed; keyboard accessible.
(() => {
  const MAX_ZOOM = 12;

  function init(root) {
    if (root.dataset.ready) return;
    root.dataset.ready = "1";

    const stage = root.querySelector(".compare-stage");
    const before = root.querySelector(".compare-before");
    const divider = root.querySelector(".compare-divider");
    const imgs = root.querySelectorAll(".compare-layer img");
    const readout = root.querySelector(".compare-zoom");

    let split = 0.5; // divider position, fraction of stage width
    let scale = 1;
    let tx = 0; // pan offset in px (top-left corner of the scaled image)
    let ty = 0;

    const size = () => stage.getBoundingClientRect();

    function clampPan() {
      const { width, height } = size();
      tx = Math.min(0, Math.max(width - width * scale, tx));
      ty = Math.min(0, Math.max(height - height * scale, ty));
    }

    function render(animate = false) {
      clampPan();
      root.classList.toggle("is-animating", animate);
      root.classList.toggle("is-zoomed", scale > 1.001);
      root.classList.toggle("is-pixelated", scale >= 3);
      const t = `translate(${tx}px, ${ty}px) scale(${scale})`;
      imgs.forEach((img) => (img.style.transform = t));
      before.style.clipPath = `inset(0 ${(1 - split) * 100}% 0 0)`;
      divider.style.left = `${split * 100}%`;
      divider.setAttribute("aria-valuenow", Math.round(split * 100));
      readout.textContent = `${scale.toFixed(1)}×`;
    }

    // Zoom to `next`, keeping the stage point (px, py) fixed on screen.
    function zoomAt(next, px, py, animate = false) {
      next = Math.min(MAX_ZOOM, Math.max(1, next));
      tx = px - ((px - tx) / scale) * next;
      ty = py - ((py - ty) / scale) * next;
      scale = next;
      render(animate);
    }

    const center = () => {
      const { width, height } = size();
      return [width / 2, height / 2];
    };

    function setSplit(clientX) {
      const r = size();
      split = Math.min(1, Math.max(0, (clientX - r.left) / r.width));
      render();
    }

    // --- Pointer input: divider drag, pan, pinch ---------------------------------------------------
    const pointers = new Map();
    let mode = null; // "split" | "pan" | "pinch"
    let last = null;
    let pinchStart = null;

    stage.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      try {
        stage.setPointerCapture(e.pointerId);
      } catch {}
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchStart = { dist: Math.hypot(a.x - b.x, a.y - b.y), scale };
        mode = "pinch";
        return;
      }
      const onDivider = divider.contains(e.target);
      mode = onDivider || scale <= 1.001 ? "split" : "pan";
      last = { x: e.clientX, y: e.clientY };
      if (mode === "split") setSplit(e.clientX);
      root.classList.toggle("is-panning", mode === "pan");
    });

    stage.addEventListener("pointermove", (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (mode === "pinch" && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const r = size();
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        zoomAt((pinchStart.scale * dist) / pinchStart.dist, (a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top);
      } else if (mode === "split") {
        setSplit(e.clientX);
      } else if (mode === "pan") {
        tx += e.clientX - last.x;
        ty += e.clientY - last.y;
        last = { x: e.clientX, y: e.clientY };
        render();
      }
    });

    const endPointer = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size === 0) {
        mode = null;
        root.classList.remove("is-panning");
      } else if (mode === "pinch") {
        // Back to one finger: continue as a pan.
        const p = [...pointers.values()][0];
        mode = "pan";
        last = { x: p.x, y: p.y };
      }
    };
    stage.addEventListener("pointerup", endPointer);
    stage.addEventListener("pointercancel", endPointer);

    // Wheel zooms around the cursor; at 1× scrolling down lets the page scroll as usual.
    stage.addEventListener(
      "wheel",
      (e) => {
        if (scale <= 1.001 && e.deltaY > 0) return;
        e.preventDefault();
        const r = size();
        zoomAt(scale * Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
      },
      { passive: false }
    );

    // Double click: zoom in at the point, or reset when already zoomed.
    stage.addEventListener("dblclick", (e) => {
      if (divider.contains(e.target)) return;
      const r = size();
      if (scale > 1.001) reset();
      else zoomAt(3, e.clientX - r.left, e.clientY - r.top, true);
    });

    function reset() {
      scale = 1;
      tx = ty = 0;
      render(true);
    }

    // --- Buttons and keyboard -------------------------------------------------------------------
    root.querySelector(".compare-toolbar").addEventListener("click", (e) => {
      const action = e.target.closest("[data-action]")?.dataset.action;
      if (action === "zoom-in") zoomAt(scale * 1.6, ...center(), true);
      if (action === "zoom-out") zoomAt(scale / 1.6, ...center(), true);
      if (action === "reset") reset();
      if (action === "focus") {
        const [fx, fy] = root.dataset.focus.split(",").map(Number);
        const { width, height } = size();
        scale = 4;
        tx = width / 2 - fx * width * scale;
        ty = height / 2 - fy * height * scale;
        render(true);
      }
    });

    divider.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 0.1 : 0.02;
      const moves = { ArrowLeft: -step, ArrowRight: step, Home: -1, End: 1 };
      if (!(e.key in moves)) return;
      e.preventDefault();
      e.stopPropagation();
      split = Math.min(1, Math.max(0, split + moves[e.key]));
      render();
    });

    stage.addEventListener("keydown", (e) => {
      const pan = 40;
      const keys = {
        "+": () => zoomAt(scale * 1.6, ...center(), true),
        "=": () => zoomAt(scale * 1.6, ...center(), true),
        "-": () => zoomAt(scale / 1.6, ...center(), true),
        0: reset,
        ArrowLeft: () => ((tx += pan), render()),
        ArrowRight: () => ((tx -= pan), render()),
        ArrowUp: () => ((ty += pan), render()),
        ArrowDown: () => ((ty -= pan), render()),
      };
      if (!keys[e.key] || (e.key.startsWith("Arrow") && scale <= 1.001)) return;
      e.preventDefault();
      keys[e.key]();
    });

    root.addEventListener("transitionend", () => root.classList.remove("is-animating"));
    window.addEventListener("resize", () => render());
    render();
  }

  const start = () => document.querySelectorAll("[data-compare]").forEach(init);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
