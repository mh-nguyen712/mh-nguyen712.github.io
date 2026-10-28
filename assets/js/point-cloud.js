// Interactive point-cloud background: a diffusion model, live.
// Points start as Gaussian noise and are "denoised" into a dataset from assets/js/point-cloud-shapes.js (ML toy data,
// Zernike polynomials, optimization landscapes, classic surfaces...), then noised back and morphed into the next one.
// Positions follow the DDPM forward marginal
//   x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps     (cosine schedule)
// The cloud turns toward the mouse, and points near the cursor are pushed away and spring back. On wide screens two
// clouds fill the empty margins on both sides of the content. Colour comes from the --cloud CSS variable.
(() => {
  const canvas = document.getElementById("point-cloud");
  const SHAPES = window.PointCloudShapes;
  if (!canvas || !canvas.getContext || !SHAPES || !SHAPES.length) return;
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const LEVELS = 12; // depth buckets, so we only change fillStyle a few times per frame
  const REPEL_RADIUS = 130;
  const LINK_RADIUS = 110;
  const NOISE_STD = 0.55;
  const CONTENT_WIDTH = 760; // $max-width in assets/css/main.scss
  const DUAL_MIN_MARGIN = 420; // side margin (px) from which we draw two clouds
  const POINTS = { mobile: 1400, single: 3200, dual: 2600 }; // points per cloud

  // Seconds spent in each phase of one cycle.
  const PHASES = [
    { name: "denoising", duration: 5.5 },
    { name: "sample", duration: 4 },
    { name: "noising", duration: 3 },
    { name: "noise", duration: 0.8 },
  ];

  let width = 0;
  let height = 0;
  let mode = null; // "mobile" | "single" | "dual"
  let color = "94, 231, 255";
  let maxAlpha = 0.6;

  let n = 0; // points per cloud
  let shapeData = []; // one Float32Array(3n) per shape
  let clouds = []; // [{ eps, offsets, rot, shift, spin }]
  let shapeIndex = Math.floor(Math.random() * SHAPES.length); // start somewhere different on every page load
  let phaseIndex = 0;
  let phaseTime = 0;

  const mouse = { nx: 0, ny: 0, x: -9999, y: -9999, active: false };
  let clock = 0;
  let last = 0;

  function readTheme() {
    const style = getComputedStyle(document.documentElement);
    color = style.getPropertyValue("--cloud").trim() || color;
    maxAlpha = parseFloat(style.getPropertyValue("--cloud-alpha")) || maxAlpha;
  }

  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  function currentMode() {
    if (width < 760) return "mobile";
    return (width - Math.min(CONTENT_WIDTH, width)) / 2 >= DUAL_MIN_MARGIN ? "dual" : "single";
  }

  // Where each cloud sits on screen.
  function slots() {
    if (mode === "mobile") return [{ cx: width * 0.5, cy: height * 0.32, radius: width * 0.4 }];
    if (mode === "single") return [{ cx: width * 0.72, cy: height * 0.46, radius: Math.min(height * 0.34, width * 0.24) }];
    const margin = (width - CONTENT_WIDTH) / 2;
    const radius = Math.min(height * 0.34, margin * 0.5);
    return [
      { cx: margin * 0.5, cy: height * 0.5, radius },
      { cx: width - margin * 0.5, cy: height * 0.46, radius },
    ];
  }

  function build() {
    n = POINTS[mode];
    shapeData = SHAPES.map((shape) => shape.make(n));
    const count = mode === "dual" ? 2 : 1;
    clouds = Array.from({ length: count }, (_, k) => {
      const eps = new Float32Array(n * 3);
      for (let i = 0; i < n * 3; i++) eps[i] = randn() * NOISE_STD;
      return {
        eps,
        offsets: new Float32Array(n * 4), // screen-space repulsion offset + velocity
        rot: { x: 0.25, y: 0 },
        shift: k * Math.floor(SHAPES.length / 2), // the second cloud shows a different shape
        spin: k === 0 && count === 2 ? -1 : 1, // mirror the spin of the left cloud
      };
    });
  }

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Only rebuild when the layout mode changes (mobile address bars resize the viewport on scroll).
    const next = currentMode();
    if (next !== mode) {
      mode = next;
      build();
    }
  }

  // ---------------------------------------------------------------------------
  // Diffusion schedule
  // ---------------------------------------------------------------------------

  // Cosine schedule (Nichol & Dhariwal, 2021): abar(0) = 1, abar(1) = 0.
  function alphaBar(t) {
    const s = 0.008;
    const f = (u) => Math.cos(((u + s) / (1 + s)) * (Math.PI / 2)) ** 2;
    return Math.max(0, f(t) / f(0));
  }

  // Advance the cycle; returns diffusion time t in [0, 1] (1 = pure noise).
  function step(dt) {
    phaseTime += dt;
    while (phaseTime >= PHASES[phaseIndex].duration) {
      phaseTime -= PHASES[phaseIndex].duration;
      phaseIndex = (phaseIndex + 1) % PHASES.length;
      if (phaseIndex === 0) shapeIndex = (shapeIndex + 1) % SHAPES.length; // switch dataset while it is pure noise
    }
    const p = phaseTime / PHASES[phaseIndex].duration;
    return [1 - p, 0, p, 1][phaseIndex];
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  function frame(now) {
    const dt = last ? Math.min((now - last) / 1000, 0.05) : 0.016;
    last = now;
    clock += dt;
    const t = reduceMotion ? 0 : step(dt);
    const ab = alphaBar(t);
    const a = Math.sqrt(ab);
    const b = Math.sqrt(1 - ab);
    const jitter = reduceMotion ? 0 : 0.03 * b; // a little Langevin-style shake while noisy
    const fov = 4;
    const pointSize = mode === "mobile" ? 1.6 : 1.45;

    ctx.clearRect(0, 0, width, height);
    const buckets = Array.from({ length: LEVELS }, () => []);
    const near = [];

    slots().forEach(({ cx, cy, radius }, c) => {
      const cloud = clouds[c];
      if (!cloud) return;
      const shape = (shapeIndex + cloud.shift) % SHAPES.length;
      const data = shapeData[shape];
      const { eps, offsets, rot } = cloud;

      // Ease rotation toward the mouse, on top of a slow drift and a little scroll-driven spin. The per-shape tilt
      // changes while the cloud is pure (isotropic) noise, so the switch is invisible.
      const targetY = cloud.spin * clock * 0.25 + mouse.nx * 0.9 + window.scrollY * 0.0012;
      const targetX = -mouse.ny * 0.6 + 0.25 + (SHAPES[shape].tilt || 0) + Math.sin(clock * 0.3) * 0.1;
      rot.y += (targetY - rot.y) * 0.05;
      rot.x += (targetX - rot.x) * 0.05;
      const sy = Math.sin(rot.y);
      const cyR = Math.cos(rot.y);
      const sx = Math.sin(rot.x);
      const cxR = Math.cos(rot.x);

      for (let i = 0; i < n; i++) {
        const k = i * 3;
        const x = a * data[k] + b * eps[k] + (jitter ? (Math.random() - 0.5) * jitter : 0);
        const y = a * data[k + 1] + b * eps[k + 1] + (jitter ? (Math.random() - 0.5) * jitter : 0);
        const z = a * data[k + 2] + b * eps[k + 2];

        // Rotate around Y then X.
        const tx = x * cyR - z * sy;
        let tz = x * sy + z * cyR;
        const ty = y * cxR - tz * sx;
        tz = y * sx + tz * cxR;

        const scale = fov / (fov + tz);
        let px = cx + tx * radius * scale;
        let py = cy + ty * radius * scale;

        // Spring-back offset, pushed away from the cursor.
        const o = i * 4;
        if (mouse.active) {
          const dx = px + offsets[o] - mouse.x;
          const dy = py + offsets[o + 1] - mouse.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < REPEL_RADIUS * REPEL_RADIUS && d2 > 0.01) {
            const d = Math.sqrt(d2);
            const force = (1 - d / REPEL_RADIUS) * 2.2;
            offsets[o + 2] += (dx / d) * force;
            offsets[o + 3] += (dy / d) * force;
          }
        }
        offsets[o + 2] = (offsets[o + 2] - offsets[o] * 0.06) * 0.82;
        offsets[o + 3] = (offsets[o + 3] - offsets[o + 1] * 0.06) * 0.82;
        offsets[o] += offsets[o + 2];
        offsets[o + 1] += offsets[o + 3];
        px += offsets[o];
        py += offsets[o + 1];

        // Depth → brightness/size: front points are brighter and larger.
        const depth = Math.min(Math.max((1 - tz / 1.6) / 2, 0), 1);
        const level = Math.min(LEVELS - 1, Math.floor(depth * LEVELS));
        buckets[level].push(px, py, pointSize * scale * (0.6 + depth));

        if (mouse.active && depth > 0.45 && near.length < 120) {
          const dx = px - mouse.x;
          const dy = py - mouse.y;
          if (dx * dx + dy * dy < LINK_RADIUS * LINK_RADIUS * 2.2) near.push(px, py);
        }
      }
    });

    const dim = mode === "mobile" ? 0.5 : 1;
    for (let l = 0; l < LEVELS; l++) {
      const bucket = buckets[l];
      if (!bucket.length) continue;
      const alpha = dim * maxAlpha * (0.22 + 0.78 * Math.pow(l / (LEVELS - 1), 1.3));
      ctx.fillStyle = `rgba(${color}, ${alpha.toFixed(3)})`;
      for (let j = 0; j < bucket.length; j += 3) {
        const s = bucket[j + 2];
        ctx.fillRect(bucket[j] - s / 2, bucket[j + 1] - s / 2, s, s);
      }
    }

    // Faint "constellation" lines from the cursor to the nearest front points.
    if (near.length) {
      ctx.lineWidth = 0.6;
      for (let j = 0; j < near.length && j < 60; j += 2) {
        const dx = near[j] - mouse.x;
        const dy = near[j + 1] - mouse.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        const alpha = dim * maxAlpha * 0.5 * (1 - d / (LINK_RADIUS * 1.5));
        if (alpha <= 0) continue;
        ctx.strokeStyle = `rgba(${color}, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(mouse.x, mouse.y);
        ctx.lineTo(near[j], near[j + 1]);
        ctx.stroke();
      }
    }
  }

  let raf = null;
  function loop(now) {
    frame(now);
    raf = requestAnimationFrame(loop);
  }
  function start() {
    if (reduceMotion) return frame(performance.now());
    last = 0;
    if (raf === null) raf = requestAnimationFrame(loop);
  }
  function stop() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
  }

  window.addEventListener(
    "pointermove",
    (e) => {
      mouse.nx = (e.clientX / width) * 2 - 1;
      mouse.ny = (e.clientY / height) * 2 - 1;
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.active = true;
      if (reduceMotion) frame(performance.now());
    },
    { passive: true }
  );
  document.addEventListener("pointerleave", () => (mouse.active = false));
  window.addEventListener("blur", () => (mouse.active = false));
  window.addEventListener("resize", () => {
    resize();
    if (reduceMotion) frame(performance.now());
  });
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));

  // Re-read colours when the theme toggle flips data-theme.
  new MutationObserver(() => {
    readTheme();
    if (reduceMotion) frame(performance.now());
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  readTheme();
  resize();
  start();
})();
