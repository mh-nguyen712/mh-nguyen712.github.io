// Interactive point-cloud background: a diffusion model, live.
// Points start as Gaussian noise and are "denoised" into a dataset (Swiss roll, a neural network, an Airy pattern,
// a Gaussian mixture), then noised back and morphed into the next one. Positions follow the DDPM forward marginal
//   x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps     (cosine schedule)
// The cloud turns toward the mouse, and points near the cursor are pushed away and spring back.
// Colour comes from the --cloud CSS variable, so it follows the light/dark theme.
(() => {
  const canvas = document.getElementById("point-cloud");
  if (!canvas || !canvas.getContext) return;
  const ctx = canvas.getContext("2d");
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const LEVELS = 12; // depth buckets, so we only change fillStyle a few times per frame
  const REPEL_RADIUS = 130;
  const LINK_RADIUS = 110;
  const NOISE_STD = 0.55;

  // Seconds spent in each phase of one cycle.
  const PHASES = [
    { name: "denoising", duration: 5.5 },
    { name: "sample", duration: 4 },
    { name: "noising", duration: 3 },
    { name: "noise", duration: 0.8 },
  ];

  let width = 0;
  let height = 0;
  let dpr = 1;
  let wasMobile = null;
  let color = "94, 231, 255";
  let maxAlpha = 0.6;

  let n = 0;
  let eps; // fixed Gaussian noise, 3 floats per point
  let shapes = []; // [{ name, data: Float32Array(3n) }]
  let offsets; // screen-space repulsion offset + velocity, 4 floats per point
  let shapeIndex = 0;
  let phaseIndex = 0;
  let phaseTime = 0;

  const mouse = { nx: 0, ny: 0, x: -9999, y: -9999, active: false };
  const rot = { x: 0, y: 0 };
  let clock = 0;
  let last = 0;

  function readTheme() {
    const style = getComputedStyle(document.documentElement);
    color = style.getPropertyValue("--cloud").trim() || color;
    maxAlpha = parseFloat(style.getPropertyValue("--cloud-alpha")) || maxAlpha;
  }

  // ---------------------------------------------------------------------------
  // Datasets (all roughly inside the unit ball)
  // ---------------------------------------------------------------------------

  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  function swissRoll(count) {
    const out = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = 1.5 * Math.PI * (1 + 2 * Math.random());
      const r = theta / (4.5 * Math.PI);
      out[i * 3] = r * Math.cos(theta) + randn() * 0.012;
      out[i * 3 + 1] = r * Math.sin(theta) + randn() * 0.012;
      out[i * 3 + 2] = (Math.random() * 2 - 1) * 0.45;
    }
    return out;
  }

  function gaussianMixture(count) {
    const k = 7;
    const golden = Math.PI * (3 - Math.sqrt(5));
    const centers = [];
    for (let j = 0; j < k; j++) {
      const y = 1 - (j / (k - 1)) * 2;
      const r = Math.sqrt(1 - y * y);
      centers.push([Math.cos(golden * j) * r * 0.75, y * 0.75, Math.sin(golden * j) * r * 0.75]);
    }
    const out = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const c = centers[i % k];
      const s = 0.07 + 0.05 * ((i % k) / k);
      out[i * 3] = c[0] + randn() * s;
      out[i * 3 + 1] = c[1] + randn() * s;
      out[i * 3 + 2] = c[2] + randn() * s;
    }
    return out;
  }

  // Bessel function J1 (rational/asymptotic approximation from Numerical Recipes).
  function besselJ1(x) {
    const ax = Math.abs(x);
    if (ax < 8) {
      const y = x * x;
      const p = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.4826 + y * -30.16036606)))));
      const q = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y))));
      return p / q;
    }
    const z = 8 / ax;
    const y = z * z;
    const xx = ax - 2.356194491;
    const p = 1 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * -0.240337019e-6)));
    const q = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
    const ans = Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * p - z * Math.sin(xx) * q);
    return x < 0 ? -ans : ans;
  }

  // Airy pattern (in-focus PSF of a circular aperture) drawn as an intensity surface over the focal plane:
  //   I(v) = (2 J1(v) / v)^2,  height ∝ I^gamma (gamma < 1 so the faint diffraction rings stay visible).
  // Points sit on concentric rings, slightly denser near the centre to resolve the main lobe.
  function airyPattern(count) {
    const V_MAX = 13.3; // out to the 4th dark ring
    const GAMMA = 0.35;
    const HEIGHT = 1.2;
    const RINGS = 40;
    const radii = Array.from({ length: RINGS }, (_, k) => ((k + 1) / RINGS) ** 1.25);
    const radiusSum = radii.reduce((a, b) => a + b, 0);
    const out = new Float32Array(count * 3);
    let i = 0;
    const put = (x, z, r) => {
      if (i >= count) return;
      const v = r * V_MAX;
      const intensity = v < 1e-9 ? 1 : ((2 * besselJ1(v)) / v) ** 2;
      out[i * 3] = x;
      out[i * 3 + 1] = 0.45 - HEIGHT * intensity ** GAMMA; // screen y points down, so the peak points up
      out[i * 3 + 2] = z;
      i++;
    };
    put(0, 0, 0);
    for (const r of radii) {
      const m = Math.max(3, Math.floor((r / radiusSum) * (count - 1)));
      for (let j = 0; j < m; j++) {
        const a = (j / m) * 2 * Math.PI;
        put(r * Math.cos(a), r * Math.sin(a), r);
      }
    }
    // Rounding can leave a few slots: park them on the outer rim.
    for (; i < count; ) {
      const a = Math.random() * 2 * Math.PI;
      put(Math.cos(a), Math.sin(a), 1);
    }
    return out;
  }

  // A small MLP: layers of nodes (grids in the y-z plane) along x, with edges drawn as dotted lines.
  function neuralNetwork(count) {
    const layers = [3, 4, 5, 4, 2]; // grid side length per layer
    const nodes = layers.map((side, l) => {
      const x = -1 + (2 * l) / (layers.length - 1);
      const gap = 0.24;
      const list = [];
      for (let a = 0; a < side; a++) for (let b = 0; b < side; b++) list.push([x, (a - (side - 1) / 2) * gap, (b - (side - 1) / 2) * gap]);
      return list;
    });
    const pick = (list) => list[Math.floor(Math.random() * list.length)];
    const out = new Float32Array(count * 3);
    const nodePoints = Math.floor(count * 0.4);
    const allNodes = nodes.flat();
    for (let i = 0; i < nodePoints; i++) {
      const c = allNodes[i % allNodes.length];
      out[i * 3] = c[0] + randn() * 0.02;
      out[i * 3 + 1] = c[1] + randn() * 0.02;
      out[i * 3 + 2] = c[2] + randn() * 0.02;
    }
    // Edges: a random subset of connections between consecutive layers, ~8 points each.
    const edges = [];
    const edgeCount = Math.ceil((count - nodePoints) / 8);
    for (let e = 0; e < edgeCount; e++) {
      const l = Math.floor(Math.random() * (layers.length - 1));
      edges.push([pick(nodes[l]), pick(nodes[l + 1])]);
    }
    for (let i = nodePoints; i < count; i++) {
      const [a, b] = edges[(i - nodePoints) % edges.length];
      const t = Math.random();
      out[i * 3] = a[0] + (b[0] - a[0]) * t;
      out[i * 3 + 1] = a[1] + (b[1] - a[1]) * t;
      out[i * 3 + 2] = a[2] + (b[2] - a[2]) * t;
    }
    return out;
  }

  function build() {
    n = width < 760 ? 900 : 1600;
    eps = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) eps[i] = randn() * NOISE_STD;
    shapes = [
      { name: "swiss roll", data: swissRoll(n) },
      { name: "neural network", data: neuralNetwork(n) },
      { name: "airy pattern", data: airyPattern(n) },
      { name: "gaussian mixture", data: gaussianMixture(n) },
    ];
    offsets = new Float32Array(n * 4);
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Only rebuild when crossing the mobile breakpoint (mobile address bars resize the viewport on scroll).
    const mobile = width < 760;
    if (!shapes.length || mobile !== wasMobile) build();
    wasMobile = mobile;
  }

  function layout() {
    const mobile = width < 760;
    return {
      mobile,
      cx: mobile ? width * 0.5 : width * 0.72,
      cy: mobile ? height * 0.32 : height * 0.46,
      radius: mobile ? width * 0.4 : Math.min(height * 0.34, width * 0.24),
    };
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
      if (phaseIndex === 0) shapeIndex = (shapeIndex + 1) % shapes.length; // switch dataset while it is pure noise
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
    const data = shapes[shapeIndex].data;
    const { mobile, cx, cy, radius } = layout();

    // Ease rotation toward the mouse, on top of a slow drift and a little scroll-driven spin.
    const targetY = clock * 0.25 + mouse.nx * 0.9 + window.scrollY * 0.0012;
    const targetX = -mouse.ny * 0.6 + 0.25 + Math.sin(clock * 0.3) * 0.1;
    rot.y += (targetY - rot.y) * 0.05;
    rot.x += (targetX - rot.x) * 0.05;
    const sy = Math.sin(rot.y),
      cyR = Math.cos(rot.y);
    const sx = Math.sin(rot.x),
      cxR = Math.cos(rot.x);
    const fov = 4;

    ctx.clearRect(0, 0, width, height);
    const buckets = Array.from({ length: LEVELS }, () => []);
    const near = [];

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
      buckets[level].push(px, py, 1.7 * scale * (0.6 + depth));

      if (mouse.active && depth > 0.45) {
        const dx = px - mouse.x,
          dy = py - mouse.y;
        if (dx * dx + dy * dy < LINK_RADIUS * LINK_RADIUS * 2.2) near.push(px, py);
      }
    }

    const dim = mobile ? 0.5 : 1;
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
        const dx = near[j] - mouse.x,
          dy = near[j + 1] - mouse.y;
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
