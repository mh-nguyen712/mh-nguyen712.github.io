// Interactive point-cloud background: a diffusion model, live (WebGL).
// Points start as Gaussian noise and are "denoised" into a dataset from assets/js/point-cloud-shapes.js (ML toy data,
// Zernike polynomials, optimization landscapes, classic surfaces...), then noised back and morphed into the next one.
// Positions follow the DDPM forward marginal
//   x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps_t     (cosine schedule)
// where eps_t = cos(theta) * eps_1 + sin(theta) * eps_2 slowly rotates between two independent Gaussian samples: the
// noise swirls, yet eps_t stays exactly N(0, I), so the marginal is the right one at every instant.
//
// Everything is computed per point in the vertex shader, so tens of thousands of points are cheap:
//   - the cursor re-noises the points around it (they dissolve, then denoise back along its trail) and pushes them away;
//   - a click or tap sends a ripple through the cloud, and scrolling fast shakes a little noise back in;
//   - the clouds turn toward the mouse, and noisy / out-of-focus points are drawn as soft bokeh discs that sharpen as
//     the sample forms.
// On wide screens two clouds fill the margins on both sides of the content; points behind the text column are dimmed.
// Colours come from CSS variables (assets/css/main.scss): --cloud-1/2/3 ("r, g, b" from low to high values of each
// shape's colour coordinate), --cloud-alpha, and --cloud-blend (add | normal).
(() => {
  const canvas = document.getElementById("point-cloud");
  const SHAPES = window.PointCloudShapes;
  if (!canvas || !canvas.getContext || !SHAPES || !SHAPES.length) return;
  const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl) return; // no WebGL: the CSS background is all there is
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const NOISE_STD = 0.55;
  const CONTENT_WIDTH = 760; // $max-width in assets/css/main.scss
  const DUAL_MIN_MARGIN = 420; // side margin (px) from which we draw two clouds
  const BEHIND_TEXT = 0.3; // brightness of the points behind the content column, so the text stays readable
  const POINTS = { mobile: 4500, single: 12000, dual: 10000 }; // points per cloud
  const DUST = { mobile: 0, single: 420, dual: 600 }; // faint drifting specks that fill the rest of the screen
  const TRAIL = 10; // cursor positions remembered by the shader (the first one is the cursor itself)
  const TRAIL_LIFE = 1.3; // seconds for the cursor's wake to heal
  const RIPPLES = 4;
  const RIPPLE_LIFE = 1.5;
  const RIPPLE_SPEED = 620; // px / s
  const STATE_KEY = "point-cloud"; // sessionStorage: lets the animation carry on across page navigations

  // Seconds spent in each phase of one cycle.
  const PHASES = [
    { name: "denoising", duration: 5.5 },
    { name: "sample", duration: 4.5 },
    { name: "noising", duration: 3 },
    { name: "noise", duration: 0.8 },
  ];

  // ---------------------------------------------------------------------------
  // Shaders
  // ---------------------------------------------------------------------------

  const VERTEX = `
    precision highp float;
    #define TRAIL ${TRAIL}
    #define RIPPLES ${RIPPLES}

    attribute vec4 aPos;   // x_0 (xyz) and the colour coordinate in [0, 1] (w)
    attribute vec3 aEps1;  // two independent N(0, I) samples
    attribute vec3 aEps2;
    attribute vec4 aRand;  // uniform randoms in [0, 1)

    uniform float uTime;
    uniform float uT;        // diffusion time in [0, 1] (1 = pure noise)
    uniform float uKick;     // extra noise from scrolling
    uniform float uNoise;    // noise standard deviation
    uniform mat3 uRot;
    uniform vec2 uCenter;    // px
    uniform float uRadius;   // px
    uniform vec2 uViewport;  // px
    uniform float uDpr;
    uniform float uSize;     // px
    uniform float uAlpha;
    uniform float uDof;      // px of blur per unit of depth away from the focal plane
    uniform float uReach;    // px, radius of the cursor's influence
    uniform vec3 uTrail[TRAIL];    // x, y, strength
    uniform vec4 uRipple[RIPPLES]; // x, y, radius, amplitude
    uniform vec4 uMask;      // content column: left, right, feather (px) and how much the points behind it are dimmed
    uniform vec3 uColA;
    uniform vec3 uColB;
    uniform vec3 uColC;

    varying vec4 vColor;
    varying float vBlur;

    // Cosine schedule (Nichol & Dhariwal, 2021): abar(0) = 1, abar(1) = 0.
    float alphaBar(float t) {
      float f = cos((t + 0.008) / 1.008 * 1.5707963);
      return clamp(f * f / 0.999845, 0.0, 1.0);
    }

    vec3 sampleAt(float t, vec3 eps) {
      float ab = alphaBar(t);
      return sqrt(ab) * aPos.xyz + sqrt(1.0 - ab) * eps;
    }

    // Rotate, then perspective-project: returns screen px (xy) and depth (z, negative = towards the viewer).
    vec3 project(vec3 x) {
      vec3 v = uRot * x;
      return vec3(uCenter + v.xy * (uRadius * 4.0 / (4.0 + v.z)), v.z);
    }

    // Cursor falloff: 1 at the cursor, 0 from uReach on.
    float kernel(vec2 d) {
      float q = clamp(1.0 - dot(d, d) / (uReach * uReach), 0.0, 1.0);
      return q * q;
    }

    void main() {
      float theta = uTime * (0.2 + 0.5 * aRand.x) + aRand.y * 6.2831853;
      vec3 eps = (cos(theta) * aEps1 + sin(theta) * aEps2) * uNoise;

      // Points are a little out of step with each other, so the sample condenses rather than snaps into place.
      float t = clamp((uT - 0.22 * aRand.z) / 0.78, 0.0, 1.0);
      t = min(1.0, t + uKick * (0.3 + 0.7 * aRand.w));
      vec3 p = project(sampleAt(t, eps));

      // The cursor (and its fading trail) re-noises the points it passes over. The kernel has compact support: even
      // a little noise scrambles a fine grid, so the effect must really stop at uReach.
      float w = 0.0;
      for (int k = 0; k < TRAIL; k++) {
        w = max(w, uTrail[k].z * kernel(p.xy - uTrail[k].xy));
      }
      if (w > 0.0) {
        t = max(t, 0.6 * w);
        p = project(sampleAt(t, eps));
        vec2 d = p.xy - uTrail[0].xy;
        p.xy += normalize(d + 0.001) * (uTrail[0].z * kernel(d) * 24.0);
      }

      // Click ripples: an expanding ring that shoves the points outward and lights them up.
      float flash = 0.0;
      for (int k = 0; k < RIPPLES; k++) {
        vec2 d = p.xy - uRipple[k].xy;
        float dl = length(d) + 0.001;
        float q = (dl - uRipple[k].z) / 54.0;
        float ring = uRipple[k].w * exp(-q * q);
        p.xy += (d / dl) * (ring * 34.0);
        flash += ring;
      }

      float noise = sqrt(1.0 - alphaBar(t));
      float scale = 4.0 / (4.0 + p.z);
      float depth = clamp(0.5 - p.z / 3.2, 0.0, 1.0); // 1 = front

      // Depth of field: points off the focal plane, and noisy ones, spread into dimmer discs.
      float base = uSize * scale * (0.55 + 0.9 * depth) * (0.7 + 0.6 * aRand.w);
      float blur = (abs(p.z + 0.25) + 0.45 * noise) * uDof;
      float size = base + blur;
      gl_PointSize = size * uDpr;
      vBlur = clamp(blur / size * 1.25, 0.0, 1.0);

      float twinkle = 0.82 + 0.18 * sin(uTime * (0.7 + 2.3 * aRand.x) + aRand.y * 40.0);
      float bright = (0.3 + 0.7 * pow(depth, 1.3)) * twinkle * (1.0 + 1.5 * flash + 0.5 * w);
      float spread = pow(base / size, 1.2);
      float outside = max(uMask.x - p.x, p.x - uMask.y);
      float mask = mix(uMask.w, 1.0, smoothstep(-30.0, uMask.z, outside));

      vec3 color = aPos.w < 0.5 ? mix(uColA, uColB, aPos.w * 2.0) : mix(uColB, uColC, aPos.w * 2.0 - 1.0);
      color = mix(color, uColB, 0.55 * noise); // colour emerges with the structure
      vColor = vec4(color, uAlpha * bright * spread * mask * (1.0 - 0.25 * noise));

      gl_Position = vec4(p.x / uViewport.x * 2.0 - 1.0, 1.0 - p.y / uViewport.y * 2.0, 0.0, 1.0);
    }`;

  const FRAGMENT = `
    precision mediump float;
    varying vec4 vColor;
    varying float vBlur;

    void main() {
      vec2 q = gl_PointCoord * 2.0 - 1.0;
      float r2 = dot(q, q);
      if (r2 > 1.0) discard;
      float sharp = exp(-2.6 * r2);                                          // in focus: a soft dot
      float disc = smoothstep(1.0, 0.55, r2) * (0.65 + 0.35 * r2) * 0.7;     // out of focus: a disc with a brighter rim
      float a = mix(sharp, disc, vBlur) * vColor.a;
      gl_FragColor = vec4(vColor.rgb * a, a); // premultiplied
    }`;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  let width = 0;
  let height = 0;
  let dpr = 1;
  let mode = null; // "mobile" | "single" | "dual"
  let needsResize = true;

  let program = null;
  let loc = {}; // attribute and uniform locations
  let n = 0; // points per cloud
  let clouds = []; // [{ fixed, pos, loaded, rot, shift, spin }]
  let dust = null; // { fixed, pos, count }
  let shapeCache = new Map(); // shape index -> Float32Array(4n)
  let shapeIndex = Math.floor(Math.random() * SHAPES.length); // start somewhere different on every visit
  let phaseIndex = 0;
  let phaseTime = 0;

  const theme = { colors: [new Float32Array(3), new Float32Array(3), new Float32Array(3)], alpha: 0.7, add: true };
  const pointer = { x: 0, y: 0, nx: 0, ny: 0, fx: 0, fy: 0, strength: 0, active: false, moved: 0, seen: false };
  const trail = []; // [{ x, y, born }]
  const ripples = []; // [{ x, y, born }]
  const trailData = new Float32Array(TRAIL * 3);
  const rippleData = new Float32Array(RIPPLES * 4);
  const rotation = new Float32Array(9);
  let scrollY = window.scrollY;
  let kick = 0;
  let clock = 0;
  let last = 0;
  let restored = null;
  let ready = false;

  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  function readTheme() {
    const style = getComputedStyle(document.documentElement);
    ["96, 110, 255", "94, 231, 255", "255, 140, 214"].forEach((fallback, k) => {
      const rgb = (style.getPropertyValue(`--cloud-${k + 1}`).trim() || fallback).split(",").map((v) => parseFloat(v) / 255);
      if (rgb.length === 3 && rgb.every(isFinite)) theme.colors[k].set(rgb);
    });
    theme.alpha = parseFloat(style.getPropertyValue("--cloud-alpha")) || 0.7;
    theme.add = style.getPropertyValue("--cloud-blend").trim() !== "normal";
  }

  // Carry the animation over to the next page: same shape, same phase, same orientation.
  function saveState() {
    try {
      const rots = clouds.map((c) => [c.rot.x, c.rot.y]);
      sessionStorage.setItem(STATE_KEY, JSON.stringify({ shapeIndex, phaseIndex, phaseTime, clock, rots, at: Date.now() }));
    } catch (e) {}
  }

  function loadState() {
    try {
      const state = JSON.parse(sessionStorage.getItem(STATE_KEY));
      if (!state || Date.now() - state.at > 4000) return;
      shapeIndex = state.shapeIndex % SHAPES.length;
      phaseIndex = state.phaseIndex % PHASES.length;
      phaseTime = state.phaseTime;
      clock = state.clock;
      restored = state.rots;
    } catch (e) {}
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  function currentMode() {
    if (width < 760) return "mobile";
    return (width - CONTENT_WIDTH) / 2 >= DUAL_MIN_MARGIN ? "dual" : "single";
  }

  // Where each cloud sits on screen.
  function slots() {
    if (mode === "mobile") return [{ cx: width * 0.5, cy: height * 0.34, radius: Math.min(width * 0.42, height * 0.3) }];
    const margin = (width - CONTENT_WIDTH) / 2;
    if (mode === "single") {
      // One cloud on the right, pushed into the margin as far as it fits.
      const radius = Math.min(height * 0.34, width * 0.24);
      return [{ cx: width - Math.max(margin * 0.6, radius * 0.82), cy: height * 0.47, radius }];
    }
    const radius = Math.min(height * 0.35, margin * 0.52);
    return [
      { cx: margin * 0.5, cy: height * 0.52, radius },
      { cx: width - margin * 0.5, cy: height * 0.46, radius },
    ];
  }

  // ---------------------------------------------------------------------------
  // Geometry
  // ---------------------------------------------------------------------------

  // A shape's points as (x, y, z, c), where c in [0, 1] picks the colour: the shape's `color(x, y, z)` rescaled to
  // [0, 1], or its height by default.
  function shapeData(index) {
    if (shapeCache.has(index)) return shapeCache.get(index);
    const shape = SHAPES[index];
    const xyz = shape.make(n);
    const out = new Float32Array(n * 4);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = xyz[i * 3];
      const y = xyz[i * 3 + 1];
      const z = xyz[i * 3 + 2];
      const c = shape.color ? shape.color(x, y, z) : -y;
      out.set([x, y, z, c], i * 4);
      if (c < lo) lo = c;
      if (c > hi) hi = c;
    }
    const span = hi - lo || 1;
    for (let i = 0; i < n; i++) out[i * 4 + 3] = (out[i * 4 + 3] - lo) / span;
    shapeCache.set(index, out);
    return out;
  }

  const shapeOf = (cloud) => (shapeIndex + cloud.shift) % SHAPES.length;

  function buffer(data, usage) {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, data, usage);
    return b;
  }

  // Per-point constants: eps_1 (3), eps_2 (3), randoms (4).
  function fixedData(count) {
    const data = new Float32Array(count * 10);
    for (let i = 0; i < count; i++) {
      for (let d = 0; d < 6; d++) data[i * 10 + d] = randn();
      for (let d = 6; d < 10; d++) data[i * 10 + d] = Math.random();
    }
    return data;
  }

  function build() {
    clouds.forEach((c) => [c.fixed, c.pos].forEach((b) => gl.deleteBuffer(b)));
    if (dust) [dust.fixed, dust.pos].forEach((b) => gl.deleteBuffer(b));
    n = POINTS[mode];
    shapeCache = new Map();
    const count = mode === "dual" ? 2 : 1;
    clouds = Array.from({ length: count }, (_, k) => {
      const cloud = {
        fixed: buffer(fixedData(n), gl.STATIC_DRAW),
        pos: null,
        loaded: -1,
        rot: { x: restored && restored[k] ? restored[k][0] : 0.25, y: restored && restored[k] ? restored[k][1] : 0 },
        shift: k * Math.floor(SHAPES.length / 2), // the second cloud shows a different shape
        spin: k === 0 && count === 2 ? -1 : 1, // mirror the spin of the left cloud
      };
      cloud.loaded = shapeOf(cloud);
      cloud.pos = buffer(shapeData(cloud.loaded), gl.DYNAMIC_DRAW);
      return cloud;
    });
    restored = null;

    dust = null;
    const specks = DUST[mode];
    if (specks) {
      const pos = new Float32Array(specks * 4);
      for (let i = 0; i < specks * 4; i++) pos[i] = i % 4 === 3 ? Math.random() : Math.random() * 2 - 1;
      dust = { fixed: buffer(fixedData(specks), gl.STATIC_DRAW), pos: buffer(pos, gl.STATIC_DRAW), count: specks };
    }
  }

  function setup() {
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);

    loc = {};
    for (const name of ["aPos", "aEps1", "aEps2", "aRand"]) {
      loc[name] = gl.getAttribLocation(program, name);
      gl.enableVertexAttribArray(loc[name]);
    }
    const uniforms = "uTime uT uKick uNoise uRot uCenter uRadius uViewport uDpr uSize uAlpha uDof uReach uTrail uRipple uMask uColA uColB uColC";
    for (const name of uniforms.split(" ")) loc[name] = gl.getUniformLocation(program, name);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
  }

  function resize() {
    needsResize = false;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = document.documentElement.clientWidth; // without the scrollbar
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    // Only rebuild when the layout mode changes (mobile address bars resize the viewport on scroll).
    const next = currentMode();
    if (next !== mode) {
      mode = next;
      build();
    }
  }

  // ---------------------------------------------------------------------------
  // Animation
  // ---------------------------------------------------------------------------

  // Advance the cycle; returns diffusion time t in [0, 1] (1 = pure noise).
  function step(dt) {
    phaseTime += dt;
    while (phaseTime >= PHASES[phaseIndex].duration) {
      phaseTime -= PHASES[phaseIndex].duration;
      phaseIndex = (phaseIndex + 1) % PHASES.length;
      if (phaseIndex === 0) shapeIndex = (shapeIndex + 1) % SHAPES.length; // switch dataset while it is pure noise
      // Sample the next dataset ahead of time, while nothing is moving.
      if (phaseIndex === 1) setTimeout(() => clouds.forEach((c) => shapeData((shapeOf(c) + 1) % SHAPES.length)), 300);
    }
    const p = phaseTime / PHASES[phaseIndex].duration;
    return [1 - p, 0, p, 1][phaseIndex];
  }

  // Follow the cursor with a little inertia, and remember where it has been.
  function updatePointer(dt) {
    const ease = 1 - Math.exp(-dt * 14);
    pointer.fx += (pointer.x - pointer.fx) * ease;
    pointer.fy += (pointer.y - pointer.fy) * ease;
    const awake = pointer.active && clock - pointer.moved < 2.5; // a parked cursor lets the cloud heal
    pointer.strength += ((awake ? 1 : 0) - pointer.strength) * (1 - Math.exp(-dt * (awake ? 10 : 2.5)));

    const head = trail[trail.length - 1];
    if (pointer.strength > 0.05 && (!head || Math.hypot(pointer.fx - head.x, pointer.fy - head.y) > 16)) {
      trail.push({ x: pointer.fx, y: pointer.fy, born: clock });
      if (trail.length > TRAIL - 1) trail.shift();
    }
    while (trail.length && clock - trail[0].born > TRAIL_LIFE) trail.shift();
    while (ripples.length && clock - ripples[0].born > RIPPLE_LIFE) ripples.shift();

    trailData.fill(0);
    trailData.set([pointer.fx, pointer.fy, pointer.strength]);
    trail.forEach((s, k) => {
      const life = 1 - (clock - s.born) / TRAIL_LIFE;
      trailData.set([s.x, s.y, 0.9 * life * life], (k + 1) * 3);
    });
    rippleData.fill(0);
    ripples.forEach((r, k) => {
      const age = clock - r.born;
      const life = 1 - age / RIPPLE_LIFE;
      rippleData.set([r.x, r.y, age * RIPPLE_SPEED, life * life], k * 4);
    });
  }

  // Column-major matrix for "rotate around Y, then around X".
  function setRotation(rx, ry) {
    const sx = Math.sin(rx);
    const cx = Math.cos(rx);
    const sy = Math.sin(ry);
    const cy = Math.cos(ry);
    rotation.set([cy, -sx * sy, cx * sy, 0, cx, sx, -sy, -sx * cy, cx * cy]);
    gl.uniformMatrix3fv(loc.uRot, false, rotation);
  }

  function draw(fixed, pos, count) {
    gl.bindBuffer(gl.ARRAY_BUFFER, fixed);
    gl.vertexAttribPointer(loc.aEps1, 3, gl.FLOAT, false, 40, 0);
    gl.vertexAttribPointer(loc.aEps2, 3, gl.FLOAT, false, 40, 12);
    gl.vertexAttribPointer(loc.aRand, 4, gl.FLOAT, false, 40, 24);
    gl.bindBuffer(gl.ARRAY_BUFFER, pos);
    gl.vertexAttribPointer(loc.aPos, 4, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, count);
  }

  function frame(now) {
    const dt = last ? Math.min((now - last) / 1000, 0.05) : 0.016;
    last = now;
    clock += dt;
    if (needsResize) resize();
    const t = reduceMotion ? 0 : step(dt);
    if (!reduceMotion) updatePointer(dt);

    // Scrolling fast shakes some noise back into the sample.
    const y = window.scrollY;
    const speed = Math.abs(y - scrollY) / dt;
    scrollY = y;
    kick += (Math.min(0.14, speed / 16000) - kick) * (1 - Math.exp(-dt * 5));

    const mobile = mode === "mobile";
    const left = (width - CONTENT_WIDTH) / 2;

    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniform1f(loc.uTime, clock);
    gl.uniform2f(loc.uViewport, width, height);
    gl.uniform1f(loc.uDpr, dpr);
    gl.uniform1f(loc.uReach, mobile ? 110 : 140);
    gl.uniform3fv(loc.uTrail, trailData);
    gl.uniform4fv(loc.uRipple, rippleData);
    if (mobile) gl.uniform4f(loc.uMask, -1e5, 1e5, 1, BEHIND_TEXT);
    else gl.uniform4f(loc.uMask, left, left + CONTENT_WIDTH, 160, BEHIND_TEXT);
    gl.uniform3fv(loc.uColA, theme.colors[0]);
    gl.uniform3fv(loc.uColB, theme.colors[1]);
    gl.uniform3fv(loc.uColC, theme.colors[2]);
    // Dark theme: points add up their light (glow). Light theme: ordinary alpha blending (ink).
    gl.blendFunc(gl.ONE, theme.add ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA);

    // Dust: a slowly turning field of specks around the whole screen, with a little parallax on mouse and scroll.
    if (dust) {
      setRotation(pointer.ny * -0.05 + y * 0.00035, clock * 0.012 + pointer.nx * 0.08);
      gl.uniform1f(loc.uT, 0);
      gl.uniform1f(loc.uKick, 0);
      gl.uniform1f(loc.uNoise, 0.05);
      gl.uniform2f(loc.uCenter, width / 2, height / 2);
      gl.uniform1f(loc.uRadius, Math.hypot(width, height) * 0.62);
      gl.uniform1f(loc.uSize, 2.6);
      gl.uniform1f(loc.uAlpha, theme.alpha * 0.55);
      gl.uniform1f(loc.uDof, 4.5);
      draw(dust.fixed, dust.pos, dust.count);
    }

    slots().forEach(({ cx, cy, radius }, c) => {
      const cloud = clouds[c];
      if (!cloud) return;
      const shape = shapeOf(cloud);
      if (cloud.loaded !== shape) {
        gl.bindBuffer(gl.ARRAY_BUFFER, cloud.pos);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, shapeData(shape));
        cloud.loaded = shape;
      }

      // Ease rotation toward the mouse, on top of a slow drift and a little scroll-driven spin. The per-shape tilt
      // changes while the cloud is pure (isotropic) noise, so the switch is invisible.
      const targetY = cloud.spin * clock * 0.25 + pointer.nx * 0.9 + y * 0.0012;
      const targetX = -pointer.ny * 0.6 + 0.25 + (SHAPES[shape].tilt || 0) + Math.sin(clock * 0.3) * 0.1;
      const ease = reduceMotion ? 1 : 1 - Math.exp(-dt * 3);
      cloud.rot.y += (targetY - cloud.rot.y) * ease;
      cloud.rot.x += (targetX - cloud.rot.x) * ease;
      setRotation(cloud.rot.x, cloud.rot.y);

      gl.uniform1f(loc.uT, t);
      gl.uniform1f(loc.uKick, kick);
      gl.uniform1f(loc.uNoise, NOISE_STD);
      gl.uniform2f(loc.uCenter, cx, cy + Math.sin(clock * 0.45 + c * 2) * 7);
      gl.uniform1f(loc.uRadius, radius);
      gl.uniform1f(loc.uSize, mobile ? 3.4 : 3.1);
      gl.uniform1f(loc.uAlpha, theme.alpha);
      gl.uniform1f(loc.uDof, mobile ? 1.6 : 2.4);
      draw(cloud.fixed, cloud.pos, n);
    });

    if (!ready) canvas.classList.add("is-ready");
    ready = true;
  }

  let raf = null;
  function loop(now) {
    frame(now);
    raf = requestAnimationFrame(loop);
  }
  function start() {
    if (gl.isContextLost()) return;
    if (reduceMotion) return frame(performance.now());
    last = 0;
    if (raf === null) raf = requestAnimationFrame(loop);
  }
  function stop() {
    if (raf !== null) cancelAnimationFrame(raf);
    raf = null;
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------

  function track(e) {
    pointer.x = e.clientX;
    pointer.y = e.clientY;
    pointer.nx = (e.clientX / width) * 2 - 1;
    pointer.ny = (e.clientY / height) * 2 - 1;
    pointer.active = true;
    pointer.moved = clock;
    if (!pointer.seen) {
      // First sighting: don't sweep in from the corner.
      pointer.seen = true;
      pointer.fx = pointer.x;
      pointer.fy = pointer.y;
    }
  }

  if (!reduceMotion) {
    window.addEventListener("pointermove", track, { passive: true });
    window.addEventListener(
      "pointerdown",
      (e) => {
        if (e.button > 0) return;
        track(e);
        ripples.push({ x: e.clientX, y: e.clientY, born: clock });
        if (ripples.length > RIPPLES) ripples.shift();
      },
      { passive: true }
    );
    const release = (e) => {
      if (e.pointerType !== "mouse") pointer.active = false; // a finger that lifts is gone
    };
    window.addEventListener("pointerup", release, { passive: true });
    window.addEventListener("pointercancel", release, { passive: true });
    document.documentElement.addEventListener("pointerleave", () => (pointer.active = false));
    window.addEventListener("blur", () => (pointer.active = false));
    document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
    window.addEventListener("pagehide", saveState);
  }

  window.addEventListener("resize", () => {
    needsResize = true;
    if (reduceMotion) frame(performance.now());
  });

  // Re-read colours when the theme toggle flips data-theme.
  new MutationObserver(() => {
    readTheme();
    if (reduceMotion) frame(performance.now());
  }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    stop();
  });
  canvas.addEventListener("webglcontextrestored", () => {
    clouds = [];
    dust = null;
    mode = null;
    needsResize = true;
    setup();
    start();
  });

  try {
    setup();
  } catch (e) {
    console.warn("point-cloud:", e);
    return;
  }
  if (!reduceMotion) loadState();
  readTheme();
  start();
})();
