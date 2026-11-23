// Interactive background (WebGL): one of the scenes in assets/js/point-cloud-scenes.js, thousands of points whose
// motion is computed on the GPU.
//
// Each scene condenses out of Gaussian noise when it appears, like a diffusion model drawing a sample:
//   x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps_t     (cosine schedule)
// where eps_t = cos(theta) * eps_1 + sin(theta) * eps_2 rotates slowly between two independent Gaussian samples, so the
// noise swirls while staying exactly N(0, I). While the cursor moves, it pulls the points near it in a little, like a
// small weight; a click or tap sends a ripple, and fast scrolling shakes a little noise back in. Points behind the
// text column are dimmed so the text stays readable.
//
// Each time someone arrives on the site (a new tab, a link from elsewhere, a reload) one of the scenes is picked at
// random, never the same as last time; it stays while they move between pages of the site.
// On localhost a small panel lets you switch scenes (for the rest of that visit). Colours come from CSS variables (assets/css/main.scss): --cloud-1/2/3 ("r, g, b" from low to
// high values of each scene's colour coordinate), --cloud-alpha, and --cloud-blend (add | normal).
(() => {
  const canvas = document.getElementById("point-cloud");
  const SCENES = window.PointCloudScenes;
  if (!canvas || !canvas.getContext || !SCENES || !SCENES.length) return;
  const gl = canvas.getContext("webgl", { alpha: true, premultipliedAlpha: true, antialias: false, depth: false, stencil: false });
  if (!gl) return; // no WebGL: the CSS background is all there is
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const LAB = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);

  const DEFAULT = { scene: "ocean", variant: 0 }; // fallback for an unknown scene id
  const CONTENT_WIDTH = 760; // $max-width in assets/css/main.scss
  const BEHIND_TEXT = 0.3; // brightness of the points behind the content column
  const PULL = 0.3; // the cursor draws a point in by at most this fraction of its distance...
  const PULL_RANGE = { mobile: 55, desktop: 70 }; // ...at this distance (px); less nearer to it and further away
  const PULL_IDLE = 1.5; // seconds after the last move before the cursor lets go
  const RIPPLES = 4;
  const RIPPLE_LIFE = 1.5;
  const RIPPLE_SPEED = 620; // px / s
  const OUT = 1.1; // seconds to dissolve a scene into noise
  const IN = 3.2; // seconds to condense the next one out of it
  const STATE_KEY = "point-cloud-scene"; // sessionStorage: carries the animation over to the next page
  const PICK_KEY = "point-cloud-pick"; // sessionStorage: the scene picked for this visit
  const LAST_KEY = "point-cloud-last"; // localStorage: the scene shown last, not to be picked twice in a row

  // What every scene starts from; scenes and their variants override some of it.
  const DEFAULTS = {
    layout: "full", // full | floor (see slots())
    points: { mobile: 4000, desktop: 8000 },
    spin: 0, // rad / s around the vertical axis
    sway: 0, // amplitude (rad) of a slow back-and-forth turn
    tilt: 0, // rad around the horizontal axis
    wobble: 0, // amplitude (rad) of a slow nod
    mouseX: 0.15, // how far the cursor turns the scene
    mouseY: 0.1,
    scrollSpin: 0, // rad per scrolled px
    bob: 0, // px of slow up-and-down drift
    size: 3.1, // px
    dof: 2.4, // px of blur per unit of depth away from the focal plane
    focus: 0, // depth of the focal plane
    fov: 4, // camera distance: smaller = stronger perspective
    noise: 0.55, // standard deviation of the noise the scene condenses out of
    alpha: 1,
    behind: BEHIND_TEXT, // brightness of the points behind the content column
    link: 0.34, // network: link length
    drift: 0.07, // network: how far nodes wander
  };

  // ---------------------------------------------------------------------------
  // Shaders
  // ---------------------------------------------------------------------------

  const HEADER = `
    precision highp float;
    #define RIPPLES ${RIPPLES}

    attribute vec4 aData;  // per-point data from the scene's make()
    attribute vec3 aEps1;  // two independent N(0, I) samples
    attribute vec3 aEps2;
    attribute vec4 aRand;  // uniform randoms in [0, 1)

    uniform float uTime;
    uniform float uT;        // diffusion time in [0, 1] (1 = pure noise)
    uniform float uKick;     // extra noise from scrolling
    uniform float uNoise;
    uniform float uVariant;
    uniform float uScroll;   // scroll position in screen heights
    uniform mat3 uRot;
    uniform vec2 uCenter;    // px
    uniform float uRadius;   // px
    uniform float uFov;
    uniform vec2 uViewport;  // px
    uniform float uDpr;
    uniform float uSize;
    uniform float uAlpha;
    uniform float uDof;
    uniform float uFocus;
    uniform vec3 uCursor;    // x, y (px) and strength (0 while the cursor rests)
    uniform float uRange;    // px, distance at which the cursor pulls hardest
    uniform float uPull;
    uniform vec4 uRipple[RIPPLES]; // x, y, radius, amplitude
    uniform vec4 uMask;      // content column: left, right, feather (px), brightness behind it
    uniform vec3 uColA;
    uniform vec3 uColB;
    uniform vec3 uColC;

    varying vec4 vColor;
    varying float vBlur;

    const float TAU = 6.2831853;

    float hash(float x) { return fract(sin(x * 12.9898 + 78.233) * 43758.5453); }

    vec2 rotate2(vec2 p, float a) {
      float c = cos(a), s = sin(a);
      return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
    }

    vec3 rotateAxis(vec3 p, vec3 axis, float a) {
      float c = cos(a), s = sin(a);
      return p * c + cross(axis, p) * s + axis * dot(axis, p) * (1.0 - c);
    }

    // Point at s in [0, 1) along the outline of the regular n-gon inscribed in the unit circle (evenly spaced).
    vec2 polygonPoint(float s, float n) {
      float e = floor(s * n);
      float k = TAU / n;
      return mix(vec2(cos(e * k), sin(e * k)), vec2(cos(e * k + k), sin(e * k + k)), fract(s * n));
    }

    // 1 at a vertex, falling quickly to 0 along the edges.
    float vertexGlow(float s, float n) {
      return pow(abs(fract(s * n) - 0.5) * 2.0, 12.0);
    }
  `;

  const MAIN = `
    // Cosine schedule (Nichol & Dhariwal, 2021): abar(0) = 1, abar(1) = 0.
    float alphaBar(float t) {
      float f = cos((t + 0.008) / 1.008 * 1.5707963);
      return clamp(f * f / 0.999845, 0.0, 1.0);
    }

    // Rotate, then perspective-project: returns screen px (xy) and depth (z, negative = towards the viewer).
    vec3 project(vec3 x) {
      vec3 v = uRot * x;
      return vec3(uCenter + v.xy * (uRadius * uFov / max(uFov + v.z, 0.25 * uFov)), v.z);
    }

    void main() {
      float glow = 1.0;
      vec4 x0 = shape(aData, aRand, uTime, glow);
      float theta = uTime * (0.2 + 0.5 * aRand.x) + aRand.y * TAU;
      vec3 eps = (cos(theta) * aEps1 + sin(theta) * aEps2) * uNoise;

      // Points are a little out of step with each other, so the sample condenses rather than snaps into place.
      float t = clamp((uT - 0.22 * aRand.z) / 0.78, 0.0, 1.0);
      t = min(1.0, t + uKick * (0.3 + 0.7 * aRand.w));
      float ab = alphaBar(t);
      vec3 p = project(sqrt(ab) * x0.xyz + sqrt(1.0 - ab) * eps);

      // The cursor is a small weight: points near it are drawn towards it, smoothly and the same way every time. The
      // displacement uPull * r * exp(-r^2 / 2 uRange^2) is largest at r = uRange and never carries a point past the cursor.
      vec2 toCursor = uCursor.xy - p.xy;
      float well = uCursor.z * exp(-dot(toCursor, toCursor) / (2.0 * uRange * uRange));
      p.xy += toCursor * (uPull * well);

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

      float noise = sqrt(1.0 - ab);
      float scale = uFov / max(uFov + p.z, 0.25 * uFov);
      float depth = clamp(0.5 - p.z / 3.2, 0.0, 1.0); // 1 = front
      glow = mix(glow, 1.0, noise);

      // Depth of field: points off the focal plane, and noisy ones, spread into dimmer discs.
      float base = uSize * scale * (0.55 + 0.9 * depth) * (0.7 + 0.6 * aRand.w) * (0.8 + 0.2 * min(glow, 2.0));
      float blur = (abs(p.z - uFocus) + 0.45 * noise) * uDof;
      float size = base + blur;
      gl_PointSize = size * uDpr;
      vBlur = clamp(blur / size * 1.25, 0.0, 1.0);

      float twinkle = 0.82 + 0.18 * sin(uTime * (0.7 + 2.3 * aRand.x) + aRand.y * 40.0);
      float bright = (0.3 + 0.7 * pow(depth, 1.3)) * twinkle * glow * (1.0 + 1.5 * flash + 0.35 * well);
      float spread = pow(base / size, 1.2);
      float outside = max(uMask.x - p.x, p.x - uMask.y);
      float mask = mix(uMask.w, 1.0, smoothstep(-30.0, uMask.z, outside));
      float visible = step(0.25 * uFov, uFov + p.z); // hide points that come too close to the camera

      vec3 color = x0.w < 0.5 ? mix(uColA, uColB, x0.w * 2.0) : mix(uColB, uColC, x0.w * 2.0 - 1.0);
      color = mix(color, uColB, 0.55 * noise); // colour emerges with the structure
      vColor = vec4(color, uAlpha * bright * spread * mask * visible * (1.0 - 0.25 * noise));

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
      float sharp = exp(-2.6 * r2);                                       // in focus: a soft dot
      float disc = smoothstep(1.0, 0.55, r2) * (0.65 + 0.35 * r2) * 0.7;  // out of focus: a disc with a brighter rim
      float a = mix(sharp, disc, vBlur) * vColor.a;
      gl_FragColor = vec4(vColor.rgb * a, a); // premultiplied
    }`;

  // Screen-space points and lines (the network scene).
  const FLAT_VERTEX = `
    precision highp float;
    attribute vec2 aXY;
    attribute vec4 aColor;
    attribute float aSize;
    uniform vec2 uViewport;
    uniform float uDpr;
    uniform vec4 uMask;
    varying vec4 vColor;

    void main() {
      float outside = max(uMask.x - aXY.x, aXY.x - uMask.y);
      vColor = vec4(aColor.rgb, aColor.a * mix(uMask.w, 1.0, smoothstep(-30.0, uMask.z, outside)));
      gl_PointSize = aSize * uDpr;
      gl_Position = vec4(aXY.x / uViewport.x * 2.0 - 1.0, 1.0 - aXY.y / uViewport.y * 2.0, 0.0, 1.0);
    }`;

  const FLAT_FRAGMENT = `
    precision mediump float;
    uniform float uRound;
    varying vec4 vColor;

    void main() {
      float a = vColor.a;
      if (uRound > 0.5) {
        vec2 q = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(q, q);
        if (r2 > 1.0) discard;
        a *= exp(-1.7 * r2);
      }
      gl_FragColor = vec4(vColor.rgb * a, a);
    }`;

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  let width = 0;
  let height = 0;
  let dpr = 1;
  let mobile = false;
  let needsResize = true;

  let programs = new Map(); // scene id -> { program, u: uniform locations }
  let flat = null; // { program, u, buffer }
  let active = null; // the scene on screen: { scene, variant, settings, count, buffers..., rots: [] }
  let selected = DEFAULT; // the scene picked for this visit (or in the panel)
  let wanted = null; // the next scene, while the current one dissolves
  let phase = "in"; // in: condensing out of noise | idle | out: dissolving into noise before a switch
  let phaseTime = 0;
  let diffusionT = 1;
  let fade = 1;

  const theme = { colors: [new Float32Array(3), new Float32Array(3), new Float32Array(3)], alpha: 0.7, add: true };
  const pointer = { x: 0, y: 0, nx: 0, ny: 0, fx: 0, fy: 0, strength: 0, active: false, moved: 0, seen: false };
  const ripples = []; // [{ x, y, born }]
  const cursorData = new Float32Array(3);
  const rippleData = new Float32Array(RIPPLES * 4);
  const rotation = new Float32Array(9);
  let scrollY = window.scrollY;
  let kick = 0;
  let clock = 0;
  let last = 0;
  let ready = false;
  let restored = null;

  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  const smoothstep = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };

  function alphaBar(t) {
    const f = Math.cos(((t + 0.008) / 1.008) * (Math.PI / 2));
    return Math.min(1, Math.max(0, (f * f) / 0.999845));
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

  function palette(c, out, o) {
    const [a, b, d] = theme.colors;
    const from = c < 0.5 ? a : b;
    const to = c < 0.5 ? b : d;
    const f = c < 0.5 ? c * 2 : c * 2 - 1;
    for (let k = 0; k < 3; k++) out[o + k] = from[k] + (to[k] - from[k]) * f;
  }

  const findScene = (id) => SCENES.find((s) => s.id === id);

  // The scene to show for a selection (falling back to the default for unknown ids).
  function resolve(selection) {
    const scene = findScene(selection.scene) || findScene(DEFAULT.scene) || SCENES[0];
    return { scene: scene.id, variant: scene.id === selection.scene ? selection.variant || 0 : 0 };
  }

  // ---------------------------------------------------------------------------
  // GL setup
  // ---------------------------------------------------------------------------

  function compile(vertex, fragment, attributes) {
    const shader = (type, source) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, source);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const program = gl.createProgram();
    gl.attachShader(program, shader(gl.VERTEX_SHADER, vertex));
    gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragment));
    attributes.forEach((name, i) => gl.bindAttribLocation(program, i, name)); // same slots in every program
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    const u = {};
    const n = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const name = gl.getActiveUniform(program, i).name.replace(/\[0\]$/, "");
      u[name] = gl.getUniformLocation(program, name);
    }
    return { program, u };
  }

  // One program per scene, compiled the first time it is shown.
  function programFor(scene) {
    if (!programs.has(scene.id)) programs.set(scene.id, compile(HEADER + scene.glsl + MAIN, FRAGMENT, ["aData", "aEps1", "aEps2", "aRand"]));
    return programs.get(scene.id);
  }

  // WebGL 1 keeps attribute arrays enabled across programs: enable exactly the ones in use.
  function useAttributes(count) {
    for (let i = 0; i < 4; i++) {
      if (i < count) gl.enableVertexAttribArray(i);
      else gl.disableVertexAttribArray(i);
    }
  }

  function buffer(data, usage = gl.STATIC_DRAW) {
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

  function setup() {
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.clearColor(0, 0, 0, 0);
    flat = compile(FLAT_VERTEX, FLAT_FRAGMENT, ["aXY", "aColor", "aSize"]);
    flat.buffer = gl.createBuffer();
  }

  function free(a) {
    if (a && a.fixed) [a.fixed, a.data].forEach((b) => gl.deleteBuffer(b));
  }

  // Put a scene on screen (its points start as whatever the current diffusion time says).
  function load(target) {
    const scene = findScene(target.scene) || findScene(DEFAULT.scene);
    const variants = scene.variants || [];
    const variant = Math.max(0, Math.min(target.variant || 0, variants.length - 1));
    const settings = { ...DEFAULTS, ...(scene.settings || {}), ...(variants[variant] || {}) };
    const count = settings.points[mobile ? "mobile" : "desktop"];
    free(active);
    active = { scene, variant, settings, count, rots: [] };
    if (scene.kind === "network") {
      active.nodes = scene.make(count, variant);
      active.push = new Float32Array(count * 4);
      active.world = new Float32Array(count * 3);
      active.screen = new Float32Array(count * 3);
      active.rgb = new Float32Array(count * 3);
      active.lines = new Float32Array(count * 10 * 14);
      active.points = new Float32Array(count * 7);
    } else {
      active.program = programFor(scene);
      active.fixed = buffer(fixedData(count));
      active.data = buffer(scene.make(count, variant));
    }
  }

  function resize() {
    needsResize = false;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = document.documentElement.clientWidth; // without the scrollbar
    height = window.innerHeight;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    const next = width < 760;
    if (next !== mobile || !active) {
      mobile = next;
      load(active ? { scene: active.scene.id, variant: active.variant } : resolve(selected));
    }
  }

  // ---------------------------------------------------------------------------
  // Layout
  // ---------------------------------------------------------------------------

  // Where the scene sits on screen: { cx, cy, radius } (radius = px per unit).
  function slots(layout) {
    if (layout === "floor") return [{ cx: width / 2, cy: height * (mobile ? 0.8 : 0.74), radius: Math.max(width * 0.5, height * 0.75) }];
    return [{ cx: width / 2, cy: height / 2, radius: 0.5 * Math.hypot(width, height) }]; // full screen
  }

  // ---------------------------------------------------------------------------
  // Animation
  // ---------------------------------------------------------------------------

  // Ask for another scene: the current one dissolves into noise first.
  function show(target) {
    if (!active || reduceMotion) {
      load(target);
      phase = reduceMotion ? "idle" : "in";
      phaseTime = 0;
      if (reduceMotion) frame(performance.now());
      return;
    }
    wanted = target;
    if (phase === "in")
      phaseTime = Math.max(0, 1 - phaseTime / IN) * OUT; // dissolve from where it is
    else if (phase === "idle") phaseTime = 0;
    phase = "out";
  }

  function advance(dt) {
    phaseTime += dt;
    if (phase === "out" && phaseTime >= OUT) {
      load(wanted);
      wanted = null;
      phase = "in";
      phaseTime = 0;
    } else if (phase === "in" && phaseTime >= IN) {
      phase = "idle";
      phaseTime = 0;
    }
    if (phase === "out") {
      const p = phaseTime / OUT;
      diffusionT = p;
      fade = 1 - smoothstep(0.6, 1, p);
    } else if (phase === "in") {
      const p = phaseTime / IN;
      diffusionT = 1 - p;
      fade = smoothstep(0, 0.25, p);
    } else {
      diffusionT = 0;
      fade = 1;
    }
  }

  // Follow the cursor with a little inertia; it holds on to the points only while it moves.
  function updatePointer(dt) {
    const ease = 1 - Math.exp(-dt * 14);
    pointer.fx += (pointer.x - pointer.fx) * ease;
    pointer.fy += (pointer.y - pointer.fy) * ease;
    const awake = pointer.active && clock - pointer.moved < PULL_IDLE; // a parked cursor lets go
    pointer.strength += ((awake ? 1 : 0) - pointer.strength) * (1 - Math.exp(-dt * (awake ? 10 : 2.5)));

    while (ripples.length && clock - ripples[0].born > RIPPLE_LIFE) ripples.shift();

    cursorData.set([pointer.fx, pointer.fy, pointer.strength]);
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
  }

  // Turn a slot toward its target orientation: drift, cursor, scroll.
  function orient(s, c, count, dt) {
    const spinSign = c === 0 && count === 2 ? -1 : 1; // mirror the left cloud
    const ty = spinSign * s.spin * clock + s.sway * Math.sin(clock * 0.07) + pointer.nx * s.mouseX + scrollY * s.scrollSpin;
    const tx = s.tilt - pointer.ny * s.mouseY + Math.sin(clock * 0.3) * s.wobble;
    let rot = active.rots[c];
    if (!rot) {
      const saved = restored && restored[c];
      rot = active.rots[c] = saved ? { x: saved[0], y: saved[1] } : { x: tx, y: ty };
    }
    const ease = reduceMotion ? 1 : 1 - Math.exp(-dt * 3);
    rot.y += (ty - rot.y) * ease;
    rot.x += (tx - rot.x) * ease;
    setRotation(rot.x, rot.y);
  }

  function setCommon(p, mask) {
    const u = p.u;
    gl.uniform1f(u.uTime, clock);
    gl.uniform2f(u.uViewport, width, height);
    gl.uniform1f(u.uDpr, dpr);
    gl.uniform3fv(u.uCursor, cursorData);
    gl.uniform1f(u.uRange, PULL_RANGE[mobile ? "mobile" : "desktop"]);
    gl.uniform1f(u.uPull, PULL);
    gl.uniform4fv(u.uRipple, rippleData);
    gl.uniform4fv(u.uMask, mask);
    gl.uniform3fv(u.uColA, theme.colors[0]);
    gl.uniform3fv(u.uColB, theme.colors[1]);
    gl.uniform3fv(u.uColC, theme.colors[2]);
    gl.uniform1f(u.uScroll, scrollY / Math.max(1, height));
  }

  function drawPoints(p, fixed, data, count) {
    useAttributes(4);
    gl.bindBuffer(gl.ARRAY_BUFFER, fixed);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 40, 0);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, 40, 12);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, 40, 24);
    gl.bindBuffer(gl.ARRAY_BUFFER, data);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.POINTS, 0, count);
  }

  // The network: nodes drift, links appear between the ones that come close, and the cursor links to its neighbours.
  function drawNetwork(slot, mask, t) {
    const a = active;
    const s = a.settings;
    const n = a.count;
    const { nodes, push, world, screen, rgb, lines, points } = a;
    const ka = Math.sqrt(alphaBar(t));
    const kb = Math.sqrt(1 - alphaBar(t)) * s.noise;
    const m = rotation;
    const reach = mobile ? 110 : 150;
    const alpha = theme.alpha * fade;

    for (let i = 0; i < n; i++) {
      const o = i * 10;
      const x = nodes[o] + s.drift * Math.sin(clock * (0.1 + 0.12 * nodes[o + 3]) + 6.283 * nodes[o + 4]);
      const y = nodes[o + 1] + s.drift * Math.sin(clock * (0.09 + 0.1 * nodes[o + 4]) + 6.283 * nodes[o + 5]);
      const z = nodes[o + 2] + s.drift * Math.sin(clock * (0.08 + 0.1 * nodes[o + 5]) + 6.283 * nodes[o + 3]);
      const wx = ka * x + kb * nodes[o + 7];
      const wy = ka * y + kb * nodes[o + 8];
      const wz = ka * z + kb * nodes[o + 9];
      world.set([wx, wy, wz], i * 3);
      const vx = m[0] * wx + m[3] * wy + m[6] * wz;
      const vy = m[1] * wx + m[4] * wy + m[7] * wz;
      const vz = m[2] * wx + m[5] * wy + m[8] * wz;
      const sc = s.fov / Math.max(s.fov + vz, 0.25 * s.fov);
      let px = slot.cx + vx * slot.radius * sc;
      let py = slot.cy + vy * slot.radius * sc;

      // Spring-back offset: pushed away from the cursor and by the ripples.
      const q = i * 4;
      const dx = px + push[q] - pointer.fx;
      const dy = py + push[q + 1] - pointer.fy;
      const d2 = dx * dx + dy * dy;
      if (pointer.strength > 0.05 && d2 < reach * reach && d2 > 0.01) {
        const d = Math.sqrt(d2);
        const f = (1 - d / reach) * 1.4 * pointer.strength;
        push[q + 2] += (dx / d) * f;
        push[q + 3] += (dy / d) * f;
      }
      for (let k = 0; k < RIPPLES; k++) {
        if (!rippleData[k * 4 + 3]) continue;
        const rx = px - rippleData[k * 4];
        const ry = py - rippleData[k * 4 + 1];
        const rl = Math.hypot(rx, ry) + 0.001;
        const ring = rippleData[k * 4 + 3] * Math.exp(-(((rl - rippleData[k * 4 + 2]) / 54) ** 2));
        push[q + 2] += (rx / rl) * ring * 3;
        push[q + 3] += (ry / rl) * ring * 3;
      }
      push[q + 2] = (push[q + 2] - push[q] * 0.06) * 0.82;
      push[q + 3] = (push[q + 3] - push[q + 1] * 0.06) * 0.82;
      push[q] += push[q + 2];
      push[q + 1] += push[q + 3];
      px += push[q];
      py += push[q + 1];

      const depth = Math.min(1, Math.max(0, 0.5 - vz / 3.2));
      screen.set([px, py, depth], i * 3);
      palette(nodes[o + 6], rgb, i * 3);
      const pt = i * 7;
      points[pt] = px;
      points[pt + 1] = py;
      points.set(rgb.subarray(i * 3, i * 3 + 3), pt + 2);
      points[pt + 5] = Math.min(1, alpha * (0.7 + 0.6 * depth));
      points[pt + 6] = (4 + 6 * depth) * (0.8 + 0.5 * nodes[o + 3]) * sc;
    }

    // Links between nodes closer than s.link (in 3D), fading with length.
    let count = 0;
    const max = lines.length / 14;
    const link2 = s.link * s.link;
    const put = (x1, y1, x2, y2, c1, c2, al) => {
      const o = count * 14;
      lines[o] = x1;
      lines[o + 1] = y1;
      lines.set(rgb.subarray(c1, c1 + 3), o + 2);
      lines[o + 5] = al;
      lines[o + 6] = 1;
      lines[o + 7] = x2;
      lines[o + 8] = y2;
      lines.set(rgb.subarray(c2, c2 + 3), o + 9);
      lines[o + 12] = al;
      lines[o + 13] = 1;
      count++;
    };
    for (let i = 0; i < n && count < max; i++) {
      for (let j = i + 1; j < n && count < max; j++) {
        const dx = world[i * 3] - world[j * 3];
        const dy = world[i * 3 + 1] - world[j * 3 + 1];
        const dz = world[i * 3 + 2] - world[j * 3 + 2];
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > link2) continue;
        const k = 1 - Math.sqrt(d2) / s.link;
        const al = alpha * 0.7 * k * (0.35 + 0.33 * (screen[i * 3 + 2] + screen[j * 3 + 2]));
        put(screen[i * 3], screen[i * 3 + 1], screen[j * 3], screen[j * 3 + 1], i * 3, j * 3, al);
      }
    }
    // The cursor joins the network.
    if (pointer.strength > 0.05) {
      for (let i = 0; i < n && count < max; i++) {
        const d = Math.hypot(screen[i * 3] - pointer.fx, screen[i * 3 + 1] - pointer.fy);
        if (d > reach * 1.4) continue;
        const k = 1 - d / (reach * 1.4);
        put(pointer.fx, pointer.fy, screen[i * 3], screen[i * 3 + 1], i * 3, i * 3, alpha * 0.8 * k * k * pointer.strength);
      }
    }

    gl.useProgram(flat.program);
    useAttributes(3);
    gl.uniform2f(flat.u.uViewport, width, height);
    gl.uniform1f(flat.u.uDpr, dpr);
    gl.uniform4fv(flat.u.uMask, mask);
    gl.bindBuffer(gl.ARRAY_BUFFER, flat.buffer);
    const layout = () => {
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 28, 0);
      gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 28, 8);
      gl.vertexAttribPointer(2, 1, gl.FLOAT, false, 28, 24);
    };
    if (count) {
      gl.bufferData(gl.ARRAY_BUFFER, lines.subarray(0, count * 14), gl.DYNAMIC_DRAW);
      layout();
      gl.uniform1f(flat.u.uRound, 0);
      gl.drawArrays(gl.LINES, 0, count * 2);
    }
    gl.bufferData(gl.ARRAY_BUFFER, points.subarray(0, n * 7), gl.DYNAMIC_DRAW);
    layout();
    gl.uniform1f(flat.u.uRound, 1);
    gl.drawArrays(gl.POINTS, 0, n);
  }

  function frame(now) {
    // Clamped: two calls with the same timestamp must not divide by zero below (a NaN kick would stick for good).
    const dt = last ? Math.min(Math.max((now - last) / 1000, 0.001), 0.05) : 0.016;
    last = now;
    clock += dt;
    if (needsResize) resize();
    if (!reduceMotion) {
      advance(dt);
      updatePointer(dt);
    } else {
      diffusionT = 0;
      fade = 1;
    }

    // Scrolling fast shakes some noise back in.
    const y = window.scrollY;
    const speed = Math.abs(y - scrollY) / dt;
    scrollY = y;
    kick += (Math.min(0.14, speed / 16000) - kick) * (1 - Math.exp(-dt * 5));
    if (!Number.isFinite(kick)) kick = 0;

    gl.clear(gl.COLOR_BUFFER_BIT);
    // Dark theme: points add up their light (glow). Light theme: ordinary alpha blending (ink).
    gl.blendFunc(gl.ONE, theme.add ? gl.ONE : gl.ONE_MINUS_SRC_ALPHA);
    if (!active) return;
    const s = active.settings;
    const left = (width - CONTENT_WIDTH) / 2;
    const mask = mobile ? [-1e5, 1e5, 1, s.behind] : [left, left + CONTENT_WIDTH, 160, s.behind];

    const places = slots(s.layout);
    places.forEach((slot, c) => {
      orient(s, c, places.length, dt);
      if (active.nodes) return drawNetwork(slot, mask, diffusionT);
      const p = active.program;
      gl.useProgram(p.program);
      setCommon(p, mask);
      gl.uniformMatrix3fv(p.u.uRot, false, rotation);
      gl.uniform1f(p.u.uT, diffusionT);
      gl.uniform1f(p.u.uKick, kick);
      gl.uniform1f(p.u.uNoise, s.noise);
      gl.uniform1f(p.u.uVariant, active.variant);
      gl.uniform2f(p.u.uCenter, slot.cx, slot.cy + Math.sin(clock * 0.45 + c * 2) * s.bob);
      gl.uniform1f(p.u.uRadius, slot.radius);
      gl.uniform1f(p.u.uFov, s.fov);
      gl.uniform1f(p.u.uSize, s.size * (mobile ? 1.1 : 1));
      gl.uniform1f(p.u.uAlpha, theme.alpha * s.alpha * fade);
      gl.uniform1f(p.u.uDof, s.dof * (mobile ? 0.7 : 1));
      gl.uniform1f(p.u.uFocus, s.focus);
      drawPoints(p, active.fixed, active.data, active.count);
    });
    restored = null;

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
  // Carry the animation over to the next page, and the demo panel's choice
  // ---------------------------------------------------------------------------

  function saveState() {
    if (!active) return;
    try {
      const rots = active.rots.map((r) => [r.x, r.y]);
      const state = { scene: active.scene.id, variant: active.variant, clock, rots, at: Date.now() };
      sessionStorage.setItem(STATE_KEY, JSON.stringify(state));
    } catch (e) {}
  }

  // Same scene within a few seconds: no new entrance, just carry on.
  function loadState() {
    try {
      const state = JSON.parse(sessionStorage.getItem(STATE_KEY));
      if (!state || Date.now() - state.at > 4000) return;
      const target = resolve(selected);
      if (state.scene !== target.scene || state.variant !== (target.variant || 0)) return;
      clock = state.clock;
      restored = state.rots;
      phase = "idle";
    } catch (e) {}
  }

  // Keep the scene while the visitor moves between pages of the site; pick a new one, different from the last one
  // shown, whenever they arrive (new tab, typed address, link from elsewhere, reload).
  function pickScene() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(PICK_KEY));
      if (saved && findScene(saved.scene) && browsingWithin()) return { scene: saved.scene, variant: saved.variant || 0 };
    } catch (e) {}
    let last = null;
    try {
      last = localStorage.getItem(LAST_KEY);
    } catch (e) {}
    const pool = SCENES.filter((s) => s.id !== last);
    const choices = pool.length ? pool : SCENES;
    const scene = choices[Math.floor(Math.random() * choices.length)];
    const variants = (scene.variants || []).length;
    const pick = { scene: scene.id, variant: variants ? Math.floor(Math.random() * variants) : 0 };
    savePick(pick);
    return pick;
  }

  function savePick(pick) {
    try {
      sessionStorage.setItem(PICK_KEY, JSON.stringify(pick));
      localStorage.setItem(LAST_KEY, pick.scene);
    } catch (e) {}
  }

  // Did this page open from another page of the site (a link, or back / forward), rather than as an arrival?
  function browsingWithin() {
    const nav = performance.getEntriesByType ? performance.getEntriesByType("navigation")[0] : null;
    if (nav && nav.type === "reload") return false;
    if (nav && nav.type === "back_forward") return true;
    try {
      return new URL(document.referrer).origin === location.origin;
    } catch (e) {
      return false; // no referrer: typed address, bookmark, new tab
    }
  }

  function buildPanel() {
    const style = document.createElement("style");
    style.textContent = `
      .cloud-lab { position: fixed; left: 16px; bottom: 16px; z-index: 50; max-width: calc(100vw - 32px); padding: 10px 12px 12px;
        border-radius: 14px; border: 1px solid var(--border); background: color-mix(in srgb, var(--bg) 80%, transparent);
        backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: 0 14px 40px -18px rgba(0, 0, 0, 0.6);
        font: 12px/1.3 var(--font-mono); color: var(--muted); }
      .cloud-lab-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 8px;
        font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase; }
      .cloud-lab-row { display: flex; flex-wrap: wrap; gap: 5px; }
      .cloud-lab-row + .cloud-lab-row { margin-top: 6px; padding-top: 6px; border-top: 1px dashed var(--border); }
      .cloud-lab-row:empty { display: none; }
      .cloud-lab button { font: inherit; color: var(--muted); background: var(--surface); border: 1px solid var(--border);
        border-radius: 999px; padding: 3px 10px; cursor: pointer; transition: color 0.15s, border-color 0.15s; }
      .cloud-lab button:hover { color: var(--text); border-color: var(--border-strong); }
      .cloud-lab button[aria-pressed="true"] { color: var(--accent); border-color: var(--accent); background: var(--accent-soft); }
      .cloud-lab .cloud-lab-toggle { padding: 0 7px; border-radius: 6px; }
      .cloud-lab.is-collapsed .cloud-lab-row { display: none; }
      .cloud-lab.is-collapsed .cloud-lab-head { margin: 0; }`;
    document.head.append(style);

    const panel = document.createElement("div");
    panel.className = "cloud-lab";
    panel.innerHTML = `<div class="cloud-lab-head"><span>Background · local demo</span>
      <button type="button" class="cloud-lab-toggle" aria-label="Collapse">–</button></div>
      <div class="cloud-lab-row" data-scenes></div><div class="cloud-lab-row" data-variants></div>`;
    const scenesRow = panel.querySelector("[data-scenes]");
    const variantsRow = panel.querySelector("[data-variants]");
    const button = (label, pressed, onClick) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = label;
      b.setAttribute("aria-pressed", String(pressed));
      b.addEventListener("click", onClick);
      return b;
    };
    const save = () => savePick(selected);
    const render = () => {
      scenesRow.replaceChildren(
        ...SCENES.map((scene) =>
          button(scene.name, scene.id === selected.scene, () => {
            if (scene.id === selected.scene) return;
            selected = { scene: scene.id, variant: 0 };
            save();
            render();
            show(resolve(selected));
          })
        )
      );
      const scene = findScene(selected.scene);
      variantsRow.replaceChildren(
        ...(scene.variants || []).map((v, k) =>
          button(v.name, k === selected.variant, () => {
            if (k === selected.variant) return;
            selected = { scene: scene.id, variant: k };
            save();
            render();
            show(resolve(selected));
          })
        )
      );
    };
    const toggle = panel.querySelector(".cloud-lab-toggle");
    toggle.addEventListener("click", () => {
      const collapsed = panel.classList.toggle("is-collapsed");
      toggle.textContent = collapsed ? "+" : "–";
      toggle.setAttribute("aria-label", collapsed ? "Expand" : "Collapse");
    });
    render();
    document.body.append(panel);
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
        if (e.button > 0 || (e.target.closest && e.target.closest(".cloud-lab"))) return;
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
    const current = active ? { scene: active.scene.id, variant: active.variant } : resolve(selected);
    programs = new Map();
    active = null;
    try {
      setup();
      load(current);
    } catch (e) {
      return;
    }
    start();
  });

  selected = pickScene();
  if (LAB) buildPanel();
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
