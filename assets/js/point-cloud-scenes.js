// Scenes for the background (assets/js/point-cloud.js). Each scene is an object with
//   id, name             used by the demo panel on localhost
//   variants             optional [{ name, ...settings }]: point styles of the scene, each may override settings
//   settings             optional overrides of DEFAULTS in point-cloud.js (layout, points, tilt, size, ...)
//   make(count, variant) a Float32Array of per-point data (4 numbers per point, handed to the shader as `d`)
//   glsl                 defines vec4 shape(vec4 d, vec4 r, float time, out float glow): the point's position at `time`
//                        (xyz, in units of the scene's radius; screen y points down) and its colour coordinate (w, 0..1,
//                        low to high along --cloud-1/2/3). glow scales brightness (1 = normal). r holds four uniform
//                        randoms per point, uVariant the variant index and uScroll the scroll position in screen heights.
// The "network" scene is drawn on the CPU instead (nodes and the links between them), from 10 numbers per node.
window.PointCloudScenes = (() => {
  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
  }

  // ---------------------------------------------------------------------------
  // Ocean: a field of dots rolling in slow waves along the bottom of the screen
  // ---------------------------------------------------------------------------

  function ocean(count) {
    const out = new Float32Array(count * 4);
    const cols = Math.round(Math.sqrt(count * 1.8));
    const rows = Math.max(2, Math.floor(count / cols));
    for (let i = 0; i < count; i++) {
      const done = i < rows * cols;
      const x = done ? -1.8 + (3.6 * ((i % cols) + 0.5)) / cols : Math.random() * 3.6 - 1.8;
      const z = done ? -1 + (2 * (Math.floor(i / cols) + 0.5)) / rows : Math.random() * 2 - 1;
      out.set([x, z, 0, 0], i * 4);
    }
    return out;
  }

  const OCEAN = `
    vec4 shape(vec4 d, vec4 r, float time, out float glow) {
      float x = d.x;
      float z = d.y;
      float h = 0.075 * sin(x * 2.3 + time * 0.8) + 0.05 * sin(z * 3.7 - time * 0.6 + 1.3)
              + 0.03 * sin((x + z) * 6.0 + time * 1.2) + 0.018 * sin(x * 9.0 - z * 4.0 - time * 1.6);
      glow = 0.65 + 1.0 * smoothstep(0.0, 0.14, h);
      return vec4(x, -h, z, clamp(0.5 + h * 3.6, 0.0, 1.0));
    }`;

  // ---------------------------------------------------------------------------
  // Network: drifting nodes linked to their neighbours (drawn on the CPU, see point-cloud.js)
  // ---------------------------------------------------------------------------

  // Per node: base position (3, uniformly at random in a slab wider than the screen), three randoms for the drift (3),
  // colour coordinate (1), noise (3).
  function network(count) {
    const out = new Float32Array(count * 10);
    for (let i = 0; i < count; i++) {
      const base = [Math.random() * 2.5 - 1.25, Math.random() * 1.6 - 0.8, Math.random() * 1.6 - 0.8];
      out.set([...base, Math.random(), Math.random(), Math.random(), Math.random(), randn(), randn(), randn()], i * 10);
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // Dot grid: a uniform grid of dots, each nudged off its grid point and wandering slightly around it
  // ---------------------------------------------------------------------------

  function dotGrid(count) {
    const out = new Float32Array(count * 4);
    const side = Math.max(2, Math.round(Math.sqrt(count)));
    const step = 2 / side;
    for (let i = 0; i < count; i++) {
      const x = i < side * side ? -1 + ((i % side) + 0.5) * step : Math.random() * 2 - 1;
      const y = i < side * side ? -1 + (Math.floor(i / side) + 0.5) * step : Math.random() * 2 - 1;
      // d = (grid point, random offset of up to 0.12 grid steps)
      out.set([x, y, (Math.random() - 0.5) * 0.24 * step, (Math.random() - 0.5) * 0.24 * step], i * 4);
    }
    return out;
  }

  const DOT_GRID = `
    vec4 shape(vec4 d, vec4 r, float time, out float glow) {
      vec2 p = d.xy + d.zw + 0.002 * vec2(sin(time * (0.3 + 0.4 * r.x) + r.y * TAU), cos(time * (0.25 + 0.4 * r.y) + r.x * TAU));
      p.y = mod(p.y + 1.0 - uScroll * 0.12, 2.0) - 1.0; // moves a little with the page (the seam is off-screen)
      // A soft band of light sweeps diagonally across the grid.
      float band = 0.5 + 0.5 * sin(dot(p, vec2(0.8, 0.6)) * 3.0 - time * 0.4);
      glow = 0.55 + 1.1 * band * band * band;
      return vec4(p, 0.0, clamp(0.5 + 0.4 * (p.x - p.y), 0.0, 1.0));
    }`;

  return [
    {
      id: "ocean",
      name: "Ocean",
      settings: {
        layout: "floor",
        points: { mobile: 12000, desktop: 32000 },
        tilt: 0.42,
        mouseX: 0.06,
        mouseY: 0.04,
        fov: 2.4,
        size: 3.1,
        alpha: 1.35,
        behind: 0.17,
        dof: 1.6,
      },
      make: ocean,
      glsl: OCEAN,
    },
    {
      id: "network",
      name: "Network",
      kind: "network",
      settings: { points: { mobile: 160, desktop: 450 }, link: 0.29, sway: 0.3, tilt: 0.08, mouseX: 0.25, mouseY: 0.15, scrollSpin: 0.0004 },
      make: network,
    },
    {
      id: "grid",
      name: "Dot grid",
      settings: { points: { mobile: 3000, desktop: 7000 }, mouseX: 0.08, mouseY: 0.06, size: 3.6, alpha: 2.2, dof: 1 },
      make: dotGrid,
      glsl: DOT_GRID,
    },
  ];
})();
