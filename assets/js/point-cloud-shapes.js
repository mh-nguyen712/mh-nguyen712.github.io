// Datasets for the point-cloud background (assets/js/point-cloud.js), shown in this order.
// Each entry is { name, tilt, make(count) }: make returns a Float32Array of 3 * count coordinates, roughly inside the
// unit ball. Screen y points DOWN, so "up" is -y. `tilt` (radians, optional) tips the shape toward the viewer, e.g. to
// look down onto a height map. To add a shape, add an entry to SHAPES at the bottom.
window.PointCloudShapes = (() => {
  const TAU = 2 * Math.PI;

  function randn() {
    let u = 0;
    while (u === 0) u = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(TAU * Math.random());
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  // Centre the points (bounding box) and scale them so the farthest one sits at `radius`.
  function normalize(out, radius = 1.05) {
    const n = out.length / 3;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++)
      for (let d = 0; d < 3; d++) {
        const v = out[i * 3 + d];
        if (v < lo[d]) lo[d] = v;
        if (v > hi[d]) hi[d] = v;
      }
    const c = lo.map((v, d) => (v + hi[d]) / 2);
    let far = 0;
    for (let i = 0; i < n; i++) {
      const dx = out[i * 3] - c[0];
      const dy = out[i * 3 + 1] - c[1];
      const dz = out[i * 3 + 2] - c[2];
      far = Math.max(far, dx * dx + dy * dy + dz * dz);
    }
    const s = radius / (Math.sqrt(far) || 1);
    for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) out[i * 3 + d] = (out[i * 3 + d] - c[d]) * s;
    return out;
  }

  // Sample a parametric surface f(u, v) -> [x, y, z] on a regular grid over [0, 1)^2, with about `aspect` times more
  // samples along u than along v. Leftover slots get random (u, v).
  function grid(count, aspect, f) {
    const nv = Math.max(3, Math.round(Math.sqrt(count / aspect)));
    const nu = Math.max(3, Math.floor(count / nv));
    const out = new Float32Array(count * 3);
    let i = 0;
    const put = (p) => {
      out[i * 3] = p[0];
      out[i * 3 + 1] = p[1];
      out[i * 3 + 2] = p[2];
      i++;
    };
    for (let a = 0; a < nu && i < count; a++) for (let b = 0; b < nv && i < count; b++) put(f((a + 0.5) / nu, (b + 0.5) / nv));
    while (i < count) put(f(Math.random(), Math.random()));
    return out;
  }

  // Rescale raw height values into y so the surface spans `height`, with the largest values pointing up.
  function setHeights(out, raw, height) {
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of raw) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const span = hi - lo || 1;
    raw.forEach((v, i) => (out[i * 3 + 1] = height / 2 - (height * (v - lo)) / span));
    return out;
  }

  // Height map h(r, theta) over the unit disk, sampled on concentric rings (a little denser towards the centre).
  function disk(count, h, height) {
    const rings = Math.max(8, Math.round(Math.sqrt(count / 2.2)));
    const radii = Array.from({ length: rings }, (_, k) => ((k + 1) / rings) ** 1.15);
    const total = radii.reduce((s, r) => s + r, 0);
    const out = new Float32Array(count * 3);
    const raw = new Float32Array(count);
    let i = 0;
    const put = (r, t) => {
      out[i * 3] = r * Math.cos(t);
      out[i * 3 + 2] = r * Math.sin(t);
      raw[i] = h(r, t);
      i++;
    };
    put(0, 0);
    for (const r of radii) {
      const m = Math.max(6, Math.floor((r / total) * (count - 1)));
      for (let j = 0; j < m && i < count; j++) put(r, (j / m) * TAU);
    }
    while (i < count) put(1, Math.random() * TAU); // rounding leftovers go on the rim
    return normalize(setHeights(out, raw, height), 1.1);
  }

  // Height map h(x, z) over the square [-1, 1]^2 (e.g. an optimization landscape).
  function square(count, h, height) {
    const out = grid(count, 1, (u, v) => [2 * u - 1, 0, 2 * v - 1]);
    const raw = new Float32Array(count);
    for (let i = 0; i < count; i++) raw[i] = h(out[i * 3], out[i * 3 + 2]);
    return normalize(setHeights(out, raw, height), 1.1);
  }

  // ---------------------------------------------------------------------------
  // Machine learning
  // ---------------------------------------------------------------------------

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
    // Edges: a random subset of connections between consecutive layers, ~10 points each.
    const edges = [];
    const edgeCount = Math.ceil((count - nodePoints) / 10);
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

  // ---------------------------------------------------------------------------
  // Optics: Airy pattern and Zernike polynomials (wavefront aberrations)
  // ---------------------------------------------------------------------------

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

  // In-focus PSF of a circular aperture, I(v) = (2 J1(v) / v)^2, out to the 4th dark ring (v = 13.3).
  // Height ∝ I^0.35 so the faint diffraction rings stay visible next to the main lobe.
  function airyPattern(count) {
    return disk(
      count,
      (r) => {
        const v = r * 13.3;
        const intensity = v < 1e-9 ? 1 : ((2 * besselJ1(v)) / v) ** 2;
        return intensity ** 0.35;
      },
      1.3
    );
  }

  const factorial = (k) => (k <= 1 ? 1 : k * factorial(k - 1));

  // Zernike polynomial Z_n^m on the unit disk (m < 0 uses sin(|m| theta)).
  function zernike(n, m, r, theta) {
    const am = Math.abs(m);
    let radial = 0;
    for (let k = 0; k <= (n - am) / 2; k++) {
      radial += ((-1) ** k * factorial(n - k) * r ** (n - 2 * k)) / (factorial(k) * factorial((n + am) / 2 - k) * factorial((n - am) / 2 - k));
    }
    return radial * (m >= 0 ? Math.cos(am * theta) : Math.sin(am * theta));
  }

  const zernikeShape = (n, m, name) => ({ name, tilt: 0.45, make: (count) => disk(count, (r, t) => zernike(n, m, r, t), 0.75) });

  // ---------------------------------------------------------------------------
  // Optimization landscapes (height = objective value; minima at the bottom)
  // ---------------------------------------------------------------------------

  // Rosenbrock's banana valley, (1 - x)^2 + 100 (y - x^2)^2 on [-2, 2] x [-1, 3], log scale.
  const rosenbrock = (x, z) => {
    const X = 2 * x;
    const Y = 2 * z + 1;
    return Math.log1p((1 - X) ** 2 + 100 * (Y - X * X) ** 2);
  };

  // Himmelblau: four equal minima, on [-5, 5]^2, log scale.
  const himmelblau = (x, z) => {
    const X = 5 * x;
    const Y = 5 * z;
    return Math.log1p((X * X + Y - 11) ** 2 + (X + Y * Y - 7) ** 2);
  };

  // Rastrigin: a lattice of local minima around the global one, on [-3, 3]^2.
  const rastrigin = (x, z) => {
    const X = 3 * x;
    const Y = 3 * z;
    return 20 + X * X - 10 * Math.cos(TAU * X) + Y * Y - 10 * Math.cos(TAU * Y);
  };

  // Monkey saddle x^3 - 3 x y^2: a degenerate saddle point with three "valleys".
  const monkeySaddle = (x, z) => x ** 3 - 3 * x * z * z;

  // ---------------------------------------------------------------------------
  // Classic surfaces
  // ---------------------------------------------------------------------------

  function mobiusStrip(count) {
    return normalize(
      grid(count, 7, (u, v) => {
        const t = u * TAU;
        const s = (v - 0.5) * 0.9;
        const r = 1 + s * Math.cos(t / 2);
        return [r * Math.cos(t), r * Math.sin(t), s * Math.sin(t / 2)];
      })
    );
  }

  // Klein bottle, the "bottle" immersion in R^3 (u in [0, pi), v in [0, 2 pi)).
  function kleinBottle(count) {
    return normalize(
      grid(count, 2.5, (a, b) => {
        const u = a * Math.PI;
        const v = b * TAU;
        const cu = Math.cos(u);
        const su = Math.sin(u);
        const cv = Math.cos(v);
        const x = (-2 / 15) * cu * (3 * cv - 30 * su + 90 * cu ** 4 * su - 60 * cu ** 6 * su + 5 * cu * cv * su);
        const y =
          (-1 / 15) *
          su *
          (3 * cv -
            3 * cu ** 2 * cv -
            48 * cu ** 4 * cv +
            48 * cu ** 6 * cv -
            60 * su +
            5 * cu * cv * su -
            5 * cu ** 3 * cv * su -
            80 * cu ** 5 * cv * su +
            80 * cu ** 7 * cv * su);
        const z = (2 / 15) * (3 + 5 * cu * su) * Math.sin(v);
        return [x, y, z];
      })
    );
  }

  // A tube around the trefoil knot (sin t + 2 sin 2t, cos t - 2 cos 2t, -sin 3t).
  function trefoilKnot(count) {
    const tube = 0.42;
    return normalize(
      grid(count, 12, (a, b) => {
        const t = a * TAU;
        const phi = b * TAU;
        const c = [Math.sin(t) + 2 * Math.sin(2 * t), Math.cos(t) - 2 * Math.cos(2 * t), -Math.sin(3 * t)];
        const d = [Math.cos(t) + 4 * Math.cos(2 * t), -Math.sin(t) + 4 * Math.sin(2 * t), -3 * Math.cos(3 * t)];
        const dl = Math.hypot(...d);
        const T = d.map((x) => x / dl);
        // Normal = T x (0, 0, 1) (never degenerate for this knot), binormal = T x N.
        const nl = Math.hypot(T[1], T[0]);
        const N = [T[1] / nl, -T[0] / nl, 0];
        const B = [T[1] * N[2] - T[2] * N[1], T[2] * N[0] - T[0] * N[2], T[0] * N[1] - T[1] * N[0]];
        const cp = Math.cos(phi) * tube;
        const sp = Math.sin(phi) * tube;
        return [c[0] + cp * N[0] + sp * B[0], c[1] + cp * N[1] + sp * B[1], c[2] + cp * N[2] + sp * B[2]];
      })
    );
  }

  // Enneper's minimal surface, sampled over a disk of radius 1.25 in the parameter plane.
  function enneper(count) {
    return normalize(
      grid(count, 4, (a, b) => {
        const t = a * TAU;
        const r = 1.25 * b ** 0.6;
        const u = r * Math.cos(t);
        const v = r * Math.sin(t);
        return [u - u ** 3 / 3 + u * v * v, -(u * u - v * v), v - v ** 3 / 3 + v * u * u];
      })
    );
  }

  // ---------------------------------------------------------------------------
  // The list (interleaved so consecutive shapes differ; with two clouds, the second starts half-way through)
  // ---------------------------------------------------------------------------

  const SHAPES = [
    { name: "swiss roll", make: swissRoll },
    zernikeShape(3, 1, "zernike coma"),
    { name: "mobius strip", tilt: 0.3, make: mobiusStrip },
    { name: "rosenbrock", tilt: 0.5, make: (n) => square(n, rosenbrock, 0.8) },
    { name: "neural network", make: neuralNetwork },
    zernikeShape(2, 2, "zernike astigmatism"),
    { name: "klein bottle", make: kleinBottle },
    { name: "himmelblau", tilt: 0.5, make: (n) => square(n, himmelblau, 0.8) },
    { name: "airy pattern", tilt: 0.1, make: airyPattern },
    zernikeShape(3, 3, "zernike trefoil"),
    { name: "trefoil knot", tilt: 0.2, make: trefoilKnot },
    { name: "rastrigin", tilt: 0.55, make: (n) => square(n, rastrigin, 0.7) },
    { name: "gaussian mixture", make: gaussianMixture },
    zernikeShape(4, 0, "zernike spherical"),
    { name: "enneper surface", tilt: 0.3, make: enneper },
    { name: "monkey saddle", tilt: 0.45, make: (n) => square(n, monkeySaddle, 0.9) },
    zernikeShape(4, 4, "zernike tetrafoil"),
    zernikeShape(4, 2, "zernike secondary astigmatism"),
  ];

  return SHAPES;
})();
