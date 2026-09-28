/* Colour ramps for the data overlays.
 *
 * Both are single-hue sequential ramps (dark -> light on the dark globe), per
 * the data-viz rule "sequential = one hue". Low values fade to transparent so
 * the NASA imagery underneath stays readable.
 *   score      orange, coloured by rank percentile of land cells
 *   predictor  teal, coloured by value across Earth's envelope
 */

const SCORE_STOPS = [
  // t, r, g, b, alpha
  [0.0, 74, 26, 8, 0.0],
  [0.25, 128, 44, 14, 0.38],
  [0.5, 196, 73, 26, 0.58],
  [0.75, 240, 138, 75, 0.78],
  [0.92, 255, 196, 150, 0.9],
  [1.0, 255, 235, 214, 0.96],
];

const PREDICTOR_STOPS = [
  [0.0, 8, 44, 48, 0.55],
  [0.35, 18, 104, 98, 0.72],
  [0.7, 38, 176, 150, 0.84],
  [1.0, 206, 246, 228, 0.92],
];

function sample(stops, t) {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < stops.length; i++) {
    if (x <= stops[i][0]) {
      const a = stops[i - 1];
      const b = stops[i];
      const f = (x - a[0]) / (b[0] - a[0] || 1);
      return [0, 1, 2, 3].map((k) => a[k + 1] + (b[k + 1] - a[k + 1]) * f);
    }
  }
  return stops[stops.length - 1].slice(1);
}

export function cssGradient(kind) {
  const stops = kind === "score" ? SCORE_STOPS : PREDICTOR_STOPS;
  const parts = stops.map(([t, r, g, b, a]) => `rgba(${r},${g},${b},${Math.max(a, 0.15)}) ${t * 100}%`);
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

/* Percentile thresholds of the finite values, for rank colouring and legends. */
export function sortedFinite(values) {
  const out = [];
  for (const v of values) if (Number.isFinite(v)) out.push(v);
  return Float64Array.from(out).sort();
}

export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

function rankOf(sorted, v) {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo / sorted.length;
}

/* Paint a 720x360 grid into RGBA pixels.
 * mode "score": only the top (1 - floor) of land cells are coloured, by rank.
 * mode "predictor": linear between lo and hi. */
export function paint(values, width, height, mode, opts = {}) {
  const image = new ImageData(width, height);
  const px = image.data;
  if (mode === "score") {
    const sorted = opts.sorted || sortedFinite(values);
    const floor = opts.floor ?? 0.5;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      const p = rankOf(sorted, v);
      if (p < floor) continue;
      const [r, g, b, a] = sample(SCORE_STOPS, (p - floor) / (1 - floor));
      const o = i * 4;
      px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = Math.round(a * 255);
    }
  } else {
    const { lo, hi } = opts;
    const span = hi - lo || 1;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (!Number.isFinite(v)) continue;
      const [r, g, b, a] = sample(PREDICTOR_STOPS, (v - lo) / span);
      const o = i * 4;
      px[o] = r; px[o + 1] = g; px[o + 2] = b; px[o + 3] = Math.round(a * 255);
    }
  }
  return image;
}

export function percentileOf(sorted, v) {
  return Number.isFinite(v) ? rankOf(sorted, v) * 100 : NaN;
}
