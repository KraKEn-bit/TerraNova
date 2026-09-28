/* Earth Analogue Finder - interface controller.
 * Every number shown comes from the API (src/compute); this file only renders. */

import { Globe, Twin } from "./globe.js";
import { GodsEye, SUN_PRESETS } from "./godseye.js";
import { FlatMap } from "./flatmap.js";
import { cssGradient, paint, percentileOf, quantile, sortedFinite } from "./colors.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtPct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const fmtInt = (n) => Number(n).toLocaleString("en-US");

const state = {
  health: null,
  targets: [],
  criteria: [],
  sources: null,
  analogs: null,
  targetId: null,
  weights: {},
  defaults: {},
  data: null,          // last /api/score payload
  field: null,         // Float32Array score field
  sorted: null,        // sorted finite scores
  layer: "score",
  layerField: null,    // {values, sorted, lo, hi, unit, label}
  view: "globe",
  selected: null,      // selected result row
  pick: null,          // clicked location
  request: 0,
  topK: 20,
  mode: "target",      // "target" or "custom"
  custom: null,        // {criteria: {key: number|null}, body: "moon"|"mars"|""}
  pins: [],            // up to 3 result rows pinned for comparison
};
const MAX_PINS = 3;

const CUSTOM_ID = "__custom__";

/* ----------------------------------------------------------------- helpers */

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

function decode(field) {
  const raw = atob(field.data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, 6000);
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function spec(key) { return state.criteria.find((c) => c.key === key); }
function target() {
  if (state.mode === "custom") return customTarget();
  return state.targets.find((t) => t.id === state.targetId);
}

function customTarget() {
  const body = state.custom?.body || "";
  const criteria = {};
  for (const c of state.criteria) {
    const v = state.custom?.criteria[c.key];
    const eff = state.data?.effective_profile?.[c.key];
    criteria[c.key] = { value: v ?? null, effective_value: v == null ? null : (eff ?? v), dataset_id: "user_entered" };
  }
  return {
    id: CUSTOM_ID, name: "Custom target", short_name: "your custom target",
    body: body ? body.charAt(0).toUpperCase() + body.slice(1) : "Custom",
    summary: "A target profile you entered by hand.", criteria, source_url: "", trek_url: null,
  };
}

/* The request body shared by /api/score and /api/explain. */
function scoreRequest(extra = {}) {
  const base = state.mode === "custom"
    ? { criteria: state.custom.criteria, body: state.custom.body || null }
    : { target_id: state.targetId };
  return { ...base, weights: state.weights, ...extra };
}

function fmtValue(value, unit) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "no data";
  const abs = Math.abs(value);
  const text = abs >= 100 ? fmtInt(Math.round(value)) : abs >= 10 ? value.toFixed(1) : value.toFixed(2);
  if (unit === "NDVI") return `NDVI ${text}`;
  if (unit === "degrees") return `${text}°`;
  return `${text} ${unit}`;
}

/* "top 3.2%" wording for a percentile; never claims 0% or 100%. */
function topShare(percentile) {
  const top = 100 - percentile;
  if (top < 0.1) return `in the <b>top 0.1%</b>`;
  if (top < 10) return `in the <b class="num">top ${top.toFixed(1)}%</b>`;
  return `in the <b class="num">top ${top.toFixed(0)}%</b>`;
}

function fmtCoord(lat, lon) {
  const ns = lat >= 0 ? "N" : "S";
  const ew = lon >= 0 ? "E" : "W";
  return `${Math.abs(lat).toFixed(2)}° ${ns}, ${Math.abs(lon).toFixed(2)}° ${ew}`;
}

const NOVELTY = {
  new: ["new", "New candidate"],
  near_known: ["near", "Near a known analog"],
  known: ["known", "Known analog"],
};
function noveltyChip(n) {
  const [cls, text] = NOVELTY[n.status] || ["", n.status];
  const title = `Nearest catalogued analog: ${n.nearest_known} (${fmtInt(Math.round(n.distance_km))} km)`;
  return `<span class="novelty ${cls}" title="${esc(title)}">${text}</span>`;
}

/* --------------------------------------------------------------- the views */

const labels = $("labels");
const globe = new Globe($("globe"), { onPick: pickLocation, onHover: hover });
const flat = new FlatMap($("flatmap"), { onPick: pickLocation, onHover: hover });
let twin = null;
try { twin = new Twin($("twinCanvas")); } catch { $("twin").hidden = true; }

function activeView() { return state.view === "globe" ? globe : flat; }

function setView(view) {
  hidePeek();
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
  $("globe").hidden = view !== "globe";
  $("flatmap").hidden = view !== "map";
  globe.visible = view === "globe";
  if (view === "map") {
    flat.draw();
    const focus = state.selected || state.pick;
    if (focus) flat.flyTo(focus.lat, focus.lon, 3);
  }
  renderMarkers();
}

/* ------------------------------------------------------------ hover + peek */

const stageEl = $("globe").parentElement;
const thumbCache = new Map();   // cell key -> {src, credit}
let dwell = null;
let peekToken = 0;

function cellOf(lat, lon) {
  const row = Math.min(359, Math.max(0, Math.floor((90 - lat) / 0.5)));
  const col = Math.min(719, Math.max(0, Math.floor((lon + 180) / 0.5)));
  return { row, col, lat: 90 - (row + 0.5) * 0.5, lon: -179.75 + col * 0.5, key: `${row}_${col}` };
}

/* Offline fallback: crop the local Blue Marble texture around the cell. */
function localCrop(lat, lon) {
  const img = flat.base;
  if (!img) return null;
  const c = document.createElement("canvas");
  c.width = c.height = 204;
  const pxDeg = img.naturalWidth / 360;
  const sx = (lon - 1 + 180) * pxDeg;
  const sy = (90 - lat - 1) * pxDeg;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, sx, sy, 2 * pxDeg, 2 * pxDeg, 0, 0, 204, 204);
  return c.toDataURL("image/jpeg", 0.85);
}

async function thumbFor(cell) {
  if (thumbCache.has(cell.key)) return thumbCache.get(cell.key);
  let out;
  try {
    const res = await fetch(`/api/thumb?lat=${cell.lat}&lon=${cell.lon}`);
    if (!res.ok) throw new Error(String(res.status));
    out = { src: URL.createObjectURL(await res.blob()), credit: "NASA Blue Marble NG · GIBS · 2°×2°" };
  } catch {
    const src = localCrop(cell.lat, cell.lon);
    out = src ? { src, credit: "Offline preview · Blue Marble 10 km" } : null;
  }
  if (out) thumbCache.set(cell.key, out);
  return out;
}

function placeTip(tip, x, y) {
  const left = x + 16 + tip.offsetWidth > stageEl.clientWidth - 8 ? x - tip.offsetWidth - 16 : x + 16;
  const top = Math.min(Math.max(8, y - 20), stageEl.clientHeight - tip.offsetHeight - 8);
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${top}px`;
}

/* The preview card: satellite image of the cell plus a few facts. */
async function showPeek(lat, lon, x, y, html) {
  const tip = $("tooltip");
  const token = ++peekToken;
  const cell = cellOf(lat, lon);
  tip.classList.add("peek");
  tip.innerHTML = `<div class="peek-img loading"><i class="cellbox"></i></div>${html}`;
  tip.hidden = false;
  placeTip(tip, x, y);
  const thumb = await thumbFor(cell);
  if (token !== peekToken || tip.hidden) return;
  const box = tip.querySelector(".peek-img");
  box.classList.remove("loading");
  if (thumb) {
    const img = new Image();
    img.alt = `Satellite view around ${fmtCoord(cell.lat, cell.lon)}`;
    img.src = thumb.src;
    box.prepend(img);
    box.insertAdjacentHTML("beforeend", `<span class="src">${esc(thumb.credit)}</span>`);
  }
}

function hidePeek() {
  peekToken++;
  clearTimeout(dwell);
  const tip = $("tooltip");
  tip.hidden = true;
  tip.classList.remove("peek");
}

function cellFacts(lat, lon) {
  const values = state.layer === "score" ? state.field : state.layerField?.values;
  if (!values) return null;
  const cell = cellOf(lat, lon);
  const v = values[cell.row * 720 + cell.col];
  if (!Number.isFinite(v)) return `<div class="t-sub">${fmtCoord(lat, lon)}</div><div class="t-sub">Not scored (ocean, data gap or &lt;50% land)</div>`;
  if (state.layer === "score") {
    const p = percentileOf(state.sorted, v);
    return `<div class="t-sub">${fmtCoord(lat, lon)}</div><div class="t-title">${fmtPct(v)} match</div>
      <div class="t-sub">better than ${p.toFixed(0)}% of land · click for details</div>`;
  }
  const f = state.layerField;
  return `<div class="t-sub">${fmtCoord(lat, lon)}</div><div class="t-title">${esc(fmtValue(v, f.unit))}</div><div class="t-sub">${esc(f.label)}</div>`;
}

function hover(hit, x, y) {
  clearTimeout(dwell);
  const tip = $("tooltip");
  const facts = hit && cellFacts(hit.lat, hit.lon);
  if (!facts) { hidePeek(); return; }
  const cell = cellOf(hit.lat, hit.lon);
  // Already showing this cell: just follow the cursor.
  if (!tip.hidden && tip.dataset.cell === cell.key) { placeTip(tip, x, y); return; }
  peekToken++;
  tip.classList.remove("peek");
  tip.dataset.cell = cell.key;
  tip.innerHTML = facts;
  tip.hidden = false;
  placeTip(tip, x, y);
  // Rest on a scored cell for a moment and the satellite preview appears.
  const values = state.layer === "score" ? state.field : state.layerField?.values;
  if (Number.isFinite(values[cell.row * 720 + cell.col])) {
    dwell = setTimeout(() => showPeek(hit.lat, hit.lon, x, y, facts), 450);
  }
}

function markerPeek(el, lat, lon, html) {
  el.addEventListener("mouseenter", () => {
    const r = el.getBoundingClientRect();
    const s = stageEl.getBoundingClientRect();
    $("tooltip").dataset.cell = "";
    showPeek(lat, lon, r.left + r.width / 2 - s.left, r.top + r.height / 2 - s.top, html);
  });
  el.addEventListener("mouseleave", hidePeek);
  el.addEventListener("focus", () => el.dispatchEvent(new Event("mouseenter")));
  el.addEventListener("blur", hidePeek);
}

function resultPeekHtml(r) {
  return `<div class="t-row"><span class="t-title">#${r.rank} ${esc(r.label.text)}</span></div>
    <div class="t-row"><span class="t-sub">${fmtCoord(r.lat, r.lon)}</span>${noveltyChip(r.novelty)}</div>
    <div class="t-row"><span class="t-title num">${fmtPct(r.score)} match</span><span class="t-sub">click to open</span></div>`;
}

const markerEls = new Map();   // result index -> marker element

function renderMarkers() {
  labels.innerHTML = "";
  markerEls.clear();
  const markers = [];
  if ($("showKnown").checked && state.analogs) {
    const body = target()?.body.toLowerCase();
    for (const site of state.analogs.sites) {
      const el = document.createElement("div");
      el.className = "marker known";
      el.setAttribute("aria-label", `Known analog: ${site.name}`);
      el.tabIndex = 0;
      if (body && !site.bodies.includes(body)) el.style.filter = "brightness(0.6)";
      markerPeek(el, site.lat, site.lon, `<div class="t-title">${esc(site.name)}</div>
        <div class="t-sub">Known ${esc(site.bodies.join(" & "))} analog · ${esc(site.kind)}</div>
        <div class="t-sub">${esc(site.use)}</div>`);
      labels.appendChild(el);
      markers.push({ lat: site.lat, lon: site.lon, el });
    }
  }
  if (state.data) {
    for (const r of state.data.results) {
      const el = document.createElement("button");
      el.type = "button";
      const active = state.selected && state.selected.index === r.index;
      const pinned = state.pins.some((x) => x.index === r.index);
      el.className = "marker" + (active ? " active" : "") + (r.rank > 10 && !active ? " minor" : "") + (pinned ? " pinned" : "");
      el.textContent = r.rank;
      el.setAttribute("aria-label", `#${r.rank} ${r.label.text}, ${fmtPct(r.score)} match`);
      el.addEventListener("click", () => { hidePeek(); selectResult(r); });
      markerPeek(el, r.lat, r.lon, resultPeekHtml(r));
      markerEls.set(r.index, el);
      labels.appendChild(el);
      markers.push({ lat: r.lat, lon: r.lon, el });
    }
  }
  if (state.pick) {
    const el = document.createElement("div");
    el.className = "marker pick";
    labels.appendChild(el);
    markers.push({ lat: state.pick.lat, lon: state.pick.lon, el });
  }
  globe.setMarkers(state.view === "globe" ? markers : []);
  flat.setMarkers(state.view === "map" ? markers : []);
}

/* ------------------------------------------------------------------ layers */

function paintLayer() {
  let image;
  if (state.layer === "score") {
    if (!state.field) return;
    image = paint(state.field, 720, 360, "score", { sorted: state.sorted, floor: 0.5 });
    const q = (p) => quantile(state.sorted, p);
    // The ramp runs linearly from the 50th to the 100th percentile of land cells,
    // so "top 10%" sits at 80% of its width and "top 1%" at 98%.
    $("legend").innerHTML = `
      <div class="title">Analog score · ${esc(target()?.short_name || "custom profile")}</div>
      <div class="ramp" style="background:${cssGradient("score")}"></div>
      <div class="scale abs num"><span style="left:0">top 50%</span><span style="left:80%">top 10%</span><span style="left:98%">1%</span></div>
      ${histogram()}
      <div class="note">Brighter = closer match; uncoloured land is in the bottom half. Bars: how many land cells
        reach each score (orange = top 10%, at least ${fmtPct(q(0.9))}).</div>
      <div class="credit-line">Earth: <a href="${esc(state.sources.basemaps.earth.url)}" target="_blank" rel="noopener">${esc(state.sources.basemaps.earth.credit)}</a></div>`;
  } else {
    const f = state.layerField;
    if (!f) return;
    image = paint(f.values, 720, 360, "predictor", { lo: f.lo, hi: f.hi });
    $("legend").innerHTML = `
      <div class="title">${esc(f.label)}</div>
      <div class="ramp" style="background:${cssGradient("predictor")}"></div>
      <div class="scale num"><span>${esc(fmtValue(f.lo, f.unit))}</span><span>${esc(fmtValue(f.hi, f.unit))}</span></div>
      <div class="note">Raw Earth data behind the score. Source:
        <a href="${esc(f.url)}" target="_blank" rel="noopener">${esc(f.dataset)}</a></div>`;
  }
  globe.setOverlay(image);
  flat.setOverlay(image);
}

/* Score distribution of land cells, 24 bins; top-10% bins in the accent colour. */
function histogram() {
  const all = state.sorted;
  if (!all?.length) return "";
  // Cells vetoed to 0% by one fully mismatched criterion are counted, not drawn,
  // so they do not flatten every other bar.
  const firstNonZero = all.findIndex((v) => v > 0);
  const s = firstNonZero < 0 ? all : all.subarray(firstNonZero);
  const zero = firstNonZero < 0 ? 0 : firstNonZero;
  const lo = s[0];
  const hi = s[s.length - 1];
  const bins = new Array(24).fill(0);
  for (const v of s) bins[Math.min(23, Math.floor(((v - lo) / (hi - lo || 1)) * 24))]++;
  const top = quantile(s, 0.9);
  const max = Math.max(...bins);
  const bars = bins.map((n, i) => {
    const edge = lo + ((i + 1) / 24) * (hi - lo);
    return `<i class="${edge > top ? "hot" : ""}" style="height:${Math.max(4, (n / max) * 100).toFixed(0)}%"></i>`;
  }).join("");
  return `<div class="hist" role="img" aria-label="Distribution of scores across ${fmtInt(s.length)} land cells, from ${fmtPct(lo)} to ${fmtPct(hi)}">${bars}</div>
    <div class="scale num"><span>${fmtPct(lo)}</span><span>${zero ? `+ ${fmtInt(zero)} cells at 0% (vetoed)` : ""}</span><span>${fmtPct(hi)}</span></div>`;
}

async function setLayer(key) {
  state.layer = key;
  if (key !== "score") {
    const s = spec(key);
    const payload = await api(`/api/predictor?key=${encodeURIComponent(key)}`);
    state.layerField = {
      values: decode(payload), lo: s.earth_min, hi: s.earth_max,
      unit: s.unit, label: s.label, dataset: s.dataset_id, url: s.source_url,
    };
  }
  paintLayer();
}

/* ----------------------------------------------------------------- targets */

function renderTargets() {
  const groups = [];
  for (const t of state.targets) {
    let g = groups.find((x) => x.name === t.group);
    if (!g) groups.push((g = { name: t.group, items: [] }));
    g.items.push(t);
  }
  const card = (t) => `
    <button class="target-card" role="radio" aria-checked="${state.mode === "target" && t.id === state.targetId}" data-id="${esc(t.id)}" type="button">
      <span class="planet" style="background-image:url(assets/${esc(t.body.toLowerCase())}.jpg)"></span>
      <strong>${esc(t.short_name)}</strong>
      <small>${Math.abs(t.latitude).toFixed(1)}°${t.latitude >= 0 ? "N" : "S"} ${Math.abs(t.longitude).toFixed(1)}°${t.longitude >= 0 ? "E" : "W"}</small>
    </button>`;
  $("targets").innerHTML = groups.map((g) => `
    <p class="group-label">${esc(g.name)}</p>
    <div class="target-grid">${g.items.map(card).join("")}</div>`).join("") + `
    <p class="group-label">Your own</p>
    <div class="target-grid">
      <button class="target-card custom" role="radio" aria-checked="${state.mode === "custom"}" data-id="${CUSTOM_ID}" type="button">
        <span class="planet planet-custom" aria-hidden="true">✎</span>
        <strong>Custom target</strong><small>enter values by hand</small>
      </button>
    </div>`;
  $("targets").querySelectorAll(".target-card").forEach((b) => b.addEventListener("click", () => chooseTarget(b.dataset.id)));
}

function renderTargetDetail() {
  if (state.mode === "custom") { renderCustomForm(); return; }
  const t = target();
  const rows = state.criteria.map((c) => {
    const m = t.criteria[c.key];
    if (m.value === null || m.value === undefined) {
      return `<tr class="unused"><th scope="row">${esc(c.label)}</th><td><span class="tag" title="${esc(m.definition_note || "")}">not used for this site</span></td></tr>`;
    }
    const off = (state.weights[c.key] ?? 0) === 0;
    let used = fmtValue(m.effective_value, c.unit);
    let note = "";
    if ("earth_percentile" in m) {
      const measured = m.measured_value === null || m.measured_value === undefined
        ? "qualitative" : `measured ${esc(fmtValue(m.measured_value, c.unit))} at ${esc(m.baseline || "a finer scale")}`;
      note = `<span class="note">Earth's ${m.earth_percentile}th percentile (${measured})</span>`;
    } else if (Math.abs(m.effective_value - m.value) > 1e-9) {
      note = `<span class="note">real value ${esc(fmtValue(m.value, c.unit))} lies outside the range of 99% of Earth's land; matched to that range's edge</span>`;
    }
    if (off) { used = `<span class="tag">not scored</span>`; note = ""; }
    return `<tr><th scope="row">${esc(c.label)}</th><td>${used}${note}</td></tr>`;
  }).join("");
  $("targetDetail").innerHTML = `
    <p>${esc(t.summary)}</p>
    <div class="target-links">
      <a href="${esc(t.source_url)}" target="_blank" rel="noopener">Source: ${esc(t.source_label || "reference")}</a>
      ${t.trek_url ? `<a href="${esc(t.trek_url)}" target="_blank" rel="noopener">Open NASA ${esc(t.body)} Trek ↗</a>` : ""}
    </div>
    <p class="how">How matching works: every land cell on Earth is compared with <b>this site's
      signature</b>, the numbers below, measured on the ${esc(t.body)} by the cited missions. It is not a
      picture-to-picture comparison. Each criterion is scored 0–100% and combined with your weights.</p>
    ${t.location_note ? `<p class="hint">${esc(t.location_note)}</p>` : ""}
    <table class="profile"><caption>What Earth is matched against</caption>${rows}</table>`;
  const caption = $("twinCaption");
  caption.innerHTML = `<strong>${esc(t.short_name)}</strong>${esc(t.body)} target · ${fmtCoord(t.latitude, t.longitude)}
    <span class="twin-hint">spinning · drag to rotate · double-click to reset</span>`;
  $("twin").hidden = !twin;
  if (twin) twin.show(t.body, t.latitude, t.longitude).catch(() => {});
}

/* ------------------------------------------------------------ custom target */

function startCustom() {
  // Start from whatever target was showing, so the numbers are sensible.
  const base = state.targets.find((t) => t.id === state.targetId) || state.targets[0];
  const criteria = {};
  for (const c of state.criteria) {
    const m = base.criteria[c.key];
    criteria[c.key] = m.value === null || m.value === undefined ? null : Number(m.effective_value);
  }
  state.custom = { criteria, body: base.body.toLowerCase() };
}

function renderCustomForm() {
  const rows = state.criteria.map((c) => {
    const v = state.custom.criteria[c.key];
    const on = v !== null && v !== undefined;
    return `<div class="manual-row">
      <label class="manual-use"><input type="checkbox" data-use="${esc(c.key)}" ${on ? "checked" : ""}>
        <span>${esc(c.label)}</span></label>
      <span class="manual-input"><input type="number" step="any" inputmode="decimal" data-val="${esc(c.key)}"
        value="${on ? v : ""}" ${on ? "" : "disabled"} aria-label="${esc(c.label)} target value">
        <span class="unit">${esc(c.unit === "1" ? "" : c.unit)}</span></span>
      <span class="range-hint">Earth: ${esc(fmtValue(c.earth_min, c.unit))} to ${esc(fmtValue(c.earth_max, c.unit))}</span>
    </div>`;
  }).join("");
  const body = state.custom.body;
  $("targetDetail").innerHTML = `
    <p>Type the conditions you want to find on Earth. Untick a criterion to leave it out.
      Values outside Earth's range are matched to Earth's nearest extreme, like the real targets.</p>
    <form class="manual" id="manualForm">
      ${rows}
      <label class="manual-body">Validate against known analogs for
        <select id="manualBody">
          <option value="moon" ${body === "moon" ? "selected" : ""}>Moon</option>
          <option value="mars" ${body === "mars" ? "selected" : ""}>Mars</option>
          <option value="" ${!body ? "selected" : ""}>no validation</option>
        </select></label>
      <div class="manual-actions">
        <button class="primary" type="submit">Score my target</button>
        <button class="ghost small" type="button" id="manualReset">Copy from a real target</button>
      </div>
      <p class="hint" id="manualError" role="alert"></p>
    </form>`;
  $("twin").hidden = true;
  const form = $("manualForm");
  form.querySelectorAll("[data-use]").forEach((box) => box.addEventListener("change", () => {
    const input = form.querySelector(`[data-val="${box.dataset.use}"]`);
    input.disabled = !box.checked;
    if (box.checked && input.value === "") input.focus();
  }));
  $("manualReset").addEventListener("click", () => { startCustom(); renderCustomForm(); });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const criteria = {};
    for (const c of state.criteria) {
      const on = form.querySelector(`[data-use="${c.key}"]`).checked;
      const raw = form.querySelector(`[data-val="${c.key}"]`).value.trim();
      if (!on) { criteria[c.key] = null; continue; }
      const v = Number(raw);
      if (raw === "" || !Number.isFinite(v)) {
        $("manualError").textContent = `Enter a number for ${c.label}, or untick it.`;
        return;
      }
      criteria[c.key] = v;
    }
    if (Object.values(criteria).every((v) => v === null)) {
      $("manualError").textContent = "Tick at least one criterion.";
      return;
    }
    $("manualError").textContent = "";
    state.custom = { criteria, body: $("manualBody").value };
    for (const c of state.criteria) {
      if (criteria[c.key] !== null && state.weights[c.key] === 0 && c.key !== "elevation") state.weights[c.key] = 1;
    }
    renderWeights();
    showList();
    runScore().then(() => {
      const top = state.data?.results[0];
      if (top && state.view === "globe") globe.flyTo(top.lat, top.lon);
    });
  });
}

function chooseTarget(id) {
  if (id === CUSTOM_ID) {
    if (state.mode === "custom") return;
    state.mode = "custom";
    startCustom();
    state.selected = null;
    state.pick = null;
    renderTargets();
    renderTargetDetail();
    renderWeights();
    return;   // scored when the form is submitted
  }
  if (id === state.targetId && state.mode === "target") return;
  if (state.pins.length) { state.pins = []; renderCompare(); }
  state.mode = "target";
  state.targetId = id;
  state.selected = null;
  state.pick = null;
  state.weights = { ...target().default_weights };
  renderTargets();
  renderTargetDetail();
  renderWeights();
  showList();
  runScore().then(() => {
    const top = state.data?.results[0];
    if (top && state.view === "globe") globe.flyTo(top.lat, top.lon);
  });
}

/* ----------------------------------------------------------------- weights */

function renderWeights() {
  const t = target();
  $("weights").innerHTML = state.criteria.map((c) => {
    const unused = t && (t.criteria[c.key]?.value === null || t.criteria[c.key]?.value === undefined);
    const w = unused ? 0 : state.weights[c.key];
    let sub = "";
    if (unused) sub = `<div class="sub">Not used for ${esc(t.short_name)}.</div>`;
    else if (c.key === "elevation") sub = `<div class="sub">Off by default: Moon and Mars heights use their own datums.</div>`;
    else if (c.key === "mean_annual_temperature" && w === 0) sub = `<div class="sub">Used by cold-trap targets.</div>`;
    return `<div class="weight ${w === 0 ? "off" : ""}" data-key="${esc(c.key)}">
      <label for="w_${esc(c.key)}" title="${esc(c.description)}"><span>${esc(c.label)}</span>
        <span class="val num" id="wv_${esc(c.key)}">${unused ? "n/a" : w === 0 ? "off" : `${w.toFixed(1)}×`}</span></label>
      <input type="range" id="w_${esc(c.key)}" min="0" max="3" step="0.1" value="${w}" ${unused ? "disabled" : ""}
        aria-describedby="wv_${esc(c.key)}">${sub}</div>`;
  }).join("");
  $("weights").querySelectorAll("input[type=range]").forEach((input) => {
    input.addEventListener("input", () => {
      const key = input.id.slice(2);
      const w = parseFloat(input.value);
      state.weights[key] = w;
      $(`wv_${key}`).textContent = w === 0 ? "off" : `${w.toFixed(1)}×`;
      input.closest(".weight").classList.toggle("off", w === 0);
      if (Object.values(state.weights).every((v) => v === 0)) return;
      scoreSoon();
    });
  });
}

/* ----------------------------------------------------------------- scoring */

async function runScore() {
  const ticket = ++state.request;
  $("resultsSummary").textContent = "Scoring…";
  try {
    const data = await api("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scoreRequest({ top_k: state.topK, min_separation_cells: 3, include_field: true })),
    });
    if (ticket !== state.request) return;
    state.data = data;
    state.field = decode(data.field);
    state.sorted = sortedFinite(state.field);
    if (state.layer === "score") paintLayer();
    renderResults();
    renderValidation();
    renderMarkers();
    if (state.selected) {
      const again = data.results.find((r) => r.index === state.selected.index);
      if (again) renderDetail(again); else showList();
    }
    if (state.pick) refreshPick();
    refreshPins();
  } catch (err) {
    if (ticket === state.request) toast(`Scoring failed: ${err.message}`);
  }
}
const scoreSoon = debounce(runScore, 220);
const topKSoon = debounce(() => { if (!state.selected) showList(); runScore(); }, 250);

function renderResults() {
  const d = state.data;
  $("resultsSummary").innerHTML = `Top ${d.results.length} of ${fmtInt(state.sorted.length)} land cells, at least 1.5° apart.`;
  $("siteList").innerHTML = d.results.map((r) => `
    <li><button class="site ${state.selected?.index === r.index ? "active" : ""}" data-index="${r.index}" type="button">
      <span class="rank num">${r.rank}</span>
      <span class="name">${esc(r.label.text)}</span>
      <span class="score"><b>${fmtPct(r.score)}</b><small>match</small></span>
      <span class="meta"><span class="num">${fmtCoord(r.lat, r.lon)}</span>${noveltyChip(r.novelty)}${state.pins.some((x) => x.index === r.index) ? `<span class="pin-star" title="Pinned">● pinned</span>` : ""}</span>
      <span class="bar" aria-hidden="true"><i style="width:${(r.score * 100).toFixed(1)}%"></i></span>
    </button></li>`).join("");
  $("siteList").querySelectorAll(".site").forEach((b) => {
    const r = d.results.find((x) => String(x.index) === b.dataset.index);
    b.addEventListener("click", () => { hidePeek(); selectResult(r); });
    b.addEventListener("mouseenter", () => {
      const m = markerEls.get(r.index);
      if (m && m.style.display !== "none") m.dispatchEvent(new Event("mouseenter"));
      m?.classList.add("active");
    });
    b.addEventListener("mouseleave", () => {
      hidePeek();
      if (state.selected?.index !== r.index) markerEls.get(r.index)?.classList.remove("active");
    });
  });
}

function selectResult(r) {
  state.selected = r;
  state.pick = null;
  selectTab("results");
  renderDetail(r);
  renderMarkers();
  if (state.view === "globe") globe.flyTo(r.lat, r.lon, (globe.fitDistance || 3.6) * 0.78);
  else flat.flyTo(r.lat, r.lon, 3);
}

async function pickLocation(lat, lon) {
  state.pick = { lat, lon };
  state.selected = null;
  renderMarkers();
  selectTab("results");
  await refreshPick();
}

async function refreshPick() {
  const { lat, lon } = state.pick;
  try {
    const r = await api(`/api/explain?lat=${lat.toFixed(4)}&lon=${lon.toFixed(4)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(scoreRequest()),
    });
    if (!r.scored) {
      $("siteList").hidden = true;
      const det = $("detail");
      det.hidden = false;
      det.innerHTML = `<button class="ghost small back" type="button">← All ranked sites</button>
        <h3>${esc(r.label.text)}</h3>
        <div class="coords num">${fmtCoord(r.lat, r.lon)}</div>
        <p class="notice">${esc(r.reason)}</p>`;
      det.querySelector(".back").addEventListener("click", showList);
      return;
    }
    renderDetail(r);
  } catch (err) {
    toast(`Could not inspect that location: ${err.message}`);
  }
}

function showList() {
  state.selected = null;
  state.pick = null;
  $("detail").hidden = true;
  $("siteList").hidden = false;
  $("siteList").querySelectorAll(".site").forEach((b) => b.classList.remove("active"));
  renderMarkers();
}

function renderDetail(r) {
  const t = target();
  const d = state.data;
  const eff = d.effective_profile;
  const crits = state.criteria.map((c) => {
    const w = d.weights[c.key];
    const sim = r.similarities[c.key];
    const val = r.values[c.key];
    if (w === 0 || sim === null) {
      return `<div class="crit off"><div class="row"><span>${esc(c.label)}</span><span class="tag">not scored</span></div>
        <div class="vs">Here: ${esc(fmtValue(val, c.unit))}</div></div>`;
    }
    return `<div class="crit"><div class="row"><span>${esc(c.label)}</span><span class="pct">${fmtPct(sim)}</span></div>
      <div class="vs">Here ${esc(fmtValue(val, c.unit))} · target ${esc(fmtValue(eff[c.key], c.unit))}</div>
      <div class="bar" role="img" aria-label="${esc(c.label)} similarity ${fmtPct(sim)}"><i style="width:${(sim * 100).toFixed(1)}%"></i></div></div>`;
  }).join("");

  const nov = r.novelty;
  const novText = nov.status === "new"
    ? `No catalogued analog within 500 km (nearest: ${esc(nov.nearest_known)}, ${fmtInt(Math.round(nov.distance_km))} km).`
    : `${fmtInt(Math.round(nov.distance_km))} km from <a href="${esc(nov.source_url)}" target="_blank" rel="noopener">${esc(nov.nearest_known)}</a>.`;
  const lstNote = r.lst_source === "NASA POWER TS_RANGE fit"
    ? `<p class="notice">No MODIS surface-temperature data here: the day-night swing is estimated from NASA POWER skin temperature (see How it works).</p>` : "";
  const claims = r.rationale.claims.map((c) => `<li>${esc(c.text)} <a href="${esc(c.source_url)}" target="_blank" rel="noopener">source</a></li>`).join("");
  const caveats = r.rationale.caveats.map((c) => `<li>${esc(c.text)}</li>`).join("");
  const worldview = `https://worldview.earthdata.nasa.gov/?v=${(r.lon - 4).toFixed(2)},${(r.lat - 3).toFixed(2)},${(r.lon + 4).toFixed(2)},${(r.lat + 3).toFixed(2)}&l=MODIS_Terra_CorrectedReflectance_TrueColor`;
  const gmaps = `https://www.google.com/maps/@${r.lat.toFixed(4)},${r.lon.toFixed(4)},9z/data=!3m1!1e3`;
  const title = r.rank ? `#${r.rank} · ${esc(r.label.text)}` : esc(r.label.text);

  $("siteList").hidden = true;
  const det = $("detail");
  det.hidden = false;
  det.innerHTML = `
    <button class="ghost small back" type="button">← All ranked sites</button>
    <h3>${title}</h3>
    <div class="coords"><span class="num">${fmtCoord(r.lat, r.lon)}</span>${noveltyChip(nov)}
      <button class="link-button" type="button" data-copy="${r.lat.toFixed(3)}, ${r.lon.toFixed(3)}">Copy coordinates</button></div>
    <div class="hero">
      <div class="big num">${(r.score * 100).toFixed(0)}<small>%</small></div>
      <div class="what">match to <b>${esc(t.short_name)}</b><br>
        ${topShare(r.percentile)} of ${fmtInt(state.sorted.length)} land cells</div>
    </div>
    <div class="detail-actions">
      <button class="godseye-button" type="button" id="openEye" ${Math.abs(r.lat) > 84 ? "disabled title=\"Elevation tiles stop at about 84 degrees latitude\"" : ""}>
        <span aria-hidden="true">◉</span> God's Eye 3D view</button>
      <button class="ghost small" type="button" id="pinSite">${state.pins.some((x) => x.index === r.index) ? "Unpin" : "Pin to compare"}</button>
    </div>
    <p class="notice">${novText}</p>
    ${lstNote}
    <h4 class="section-title">Criterion by criterion</h4>
    ${crits}
    <h4 class="section-title">Verify it yourself</h4>
    <div class="verify">
      <a href="${esc(worldview)}" target="_blank" rel="noopener">NASA Worldview ↗<small>MODIS true colour here</small></a>
      <a href="${esc(gmaps)}" target="_blank" rel="noopener">Satellite view ↗<small>Google Maps imagery</small></a>
      ${t.trek_url ? `<a href="${esc(t.trek_url)}" target="_blank" rel="noopener">NASA ${esc(t.body)} Trek ↗<small>the target site</small></a>` : ""}
      ${t.source_url ? `<a href="${esc(t.source_url)}" target="_blank" rel="noopener">Target reference ↗<small>${esc(t.source_label || "")}</small></a>` : ""}
    </div>
    <details class="drawer"><summary>Why, in words (with sources)</summary><div><ul>${claims}</ul></div></details>
    <details class="drawer"><summary>Caveats</summary><div><ul>${caveats}</ul></div></details>
    <details class="drawer"><summary>Raw data behind these numbers</summary><div>
      <pre class="json">${esc(JSON.stringify({ lat: r.lat, lon: r.lon, score: r.score, percentile: r.percentile, values: r.values, similarities: r.similarities, weights: d.weights, target: d.profile, effective_target: eff, lst_source: r.lst_source, novelty: nov }, null, 2))}</pre></div></details>`;
  det.querySelector(".back").addEventListener("click", showList);
  det.querySelector("#openEye").addEventListener("click", () => openGodsEye(r));
  det.querySelector("#pinSite").addEventListener("click", () => togglePin(r));
  det.querySelector("[data-copy]").addEventListener("click", async (e) => {
    try { await navigator.clipboard.writeText(e.target.dataset.copy); e.target.textContent = "Copied"; } catch { e.target.textContent = e.target.dataset.copy; }
  });
  det.closest(".panel").scrollTop = 0;
  if (window.innerWidth <= 1020) det.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* -------------------------------------------------------------- validation */

function renderValidation() {
  const v = state.data.validation;
  const chip = $("chipAuc");
  if (!v || v.auc === null) { chip.hidden = true; $("validation").innerHTML = `<p class="hint">No validation for a custom profile.</p>`; return; }
  chip.hidden = false;
  chip.innerHTML = `<span class="dot on"></span>Validated · AUC ${v.auc.toFixed(2)}`;
  const body = v.tag === "cold_polar" ? "cold polar-desert" : v.body.charAt(0).toUpperCase() + v.body.slice(1);
  const share = v.auc >= 0.999 ? "every" : `${fmtPct(v.auc)} of`;
  const rows = v.controls.map((c) => `
    <tr><td><span class="role ${c.role}">${{ positive: "known analog", negative: "non-analog", geology: "geology only" }[c.role]}</span></td>
      <td>${c.source_url ? `<a href="${esc(c.source_url)}" target="_blank" rel="noopener">${esc(c.name)}</a>` : esc(c.name)}</td>
      <td class="n">${c.score === null ? "–" : fmtPct(c.score)}</td>
      <td class="pbar">${c.percentile === null ? "" : `<div class="bar" role="img" aria-label="percentile ${c.percentile}"><i style="width:${c.percentile}%"></i></div>`}</td>
      <td class="n">${c.percentile === null ? "–" : `p${c.percentile.toFixed(0)}`}</td></tr>`).join("");
  $("validation").innerHTML = `
    <div class="auc-hero"><div class="big num">${v.auc.toFixed(2)}</div>
      <p><b>ROC-AUC.</b> In ${share} pairing, a known ${esc(body)} analog site outscores a vegetated or humid
      reference point (${v.positives} analogs × ${v.negatives} references). 1.00 is perfect separation; 0.50 is chance.</p></div>
    <h4 class="section-title">Control sites under the current weights</h4>
    <table class="controls"><thead><tr><th>Role</th><th>Site</th><th class="n">Score</th><th colspan="2">Percentile of land</th></tr></thead>
      <tbody>${rows}</tbody></table>
    <p class="hint" style="margin-top:10px">${esc(v.method)} "Geology only" sites were chosen for rocks this
      model does not measure and are not counted. Known analogs span very different environments, so no single
      target should rank all of them at the top.</p>`;
}

/* ------------------------------------------------------------------ export */

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportGeojson() {
  const d = state.data;
  if (!d) return;
  const features = d.results.map((r) => ({
    type: "Feature",
    geometry: { type: "Point", coordinates: [r.lon, r.lat] },
    properties: { rank: r.rank, name: r.label.text, score: r.score, percentile: r.percentile, novelty: r.novelty.status, nearest_known: r.novelty.nearest_known, ...Object.fromEntries(Object.entries(r.values).map(([k, v]) => [`value_${k}`, v])), ...Object.fromEntries(Object.entries(r.similarities).map(([k, v]) => [`similarity_${k}`, v])) },
  }));
  const doc = { type: "FeatureCollection", properties: { target: d.target_id, weights: d.weights, generated_by: "Earth Analogue Finder", cell_size_degrees: 0.5 }, features };
  download(`analogs_${d.target_id}.geojson`, JSON.stringify(doc, null, 2), "application/geo+json");
}

function exportCsv() {
  const d = state.data;
  if (!d) return;
  const keys = state.criteria.map((c) => c.key);
  const head = ["rank", "name", "lat", "lon", "score", "percentile", "novelty", "nearest_known", ...keys.map((k) => `value_${k}`), ...keys.map((k) => `similarity_${k}`)];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = d.results.map((r) => [r.rank, q(r.label.text), r.lat, r.lon, r.score, r.percentile, r.novelty.status, q(r.novelty.nearest_known), ...keys.map((k) => r.values[k] ?? ""), ...keys.map((k) => r.similarities[k])].join(","));
  download(`analogs_${d.target_id}.csv`, [head.join(","), ...lines].join("\n"), "text/csv");
}

/* --------------------------------------------------------------- God's Eye */

const eye = new GodsEye($("godseye"));
let eyeReturnFocus = null;
eye.onClose = () => { eyeReturnFocus?.focus?.(); };

function openGodsEye(r, preset = null) {
  if (!r || Math.abs(r.lat) > 84) { toast("The 3D view needs elevation tiles, which stop at about 84 degrees latitude."); return Promise.resolve(); }
  hidePeek();
  eyeReturnFocus = document.activeElement;
  const t = target();
  const context = {
    targetId: state.mode === "target" ? state.targetId : null,
    targetName: t.short_name,
    subtitle: `${fmtCoord(r.lat, r.lon)} · ${fmtPct(r.score)} match to ${t.short_name}`,
    badges: `${r.rank ? `<span class="chip">#${r.rank} of ${state.data.results.length}</span>` : ""}${noveltyChip(r.novelty)}`,
  };
  if (preset) eye.sun = { ...SUN_PRESETS[preset] };
  return eye.show(r, context);
}

/* ---------------------------------------------------------- pin + compare */

function togglePin(r) {
  const i = state.pins.findIndex((x) => x.index === r.index);
  if (i >= 0) state.pins.splice(i, 1);
  else {
    if (state.pins.length >= MAX_PINS) { toast(`You can compare up to ${MAX_PINS} sites. Unpin one first.`); return; }
    state.pins.push(r);
  }
  renderCompare();
  renderMarkers();
  if (state.data) renderResults();
  if (state.selected?.index === r.index || state.pick) {
    const btn = $("pinSite");
    if (btn) btn.textContent = i >= 0 ? "Pin to compare" : "Unpin";
  }
}

async function refreshPins() {
  if (!state.pins.length) return;
  state.pins = await Promise.all(state.pins.map(async (r) => {
    try {
      const fresh = await api(`/api/explain?lat=${r.lat}&lon=${r.lon}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scoreRequest()),
      });
      const ranked = state.data.results.find((x) => x.index === fresh.index);
      return fresh.scored ? { ...fresh, rank: ranked?.rank ?? null } : r;
    } catch { return r; }
  }));
  renderCompare();
}

function renderCompare() {
  const box = $("compare");
  if (!state.pins.length) { box.hidden = true; box.innerHTML = ""; return; }
  const pins = state.pins;
  const active = state.criteria.filter((c) => (state.data?.weights[c.key] ?? 0) > 0);
  const row = (label, values, fmt) => {
    const best = Math.max(...values.map((v) => (v === null || v === undefined ? -Infinity : v)));
    return `<tr><td>${esc(label)}</td>${values.map((v) => `<td class="${v === best && pins.length > 1 ? "best" : ""}">${v === null || v === undefined ? "–" : fmt(v)}</td>`).join("")}</tr>`;
  };
  box.hidden = false;
  box.innerHTML = `
    <div class="compare-head"><h4>Compare pinned sites</h4><button class="link-button" type="button" id="clearPins">Clear</button></div>
    <table>
      <thead><tr><th></th>${pins.map((r) => `<th title="${esc(r.label.text)}">${r.rank ? `#${r.rank} ` : ""}${esc(r.label.text)}
        <button class="unpin" type="button" data-unpin="${r.index}" aria-label="Unpin ${esc(r.label.text)}">✕</button></th>`).join("")}</tr></thead>
      <tbody>
        ${row("Match score", pins.map((r) => r.score), (v) => fmtPct(v))}
        ${active.map((c) => row(c.label, pins.map((r) => r.similarities[c.key]), (v) => fmtPct(v))).join("")}
        <tr><td>Novelty</td>${pins.map((r) => `<td>${esc(NOVELTY[r.novelty.status]?.[1] || r.novelty.status)}</td>`).join("")}</tr>
      </tbody>
    </table>`;
  $("clearPins").addEventListener("click", () => { state.pins = []; renderCompare(); renderMarkers(); renderResults(); });
  box.querySelectorAll("[data-unpin]").forEach((b) => b.addEventListener("click", () => {
    togglePin(state.pins.find((x) => String(x.index) === b.dataset.unpin));
  }));
}

/* ------------------------------------------------------------------ search */

let searchItems = [];
let searchActive = -1;

function flyToPlace(lat, lon) {
  if (state.view === "globe") globe.flyTo(lat, lon, (globe.fitDistance || 3.6) * 0.7);
  else flat.flyTo(lat, lon, 4);
  pickLocation(lat, lon);
}

function renderSearch() {
  const list = $("searchResults");
  const input = $("search");
  if (!searchItems.length) {
    list.innerHTML = `<li class="s-empty" role="option" aria-disabled="true">No match. Try a town, desert, analog site or "lat, lon".</li>`;
  } else {
    list.innerHTML = searchItems.map((it, i) => `<li role="option" id="sr${i}" aria-selected="${i === searchActive}" data-i="${i}">
      <span class="s-name">${esc(it.name)}</span><span class="s-meta">${esc(it.kind)} · ${esc(it.detail || "")} · ${fmtCoord(it.lat, it.lon)}</span></li>`).join("");
  }
  list.hidden = false;
  input.setAttribute("aria-expanded", "true");
  input.setAttribute("aria-activedescendant", searchActive >= 0 ? `sr${searchActive}` : "");
  list.querySelectorAll("[data-i]").forEach((li) => li.addEventListener("mousedown", (e) => {
    e.preventDefault();
    chooseSearch(Number(li.dataset.i));
  }));
}

function closeSearch() {
  $("searchResults").hidden = true;
  $("search").setAttribute("aria-expanded", "false");
  searchActive = -1;
}

function chooseSearch(i) {
  const it = searchItems[i];
  if (!it) return;
  $("search").value = it.name;
  closeSearch();
  $("search").blur();   // give the keyboard back to the shortcuts
  flyToPlace(it.lat, it.lon);
}

const runSearch = debounce(async (q) => {
  if (!q.trim()) { closeSearch(); return; }
  try {
    searchItems = (await api(`/api/search?q=${encodeURIComponent(q)}`)).results;
    searchActive = searchItems.length ? 0 : -1;
    renderSearch();
  } catch { closeSearch(); }
}, 160);

function wireSearch() {
  const input = $("search");
  input.addEventListener("input", () => runSearch(input.value));
  input.addEventListener("blur", () => setTimeout(closeSearch, 120));
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (!searchItems.length) return;
      e.preventDefault();
      searchActive = (searchActive + (e.key === "ArrowDown" ? 1 : -1) + searchItems.length) % searchItems.length;
      renderSearch();
    } else if (e.key === "Enter") {
      e.preventDefault();
      chooseSearch(Math.max(0, searchActive));
    } else if (e.key === "Escape") {
      closeSearch();
      input.blur();
    }
  });
}

/* -------------------------------------------------------------------- tour */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let tourIndex = -1;

const TOUR = [
  {
    title: "Welcome",
    text: "Earth Analogue Finder compares every land cell on Earth with a Moon or Mars base site, using NASA data, and ranks the closest matches.",
    run: async () => { if (eye.open) eye.close(); chooseTargetById("lunar_south_pole"); showList(); await sleep(300); globe.home(); },
  },
  {
    title: "The target",
    text: "Target: the lunar south pole, where Artemis crews will land. Its signature, on the left, comes from NASA missions: no rain, no plants, huge temperature swings, rugged ground.",
    run: async () => { document.querySelector(".panel-left").scrollTo({ top: 0, behavior: "smooth" }); },
  },
  {
    title: "Where Earth matches",
    text: "The globe lights up where Earth behaves most like the target. Brighter means a closer match. Here is the top-ranked site.",
    run: async () => { await waitForData(); selectResult(state.data.results[0]); },
  },
  {
    title: "Why it matches",
    text: "Every score is explained criterion by criterion, with the dataset behind each number. Hover any numbered site to preview it from orbit.",
    run: async () => { if (!state.selected) selectResult(state.data.results[0]); },
  },
  {
    title: "God's Eye",
    text: "Descend into the site's real terrain in 3D, lit the way the Sun lights the lunar pole: never more than about 1.5° above the horizon, with black shadows.",
    run: async () => { await openGodsEye(state.selected || state.data.results[0], "lunar"); },
  },
  {
    title: "Mars",
    text: "Switch to Mars: Jezero Crater, where Perseverance landed. The ranking changes completely: now hot, flat, bare deserts win.",
    run: async () => { if (eye.open) eye.close(); chooseTargetById("jezero_crater"); await sleep(1600); },
  },
  {
    title: "Is it right?",
    text: () => `We test it. Known analog sites such as Haughton Crater and the Atacama must outscore rainforest and farmland. ROC-AUC = ${state.data?.validation?.auc?.toFixed(2) ?? "…"} (1.00 is perfect).`,
    run: async () => { await waitForData(); selectTab("validation"); renderValidation(); },
  },
  {
    title: "Your turn",
    text: "Change the weights, search any place, pin sites to compare, or type your own target. Press ? for keyboard shortcuts.",
    run: async () => { selectTab("results"); },
  },
];

async function waitForData() {
  for (let i = 0; i < 50 && !state.data; i++) await sleep(100);
}

function chooseTargetById(id) {
  const card = document.querySelector(`.target-card[data-id="${id}"]`);
  if (card && card.getAttribute("aria-checked") !== "true") card.click();
}

async function tourGo(i) {
  tourIndex = Math.max(0, Math.min(TOUR.length - 1, i));
  const step = TOUR[tourIndex];
  document.body.classList.add("touring");
  $("tourCaption").hidden = false;
  $("tourStep").textContent = `${tourIndex + 1} / ${TOUR.length} · ${step.title}`;
  $("tourText").textContent = typeof step.text === "function" ? step.text() : step.text;
  $("tourBar").style.width = `${((tourIndex + 1) / TOUR.length) * 100}%`;
  $("tourPrev").disabled = tourIndex === 0;
  $("tourNext").textContent = tourIndex === TOUR.length - 1 ? "Finish" : "Next";
  try { await step.run(); } catch (err) { toast(err.message); }
  if (typeof step.text === "function") $("tourText").textContent = step.text();
}

function tourStop() {
  tourIndex = -1;
  $("tourCaption").hidden = true;
  document.body.classList.remove("touring");
}

function wireTour() {
  $("startTour").addEventListener("click", () => tourGo(0));
  $("tourNext").addEventListener("click", () => (tourIndex >= TOUR.length - 1 ? tourStop() : tourGo(tourIndex + 1)));
  $("tourPrev").addEventListener("click", () => tourGo(tourIndex - 1));
  $("tourStop").addEventListener("click", tourStop);
}

/* ---------------------------------------------------------------- keyboard */

function stepSite(delta) {
  const list = state.data?.results;
  if (!list?.length) return;
  const i = state.selected ? list.findIndex((r) => r.index === state.selected.index) : -1;
  selectResult(list[(i + delta + list.length) % list.length]);
}

function wireKeys() {
  document.addEventListener("keydown", (e) => {
    const typing = e.target.closest("input, select, textarea, [contenteditable]");
    const dialogOpen = document.querySelector("dialog[open]");
    if (e.key === "Escape") {
      if (eye.open) { eye.close(); e.preventDefault(); return; }
      if (tourIndex >= 0) { tourStop(); return; }
      if (!dialogOpen && !typing && !$("detail").hidden) { showList(); return; }
      return;
    }
    if (typing || dialogOpen || e.ctrlKey || e.metaKey || e.altKey) return;
    if (eye.open) return;
    const k = e.key;
    const act = {
      "/": () => $("search").focus(),
      j: () => stepSite(1),
      k: () => stepSite(-1),
      e: () => openGodsEye(state.selected || (state.pick && null) || state.data?.results[0]),
      p: () => state.selected && togglePin(state.selected),
      g: () => setView("globe"),
      m: () => setView("map"),
      "+": () => $("zoomIn").click(),
      "=": () => $("zoomIn").click(),
      "-": () => $("zoomOut").click(),
      0: () => $("zoomHome").click(),
      v: () => selectTab("validation"),
      t: () => tourGo(0),
      "?": () => $("keysDialog").showModal(),
    }[k.length === 1 ? k.toLowerCase() : k];
    if (act) { e.preventDefault(); act(); }
  });
}

/* ------------------------------------------------------------ tabs, dialogs */

function selectTab(name) {
  const isResults = name === "results";
  $("tabResults").setAttribute("aria-selected", String(isResults));
  $("tabValidation").setAttribute("aria-selected", String(!isResults));
  $("results").hidden = !isResults;
  $("validation").hidden = isResults;
}

function renderMethod() {
  const rows = state.criteria.map((c) => `<tr><td>${esc(c.label)}</td><td>${esc(c.description)}</td>
    <td><a href="${esc(c.source_url)}" target="_blank" rel="noopener"><code>${esc(c.dataset_id)}</code></a></td></tr>`).join("");
  const fit = state.sources?.lst_fit;
  $("methodContent").innerHTML = `
    <p>We describe each Moon or Mars base site with a handful of measurable conditions, measure the same
      conditions for every half-degree cell of Earth's land (${fmtInt(state.health.candidate_cells)} cells) from NASA and
      partner data, and rank the cells by how closely they match. There is no AI in the scoring: it is plain,
      tested arithmetic, and every number links to its source.</p>
    <h3>1 · Target signature</h3>
    <p>Each target criterion comes from a published measurement (LRO Diviner, LOLA, Chang'E-2, HiRISE, Mars 2020 MEDA),
      cited on the target card. Two adjustments keep the comparison honest:</p>
    <ul><li><b>Beyond Earth:</b> the Moon's 120 K day-night swing exists nowhere on Earth, so it is matched against Earth's
      most extreme value (99.5th percentile) instead of penalising every cell equally.</li>
      <li><b>Different scale:</b> slopes measured over 20–50 m cannot be compared with 55 km Earth cells, so terrain targets
      are expressed as Earth terrain classes (for example "rugged" = Earth's 85th percentile), with the original measurement shown.</li></ul>
    <h3>2 · Score</h3>
    <div class="formula">similarity<sub>k</sub> = clip(1 − |earth<sub>k</sub> − target<sub>k</sub>| / range<sub>k</sub>, 0, 1)<br>
      score = ∏<sub>k</sub> similarity<sub>k</sub><sup>w<sub>k</sub> / Σw</sup> &nbsp;&nbsp;(weighted geometric mean)</div>
    <p>A geometric mean lets one completely mismatched criterion veto a site: a rainforest cannot become a lunar analog by having the right slope.</p>
    <p>Some criteria are left out for some targets ("not used"), and a target can set its own default weights: the Haworth
      cold trap switches on <b>mean temperature</b>, because "cold" is what defines a cold trap.</p>
    <h3>3 · Validate</h3>
    <p>Known analog sites (Haughton Crater, McMurdo Dry Valleys, Atacama, Apollo training sites and others, each cited) should
      outscore densely vegetated reference points. The Validation tab reports ROC-AUC for the current target and weights.</p>
    <h3>4 · Look closer: God's Eye</h3>
    <p>Open any site in 3D. Its terrain comes from AWS Terrain Tiles at about 150 m, draped with Sentinel-2 cloud-free
      imagery (ESA Copernicus data processed by EOX). The panel measures relief and slope inside the scored cell at that
      finer scale, including the share of ground a rover could drive (slopes under 15°). You can light it with a lunar polar sun.</p>
    <h3>Criteria and data</h3>
    <table class="src-table"><thead><tr><th>Criterion</th><th>What it measures</th><th>Dataset</th></tr></thead><tbody>${rows}</tbody></table>
    <h3>Known limitations</h3>
    <ul>
      <li>Cells are 0.5° (~55 km). A small feature (a summit, a crater floor) is averaged with its surroundings; Mauna Kea's cell includes its forested slopes.</li>
      <li>Where MODIS has no land-surface-temperature data (mainly Antarctica) the day-night swing is predicted from NASA POWER skin temperature${fit ? ` (linear fit, r = ${fit.r})` : ""}.</li>
      <li>NASA POWER precipitation is a reanalysis (MERRA-2) and can overestimate polar deserts.</li>
      <li>Terrain beyond ±85° latitude is not covered by the elevation tiles, so the far polar interiors are not scored.</li>
      <li>This is a screening tool for choosing where to look, not a site survey.</li>
    </ul>`;
}

function renderSources() {
  const s = state.sources;
  const datasets = (s.datasets || []).map((d) => `<tr><td>${esc(d.title)}</td><td>${esc(d.used_for || "")}</td>
    <td><a href="${esc(d.url)}" target="_blank" rel="noopener"><code>${esc(d.dataset_id)}</code></a></td></tr>`).join("");
  const targets = s.targets.map((t) => {
    const items = Object.entries(t.criteria).map(([k, c]) => `<li>${esc(spec(k)?.label || k)}: <a href="${esc(c.source_url)}" target="_blank" rel="noopener"><code>${esc(c.dataset_id)}</code></a> (${esc(c.confidence)} confidence)</li>`).join("");
    return `<h3>${esc(state.targets.find((x) => x.id === t.id)?.name || t.id)}</h3><ul>${items}</ul>`;
  }).join("");
  const maps = Object.entries(s.basemaps).map(([k, b]) => `<li>${esc(k)}: <a href="${esc(b.url)}" target="_blank" rel="noopener">${esc(b.credit)}</a></li>`).join("");
  $("sourcesContent").innerHTML = `
    <h3>Earth data (scored)</h3>
    <table class="src-table"><thead><tr><th>Dataset</th><th>Used for</th><th>ID</th></tr></thead><tbody>${datasets}</tbody></table>
    ${targets}
    <h3>Known analog catalog</h3><p>${esc(s.analogs.provenance)} ${s.analogs.sites} sites.</p>
    <h3>Imagery</h3><ul>${maps}
      <li>God's Eye terrain: AWS Terrain Tiles (Terrarium): SRTM, GMTED, ETOPO1 and others</li>
      <li>God's Eye imagery: EOxCloudless https://cloudless.eox.at by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020), CC BY-NC-SA 4.0</li>
      <li>Detail tiles and hover previews: NASA Blue Marble via NASA GIBS</li></ul>
    <h3>Software</h3><p>three.js (MIT), FastAPI, NumPy, SciPy, rasterio, zarr. Place names: Natural Earth (public domain).</p>`;
}

function wireDialogs() {
  const open = (id, render) => () => { render(); $(id).showModal(); };
  $("openMethod").addEventListener("click", open("methodDialog", renderMethod));
  $("openSources").addEventListener("click", open("sourcesDialog", renderSources));
  $("openKeys").addEventListener("click", () => $("keysDialog").showModal());
  document.querySelectorAll("dialog").forEach((d) => {
    d.querySelector("[data-close]").addEventListener("click", () => d.close());
    d.addEventListener("click", (e) => { if (e.target === d) d.close(); });
  });
}

/* ------------------------------------------------------- shareable links */

function readHash() {
  const out = {};
  for (const part of location.hash.replace(/^#/, "").split("&")) {
    const [k, v] = part.split("=");
    if (k) out[decodeURIComponent(k)] = decodeURIComponent(v || "");
  }
  return out;
}

function writeHash() {
  const parts = [state.mode === "custom" ? "target=custom" : `target=${state.targetId}`];
  if (state.topK !== 20) parts.push(`top=${state.topK}`);
  if (state.view !== "globe") parts.push(`view=${state.view}`);
  if (state.layer !== "score") parts.push(`layer=${state.layer}`);
  if (state.selected?.rank) parts.push(`site=${state.selected.rank}`);
  if ($("tabValidation").getAttribute("aria-selected") === "true") parts.push("tab=validation");
  history.replaceState(null, "", `#${parts.join("&")}`);
}

/* -------------------------------------------------------------------- boot */

async function init() {
  wireDialogs();
  wireSearch();
  wireTour();
  wireKeys();
  document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("tabResults").addEventListener("click", () => selectTab("results"));
  $("tabValidation").addEventListener("click", () => selectTab("validation"));
  $("chipAuc").addEventListener("click", () => selectTab("validation"));
  $("showKnown").addEventListener("change", renderMarkers);
  $("exportGeojson").addEventListener("click", exportGeojson);
  $("exportCsv").addEventListener("click", exportCsv);
  globe.controls.addEventListener("start", hidePeek);
  $("zoomIn").addEventListener("click", () => (state.view === "globe" ? globe.zoomBy(0.7) : flat.zoomBy(1.6)));
  $("zoomOut").addEventListener("click", () => (state.view === "globe" ? globe.zoomBy(1 / 0.7) : flat.zoomBy(1 / 1.6)));
  $("zoomHome").addEventListener("click", () => (state.view === "globe" ? globe.home() : flat.home()));
  $("resetWeights").addEventListener("click", () => {
    state.weights = state.mode === "custom" ? { ...state.defaults } : { ...target().default_weights };
    renderWeights();
    runScore();
  });
  $("topK").addEventListener("input", (e) => {
    state.topK = Number(e.target.value);
    $("topKValue").textContent = state.topK;
    topKSoon();
  });
  $("overlayOpacity").addEventListener("input", (e) => {
    const v = Number(e.target.value) / 100;
    globe.setOverlayOpacity(v);
    flat.setOverlayOpacity(v);
  });
  $("layer").addEventListener("change", (e) => setLayer(e.target.value).catch((err) => toast(err.message)));

  try {
    const [health, targets, criteria, sources, analogs] = await Promise.all([
      api("/api/health"), api("/api/targets"), api("/api/criteria"), api("/api/sources"), api("/api/analogs"),
    ]);
    if (!health.ok) throw new Error(health.detail || "backend not ready");
    Object.assign(state, { health, targets: targets.targets, criteria: criteria.criteria, sources, analogs });
    for (const w of criteria.weights) state.defaults[w.key] = w.weight;
    const hash = readHash();
    state.targetId = state.targets.some((t) => t.id === hash.target) ? hash.target : state.targets[0].id;
    state.weights = { ...target().default_weights };
    if (hash.top && Number(hash.top) >= 5) state.topK = Math.min(100, Number(hash.top));
    $("topK").value = state.topK;
    $("topKValue").textContent = state.topK;

    $("chipMode").innerHTML = health.offline ? `<span class="dot on"></span>Offline mode · local data` : `<span class="dot"></span>Online`;
    $("chipCells").textContent = `${fmtInt(health.candidate_cells)} land cells`;
    const layer = $("layer");
    layer.appendChild(new Option("Analog score", "score"));
    for (const c of state.criteria) layer.appendChild(new Option(`Data: ${c.label}`, c.key));

    renderTargets();
    renderTargetDetail();
    renderWeights();

    $("loadingText").textContent = "Loading NASA Blue Marble…";
    await Promise.all([globe.setEarth("assets/earth_hd.jpg", "assets/earth.jpg"), flat.setBase("assets/earth_hd.jpg"), runScore()]);
    if (hash.view === "map") setView("map");
    if (hash.layer && state.criteria.some((c) => c.key === hash.layer)) {
      layer.value = hash.layer;
      await setLayer(hash.layer);
    }
    const site = state.data?.results.find((r) => String(r.rank) === hash.site);
    if (site) selectResult(site);
    if (site && hash.eye === "1") openGodsEye(site, hash.sun in SUN_PRESETS ? hash.sun : null);
    else if (state.data?.results.length) globe.flyTo(state.data.results[0].lat, state.data.results[0].lon);
    if (hash.tab === "validation") selectTab("validation");
    if (hash.dialog === "method") $("openMethod").click();
    if (hash.dialog === "sources") $("openSources").click();
    $("loading").classList.add("done");
    ["click", "change"].forEach((ev) => document.addEventListener(ev, () => setTimeout(writeHash, 0)));
  } catch (err) {
    $("loadingText").textContent = `Could not start: ${err.message}`;
    $("chipMode").textContent = "backend error";
    toast(err.message);
  }
}

init();
