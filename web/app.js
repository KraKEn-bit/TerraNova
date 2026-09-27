/* Minimal test harness for the Earth-analogue API. No framework, no CDN. */

const $ = (id) => document.getElementById(id);
const state = {
  targets: [],
  criteria: [],
  ranges: {},
  weights: {},
  values: {},
  selectedTarget: null,
  dirty: false,
  lastField: null,
  layer: "score",
};

const RAMP = [
  [43, 30, 92], [42, 111, 214], [55, 201, 165],
  [232, 212, 77], [232, 100, 42], [198, 40, 40],
];

function ramp(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function decodeFloat32(payload) {
  const raw = atob(payload.data);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  const view = new DataView(bytes.buffer);
  const out = new Float32Array(bytes.length / 4);
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true);
  return out;
}

async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* body was not json */ }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json();
}

function setStatus(text) {
  $("status").textContent = text;
}

/* ---------------------------------------------------------------- setup */

async function init() {
  try {
    const [health, targets, criteria] = await Promise.all([
      api("/api/health"), api("/api/targets"), api("/api/criteria"),
    ]);
    if (!health.ok) throw new Error(health.detail || "backend not ready");

    $("health").textContent = "api ok";
    $("health").classList.add("on");
    $("offline").textContent = health.offline ? "OFFLINE=1" : "network on";
    $("offline").classList.add(health.offline ? "on" : "off");
    $("cells").textContent = `${health.candidate_cells} land cells`;

    state.targets = targets.targets;
    state.criteria = criteria.criteria;
    for (const c of state.criteria) {
      state.ranges[c.key] = c;
      state.weights[c.key] = 1;
      state.values[c.key] = (c.min + c.max) / 2;
    }

    const select = $("target");
    for (const t of state.targets) {
      const opt = document.createElement("option");
      opt.value = t.id;
      opt.textContent = `${t.short_name} (${t.body})`;
      select.appendChild(opt);
    }
    const custom = document.createElement("option");
    custom.value = "";
    custom.textContent = "custom profile";
    select.appendChild(custom);
    select.value = state.targets[0].id;

    const layer = $("layer");
    layer.appendChild(new Option("score surface", "score"));
    for (const c of state.criteria) layer.appendChild(new Option(c.label, c.key));

    buildCriteria();
    select.onchange = () => applyTarget(select.value);
    layer.onchange = () => { state.layer = layer.value; renderLayer().catch(showError); };
    $("run").onclick = () => runScore();
    $("reset").onclick = () => { if (state.selectedTarget) applyTarget(state.selectedTarget); };

    applyTarget(state.selectedTarget);
    await runScore();
  } catch (err) {
    showError(err);
  }
}

function showError(err) {
  setStatus(`error - ${err.message}`);
  $("health").textContent = "error";
  $("health").classList.add("off");
}

function applyTarget(id) {
  state.selectedTarget = id || null;
  const target = state.targets.find((t) => t.id === id);
  const note = $("targetNote");
  if (!target) {
    note.textContent = "Move any slider to define your own profile.";
    state.dirty = true;
    return;
  }
  note.textContent = target.summary;
  for (const key of Object.keys(state.ranges)) {
    const value = target.criteria[key].value;
    state.values[key] = Math.min(Math.max(value, state.ranges[key].min), state.ranges[key].max);
    state.weights[key] = 1;
  }
  state.dirty = false;
  syncInputs();
}

function buildCriteria() {
  const host = $("criteria");
  host.innerHTML = "";
  for (const c of state.criteria) {
    const row = document.createElement("div");
    row.className = "crit";
    row.innerHTML = `
      <div class="head"><span>${c.label}</span>
        <span class="val" id="v_${c.key}"></span></div>
      <input type="range" id="s_${c.key}" min="${c.min}" max="${c.max}"
             step="${(c.max - c.min) / 200}">
      <div class="sub"><span>weight</span>
        <input type="number" id="w_${c.key}" min="0" max="5" step="0.1" value="1"></div>`;
    host.appendChild(row);
    row.querySelector(`#s_${c.key}`).oninput = (ev) => {
      state.values[c.key] = parseFloat(ev.target.value);
      state.dirty = true;
      $("target").value = "";
      $("targetNote").textContent = "custom profile";
      syncInputs();
    };
    row.querySelector(`#w_${c.key}`).oninput = (ev) => {
      const w = parseFloat(ev.target.value);
      if (Number.isFinite(w) && w >= 0) state.weights[c.key] = w;
    };
  }
  syncInputs();
}

function syncInputs() {
  for (const c of state.criteria) {
    const value = state.values[c.key];
    $(`s_${c.key}`).value = value;
    $(`w_${c.key}`).value = state.weights[c.key];
    $(`v_${c.key}`).textContent = `${value.toFixed(2)} ${c.unit === "1" ? "" : c.unit}`;
  }
}

function profile() {
  const out = {};
  for (const c of state.criteria) out[c.key] = state.values[c.key];
  return out;
}

/* --------------------------------------------------------------- scoring */

async function runScore() {
  try {
    setStatus("scoring…");
    const body = {
      criteria: profile(),
      weights: state.weights,
      top_k: parseInt($("topk").value, 10) || 20,
      min_separation_cells: parseInt($("sep").value, 10) || 0,
    };
    const data = await api("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderRows(data);
    $("range").textContent =
      `score over land: ${data.score_range[0].toFixed(4)} … ${data.score_range[1].toFixed(4)}`;
    setStatus(`scored ${body.top_k} cells`);
    await renderLayer();
  } catch (err) {
    showError(err);
  }
}

function renderRows(data) {
  const body = $("rows");
  body.innerHTML = "";
  for (const r of data.results) {
    const tr = document.createElement("tr");
    const claims = (r.rationale.claims || []).map((c) =>
      `<div class="claim">${c.text} <span class="src">[${c.dataset_id || "n/a"}]</span></div>`).join("");
    const caveats = (r.rationale.caveats || []).map((c) =>
      `<div class="claim why">${c.text} <a href="${c.source_url}" target="_blank" rel="noopener">source</a></div>`).join("");
    tr.innerHTML = `
      <td>${r.rank}</td>
      <td class="score">${(r.score * 100).toFixed(2)}%</td>
      <td>${r.label.text}</td>
      <td>${r.lat.toFixed(2)}, ${r.lon.toFixed(2)}</td>
      <td class="why"><details><summary>${r.rationale.headline}</summary>
        ${claims}${caveats ? `<div class="note">caveats</div>${caveats}` : ""}</details></td>`;
    body.appendChild(tr);
  }
}

/* ------------------------------------------------------------------- map */

async function renderLayer() {
  let payload;
  let name;
  if (state.layer === "score") {
    const target = state.targets.find((t) => t.id === state.selectedTarget);
    const query = new URLSearchParams();
    if (target && !state.dirty) query.set("target_id", target.id);
    else query.set("target_id", state.targets[0].id);
    payload = await api(`/api/scorefield?${query}`);
    name = "score";
  } else {
    payload = await api(`/api/predictor?key=${encodeURIComponent(state.layer)}`);
    name = state.ranges[state.layer].label;
  }

  const values = decodeFloat32(payload);
  const lo = payload.min;
  const hi = payload.max;
  const ctx = $("map").getContext("2d");
  const image = ctx.createImageData(720, 360);

  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    const o = i * 4;
    if (!Number.isFinite(v) || lo === null || hi === null) {
      image.data[o + 3] = 0;
      continue;
    }
    const [r, g, b] = ramp(hi > lo ? (v - lo) / (hi - lo) : 0.5);
    image.data[o] = r; image.data[o + 1] = g; image.data[o + 2] = b; image.data[o + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);

  $("legLo").textContent = lo === null ? "-" : lo.toFixed(3);
  $("legHi").textContent = hi === null ? "-" : hi.toFixed(3);
  $("legName").textContent = name;
  state.lastField = { lo, hi, name };
}

async function inspectAt(clientX, clientY) {
  const canvas = $("map");
  const rect = canvas.getBoundingClientRect();
  const x = Math.floor((clientX - rect.left) / rect.width * 720);
  const y = Math.floor((clientY - rect.top) / rect.height * 360);
  const col = Math.min(719, Math.max(0, x));
  const row = Math.min(359, Math.max(0, y));
  const lat = 90 - (row + 0.5) * 0.5;
  const lon = -179.75 + col * 0.5;
  try {
    const cell = await api(`/api/cell?lat=${lat}&lon=${lon}`);
    const rows = Object.entries(cell.values).map(([k, v]) => {
      const spec = state.ranges[k];
      const text = v === null ? "no data" : `${v} ${spec.unit === "1" ? "" : spec.unit}`;
      return `<div><span class="k">${spec.label}</span><br><span class="v">${text}</span></div>`;
    }).join("");
    $("inspect").innerHTML = `
      <div><span class="k">${cell.label.text}</span><br>
        <span class="v">${cell.lat.toFixed(2)}, ${cell.lon.toFixed(2)}</span><br>
        <span class="k">${cell.candidate ? "candidate land cell" : "not a candidate cell"}</span></div>
      ${rows}`;
  } catch (err) {
    setStatus(`inspect failed - ${err.message}`);
  }
}

$("map").addEventListener("click", (ev) => inspectAt(ev.clientX, ev.clientY));
init();
