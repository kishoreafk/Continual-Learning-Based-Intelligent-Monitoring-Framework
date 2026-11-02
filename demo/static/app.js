/* SAGE-Monitor Mission Control — frontend logic (no external deps).
   Polls the demo API, renders telemetry/score charts, gate timeline,
   event log, KPIs and the safety report. */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  sessionId: null,
  selectedDataset: "scripted",
  datasets: [],
  playhead: 0, total: 0,
  lastRenderedT: -1,
  chartBase: 0,          // first window index currently on the charts
  segments: [],
  injections: [],
  pollTimer: null,
  formError: null,
};

const ACTION_COLORS = {
  NO_ACTION: "#5b6a78", ADAPT: "#4da3ff", RECALL: "#7ce38b",
  REJECT: "#ffb454", ALERT: "#ff5d6c",
};

/* ---------------- canvas line chart (self-contained) ---------------- */
function drawLines(canvas, series, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.height;
  canvas.width = W * dpr; canvas.style.height = H + "px";
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const padL = 34, padR = 6, padT = 8, padB = 14;
  const iw = W - padL - padR, ih = H - padT - padB;
  if (!series.length || !series.some(s => s.data.length)) return;

  let ymin = opts.ymin ?? Infinity, ymax = opts.ymax ?? -Infinity;
  for (const s of series) for (const v of s.data) {
    if (ymin === null || v < ymin) ymin = v;
    if (ymax === null || v > ymax) ymax = v;
  }
  if (opts.fixed) { ymin = opts.fixed[0]; ymax = opts.fixed[1]; }
  if (!isFinite(ymin)) return;
  if (ymax - ymin < 1e-9) { ymax = ymin + 1; }
  const pad = (ymax - ymin) * 0.08;
  if (!opts.fixed) { ymin -= pad; ymax += pad; }
  const n = Math.max(...series.map(s => s.data.length));
  const x2i = (i) => padL + (iw * i) / Math.max(1, n - 1);
  const y2v = (v) => padT + ih - (ih * (v - ymin)) / (ymax - ymin);

  // gridlines
  ctx.strokeStyle = "#1a222c"; ctx.fillStyle = "#5b6a78";
  ctx.font = "9px ui-monospace, monospace"; ctx.lineWidth = 1;
  for (let g = 0; g <= 3; g++) {
    const v = ymin + ((ymax - ymin) * g) / 3;
    const y = y2v(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(1), 2, y + 3);
  }
  // reference lines
  for (const rl of (opts.refLines || [])) {
    if (rl.v < ymin || rl.v > ymax) continue;
    ctx.strokeStyle = rl.color || "#3a4652";
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padL, y2v(rl.v)); ctx.lineTo(W - padR, y2v(rl.v)); ctx.stroke();
    ctx.setLineDash([]);
  }
  // series
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.4;
    ctx.beginPath();
    for (let i = 0; i < s.data.length; i++) {
      const x = x2i(i), y = y2v(s.data[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
  if (opts.xlabel !== undefined) {
    ctx.fillStyle = "#5b6a78";
    ctx.fillText(opts.xlabel, W - 90, H - 3);
  }
}

/* ---------------- action strip timeline ---------------- */
function drawTimeline(canvas, records, total, playhead) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.height;
  canvas.width = W * dpr; canvas.style.height = H + "px"; canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const base = state.chartBase;
  const span = Math.max(1, playhead - base);
  const w = W / Math.max(1, total - base);
  // ground-truth shading (known anomaly windows) in dark red
  ctx.fillStyle = "#2a1418";
  for (const s of state.segments) {
    if (s.kind === "anomaly" || s.kind === "fault") {
      const a = Math.max(base, s.win_start), b = Math.min(total, s.win_end);
      if (b > a) ctx.fillRect((a - base) * w, 0, (b - a) * w, H);
    }
  }
  // injected events shading (outlined)
  for (const inj of state.injections) {
    const a = Math.max(base, inj.win_start), b = Math.min(total, inj.win_end);
    if (b <= a) continue;
    ctx.strokeStyle = inj.kind === "fault" ? "#ff5d6c" :
      inj.kind === "drift" ? "#4da3ff" : "#c792ea";
    ctx.setLineDash([3, 3]);
    ctx.strokeRect((a - base) * w, 1, (b - a) * w, H - 2);
    ctx.setLineDash([]);
  }
  // action ticks
  for (const r of records) {
    if (r.t < base) continue;
    if (r.action === "NO_ACTION") continue;
    ctx.fillStyle = ACTION_COLORS[r.action] || "#fff";
    const x = (r.t - base) * w;
    ctx.fillRect(x, 0, Math.max(1.5, w), H * (r.action === "ALERT" || r.action === "REJECT" ? 1 : 0.62));
  }
  // playhead
  const px = (playhead - base) * w;
  ctx.fillStyle = "#e6f1ff";
  ctx.fillRect(px - 0.75, 0, 1.5, H);
}

/* ---------------- data polling ---------------- */
async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}
function readNumber(id, label, min, max) {
  const raw = $(id).value.trim();
  const value = Number(raw);
  if (raw === "" || !Number.isFinite(value)) {
    throw new Error(`${label}: enter a number`);
  }
  if (min !== undefined && value < min) throw new Error(`${label}: must be ≥ ${min}`);
  if (max !== undefined && value > max) throw new Error(`${label}: must be ≤ ${max}`);
  return value;
}
function readInt(id, label, min, max) {
  const value = readNumber(id, label, min, max);
  if (!Number.isInteger(value)) throw new Error(`${label}: enter a whole number`);
  return value;
}
function readOptionalInt(id, label, min) {
  if ($(id).value.trim() === "") return null;
  return readInt(id, label, min);
}

async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

async function poll() {
  if (!state.sessionId) return;
  try {
    const snap = await jget(`/api/sessions/${state.sessionId}/stream?since=0&limit=5000`);
    render(snap);
  } catch (e) { console.warn(e); }
}

/* ---------------- rendering ---------------- */
function render(snap) {
  state.playhead = snap.playhead; state.total = snap.total_windows;
  state.segments = snap.segments; state.injections = snap.injections;

  // badges + progress
  $("session-badge").textContent = `session ${snap.session_id} · ${snap.status}`;
  $("stream-progress").textContent =
    `${snap.playhead} / ${snap.total_windows} windows · ` +
    `${Object.entries(snap.stats).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(" · ")}`;
  $("progress-bar").style.width = `${(100 * snap.playhead) / Math.max(1, snap.total_windows)}%`;
  const done = snap.status === "done";
  $("btn-play").disabled = !(snap.status === "ready" || snap.status === "paused");
  $("btn-pause").disabled = snap.status !== "streaming";
  $("btn-step").disabled = done || snap.status === "loading" || snap.status === "training";
  $("btn-report").disabled = !snap.records.length;
  $("btn-download").href = `/api/sessions/${state.sessionId}/report/download`;
  $("train-progress").textContent = state.formError ||
    (snap.status === "training" ? "training base encoder…" :
     snap.status === "loading" ? "building dataset…" :
     snap.status === "error" ? `error: ${snap.error}` :
     `status: ${snap.status}`);
  $("train-progress").style.color = state.formError ? "#ff5d6c" : "";

  // sliding chart window: keep ~450 latest windows on screen
  const VIEW = 450;
  if (snap.playhead - state.chartBase > VIEW || state.chartBase > snap.playhead) {
    state.chartBase = Math.max(0, snap.playhead - VIEW);
  }

  // chart data
  jget(`/api/sessions/${state.sessionId}/chart?since=${state.chartBase}&limit=${VIEW + 50}`)
    .then((cd) => {
      const t = [...Array(cd.series.length).keys()];
      const colors = ["#00e5a0", "#4da3ff", "#ffb454", "#7ce38b",
        "#c792ea", "#ff8f6c", "#69d2e7", "#f07178"];
      const series = cd.channels.map((c, i) => ({
        name: c, color: colors[i % colors.length],
        data: cd.series.map(row => row[i]),
      }));
      drawLines($("telemetry"), series,
        { xlabel: `window ${state.chartBase}→${state.chartBase + cd.series.length}` });
      $("chart-legend").innerHTML = cd.channels
        .map((c, i) => `<i class="sw" style="background:${colors[i % colors.length]}"></i>${c}`)
        .join(" ");
    }).catch(() => {});

  const recs = snap.records.slice(state.chartBase);
  drawLines($("scores"), [
    { name: "anomaly", color: "#ff5d6c", data: recs.map(r => r.anomaly) },
    { name: "drift", color: "#4da3ff", data: recs.map(r => r.drift) },
    { name: "ood", color: "#c792ea", data: recs.map(r => r.ood) },
  ], { fixed: [0, 1.05], refLines: [
    { v: 0.5, color: "#3a4652" }, { v: 0.85, color: "#7a3a42" }] });
  drawTimeline($("timeline"), snap.records, snap.total_windows, snap.playhead);
  $("timeline-meta").textContent =
    `window ${state.chartBase} → ${snap.playhead} of ${snap.total_windows}`;

  // KPIs from the newest record
  if (snap.records.length) {
    const r = snap.records[snap.records.length - 1];
    $("kpi-action").textContent = r.action;
    $("kpi-action").style.color = ACTION_COLORS[r.action];
    $("kpi-reason").textContent = r.reason;
    $("kpi-anomaly").textContent = r.anomaly.toFixed(3);
    $("kpi-anomaly").style.color = r.anomaly > 0.5 ? "#ff5d6c" : "#d7e1ea";
    $("kpi-drift").textContent = r.drift.toFixed(3);
    $("kpi-ood").textContent = r.ood.toFixed(3);
    $("kpi-gate").textContent = r.gate.toFixed(3);
    const st = snap.stats;
    $("kpi-regimes").textContent = snap.n_regimes;
    $("kpi-adapt").textContent = `${st.adaptations ?? 0} / ${st.recalls ?? 0}`;
    $("kpi-reject").textContent = `${st.rejects ?? 0} / ${st.alerts ?? 0}`;
    const u = st.unsafe_adaptations ?? 0;
    $("kpi-unsafe").textContent = u;
    $("kpi-unsafe").style.color = u ? "#ff5d6c" : "#7ce38b";
    document.querySelector(".kpi.action").classList.toggle("alerting",
      r.action === "ALERT" || r.action === "REJECT");
  }

  // event log (latest first)
  if (snap.events.length) {
    $("event-log").innerHTML = snap.events.slice().reverse().map(e => `
      <div class="ev"><span class="t">t=${e.t}</span>
      <span class="a" style="color:${ACTION_COLORS[e.action]}">${e.action}</span>
      <span class="r">${e.reason}</span></div>`).join("");
  }
  // segments
  if (state.segments.length) {
    $("segment-list").innerHTML = state.segments.map(s => {
      const col = s.kind === "fault" || s.kind === "anomaly" ? "#ff5d6c" :
        s.kind === "ood" ? "#c792ea" : s.kind === "drift" ? "#4da3ff" :
        s.kind === "recall" ? "#7ce38b" : "#3b4a59";
      const l = Math.max(4, 100 * (s.win_end - s.win_start) / Math.max(1, snap.total_windows));
      return `<div class="seg-row"><span class="muted small" style="min-width:132px">${s.name}</span>
        <div class="seg-bar" style="width:${l}%;background:${col}"></div>
        <span class="muted small">${s.win_start}–${s.win_end}</span></div>`;
    }).join("");
  }
  // injection log
  $("inj-log").textContent = state.injections.length
    ? state.injections.map(i => `${i.kind} @w${i.win_start}–${i.win_end}`).join(" · ")
    : "no injections";
}

/* ---------------- report ---------------- */
async function loadReport() {
  if (!state.sessionId) return;
  try {
    const rep = await jget(`/api/sessions/${state.sessionId}/report`);
    const d = rep.detection, pa = rep.detection_point_adjust, s = rep.stats;
    $("report-box").innerHTML = `
      <b>Detection (gate = REJECT/ALERT)</b><br>
      precision <b>${d.precision.toFixed(3)}</b> · recall <b>${d.recall.toFixed(3)}</b> ·
      F1 <b>${d.f1.toFixed(3)}</b><br>
      point-adjust precision ${pa.precision.toFixed(3)} · recall
      <b>${pa.recall.toFixed(3)}</b> · F1 <b>${pa.f1.toFixed(3)}</b><br>
      false-alarm rate <b style="color:${rep.false_alarm_rate ? '#ffb454' : '#7ce38b'}">${rep.false_alarm_rate.toFixed(4)}</b> ·
      missed intervals ${rep.missed}<br>
      <b>Continual</b><br>
      regimes ${rep.n_regimes} · adaptations ${s.adaptations} · recalls ${s.recalls} ·
      rejects ${s.rejects} · alerts ${s.alerts}<br>
      <b>Safety</b><br>
      unsafe adaptations <b style="color:${s.unsafe_adaptations ? '#ff5d6c' : '#7ce38b'}">${s.unsafe_adaptations}</b>
      (rate ${rep.unsafe_adaptation_rate.toFixed(3)}) — the gate must never learn a fault.`;
  } catch (e) { $("report-box").textContent = `report error: ${e.message}`; }
}

/* ---------------- dataset catalogue ---------------- */
async function loadDatasets() {
  state.datasets = await jget("/api/datasets");
  $("dataset-list").innerHTML = "";
  for (const d of state.datasets) {
    const el = document.createElement("div");
    el.className = "dataset-card" + (d.id === state.selectedDataset ? " sel" : "");
    el.innerHTML = `<div class="dn">${d.name}
      <span class="tag">${d.kind}</span>${d.recommended ? '<span class="tag">★ demo pick</span>' : ""}</div>
      <div class="dd">${d.description}</div>`;
    el.onclick = () => {
      state.selectedDataset = d.id;
      document.querySelectorAll(".dataset-card").forEach(c => c.classList.remove("sel"));
      el.classList.add("sel");
    };
    $("dataset-list").appendChild(el);
  }
}

/* ---------------- session controls ---------------- */
async function createSession() {
  state.formError = null;
  let body;
  try {
    body = {
      dataset: state.selectedDataset,
      window: readInt("p-window", "window", 20, 240),
      stride: readInt("p-stride", "stride", 1, 64),
      epochs: readInt("p-epochs", "epochs", 1, 50),
      seed: readInt("p-seed", "seed"),
      channels: readInt("p-channels", "channels", 2, 64),
      max_windows: readOptionalInt("p-max-windows", "max windows", 50),
      speed: readNumber("p-speed", "speed", 1, 1000),
    };
  } catch (e) {
    state.formError = `error: ${e.message}`;
    $("train-progress").textContent = state.formError;
    $("train-progress").style.color = "#ff5d6c";
    return;
  }
  try {
    $("train-progress").textContent = "creating session…";
    const res = await jpost("/api/sessions", body);
    state.sessionId = res.session_id;
    state.chartBase = 0;
    $("session-badge").textContent = `session ${res.session_id} · training`;
    setControls(true);
    $("btn-download").href = `/api/sessions/${state.sessionId}/report/download`;
    startPolling();
  } catch (e) {
    state.formError = `error: ${e.message}`;
    $("train-progress").textContent = state.formError;
    $("train-progress").style.color = "#ff5d6c";
  }
}

function setControls(on) {
  ["btn-play", "btn-pause", "btn-step", "inj-fault", "inj-drift",
   "inj-ood", "inj-clean", "btn-delete"].forEach(id => $(id).disabled = !on);
  if (!on) $("btn-report").disabled = true;
}

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(poll, 600);
  poll();
}

/* ---------------- wire up ---------------- */
window.addEventListener("resize", () => poll());
$("btn-create").onclick = createSession;
$("btn-play").onclick = () => jpost(`/api/sessions/${state.sessionId}/start`).then(startPolling).catch(e => alert(e.message));
$("btn-pause").onclick = () => jpost(`/api/sessions/${state.sessionId}/pause`).catch(e => alert(e.message));
$("btn-step").onclick = () => {
  try {
    const n = readInt("p-step", "step", 1, 500);
    jpost(`/api/sessions/${state.sessionId}/step?n=${n}`)
      .then(poll).catch(e => alert(e.message));
  } catch (e) { alert(e.message); }
};
async function applyTypedSpeed() {
  if (!state.sessionId) return;
  try {
    const speed = readNumber("p-speed", "speed", 1, 1000);
    await jpost(`/api/sessions/${state.sessionId}/speed`, { speed });
  } catch (e) { alert(e.message); }
}
$("p-speed").onchange = applyTypedSpeed;
$("btn-report").onclick = loadReport;
$("btn-delete").onclick = async () => {
  await fetch(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
  state.sessionId = null; setControls(false);
  $("session-badge").textContent = "no session";
  $("stream-progress").textContent = "—"; $("progress-bar").style.width = "0%";
};
const inj = (kind) => {
  try {
    const duration = readInt("p-inj-duration", "duration", 1, 2000);
    jpost(`/api/sessions/${state.sessionId}/inject`, { kind, duration })
      .then(poll).catch(e => alert(e.message));
  } catch (e) { alert(e.message); }
};
$("inj-fault").onclick = () => inj("fault");
$("inj-drift").onclick = () => inj("drift");
$("inj-ood").onclick = () => inj("ood");
$("inj-clean").onclick = () => inj("clean");

loadDatasets().then(() => jget("/api/health")).then(h => {
  $("device-badge").textContent =
    `device: ${h.device} · torch ${h.torch}` + (h.cuda_available ? " · GPU" : " · CPU demo");
});
startPolling();
