"""Render a self-contained interactive HTML report from run logs.

Run data is embedded inline, so the page needs no server and works offline. It
is the same artifact used for development and for sharing, which removes any
chance of the demo drifting from the tool.
"""

from __future__ import annotations

import json

from .runs import CONDITION_COLORS

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; padding: 2rem;
    font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif;
    background: #fff; color: #16181d;
  }
  @media (prefers-color-scheme: dark) {
    body { background: #14161a; color: #e8eaed; }
    .card { background: #1c1f26 !important; border-color: #2b2f38 !important; }
    .note { background: #22262e !important; }
  }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  .sub { opacity: .7; margin: 0 0 1.5rem; font-size: .9rem; }
  .note {
    background: #f4f5f7; border-radius: 8px; padding: .8rem 1rem;
    font-size: .85rem; margin-bottom: 1.5rem;
  }
  .label {
    display: inline-block; font-size: .7rem; text-transform: uppercase;
    letter-spacing: .04em; padding: .1rem .4rem; border-radius: 4px;
    background: #e0e3e8; color: #40454f; margin-right: .4rem;
  }
  .grid { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
  .card { background: #fff; border: 1px solid #e3e6ea; border-radius: 10px; padding: 1rem; }
  .card h2 { font-size: .95rem; margin: 0 0 .6rem; }
  canvas { width: 100%; height: auto; display: block; }
  .controls {
    display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; margin-bottom: 1.2rem;
  }
  .toggles { display: flex; flex-wrap: wrap; gap: .6rem; }
  .toggle { display: flex; align-items: center; gap: .3rem; font-size: .85rem; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  input[type=range] { width: 260px; }
  .empty { opacity: .6; font-style: italic; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<p class="sub">Collapse dynamics across four JEPA collapse-prevention conditions.</p>

<div class="note">
  <span class="label">replication</span>
  Whether stop-gradient still matters under SIGReg at longer training durations is a
  <strong>replication attempt</strong> of a result encountered secondhand, not an established
  finding.
  <span class="label">interpretation</span>
  The SIGReg implementation was written from a summary-level reading of LeJEPA and has not
  been validated against the reference implementation.
  <span class="label">ours</span>
  Whether cheap diagnostics depart before probe accuracy is the only potentially new claim,
  and only holds if the gap above is observed at all.
</div>

<div class="controls">
  <label>step <output id="stepOut">0</output>
    <input type="range" id="stepSlider" min="0" max="0" value="0" step="1">
  </label>
  <div class="toggles" id="toggles"></div>
</div>

<div class="grid">
  <div class="card">
    <h2>Linear probe accuracy</h2>
    <canvas id="probe" width="640" height="380"></canvas>
  </div>
  <div class="card">
    <h2>Effective rank (participation ratio)</h2>
    <canvas id="rank" width="640" height="380"></canvas>
  </div>
  <div class="card">
    <h2>Mean feature std</h2>
    <canvas id="std" width="640" height="380"></canvas>
  </div>
  <div class="card">
    <h2>Embedding cloud at selected step (2D PCA)</h2>
    <canvas id="cloud" width="640" height="380"></canvas>
  </div>
</div>

<script>
const RUNS = __RUNS__;
const COLORS = __COLORS__;
const conditions = Object.keys(RUNS);
const enabled = new Set(conditions);

const allSteps = [...new Set(conditions.flatMap(c => RUNS[c].map(r => r.step)))].sort((a,b)=>a-b);
const slider = document.getElementById('stepSlider');
const stepOut = document.getElementById('stepOut');
slider.max = Math.max(allSteps.length - 1, 0);
stepOut.textContent = allSteps[0] ?? 0;

const togglesEl = document.getElementById('toggles');
conditions.forEach(condition => {
  const wrap = document.createElement('label');
  wrap.className = 'toggle';
  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = true;
  box.onchange = () => {
    box.checked ? enabled.add(condition) : enabled.delete(condition);
    redraw();
  };
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = COLORS[condition] || '#888';
  wrap.append(box, swatch, document.createTextNode(condition));
  togglesEl.append(wrap);
});

function series(condition, key) {
  return RUNS[condition].filter(r => r[key] !== undefined).map(r => [r.step, r[key]]);
}

function drawLineChart(canvasId, key) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 46;
  ctx.clearRect(0, 0, W, H);

  const active = conditions.filter(c => enabled.has(c));
  const points = active.flatMap(c => series(c, key));
  if (!points.length) {
    ctx.fillStyle = '#999'; ctx.font = '13px sans-serif';
    ctx.fillText('no data', pad, H / 2);
    return;
  }

  const xs = points.map(p => p[0]), ys = points.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs) || 1;
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMax - yMin < 1e-9) { yMax += 1; yMin -= 1; }
  const px = x => pad + (x - xMin) / (xMax - xMin || 1) * (W - pad * 1.4);
  const py = y => H - pad - (y - yMin) / (yMax - yMin) * (H - pad * 1.6);

  const axis = getComputedStyle(document.body).color;
  ctx.strokeStyle = axis; ctx.globalAlpha = .35; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, H - pad); ctx.lineTo(W - pad * .4, H - pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, H - pad); ctx.lineTo(pad, pad * .5); ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.fillStyle = axis; ctx.font = '11px sans-serif';
  ctx.fillText(yMax.toFixed(3), 4, py(yMax) + 4);
  ctx.fillText(yMin.toFixed(3), 4, py(yMin) + 4);
  ctx.fillText(String(xMin), pad, H - pad + 16);
  ctx.fillText(String(xMax), W - pad * 1.2, H - pad + 16);

  active.forEach(condition => {
    const data = series(condition, key);
    if (!data.length) return;
    ctx.strokeStyle = COLORS[condition] || '#888';
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)));
    ctx.stroke();
  });

  const selected = allSteps[+slider.value];
  ctx.strokeStyle = axis; ctx.globalAlpha = .3;
  ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(px(selected), pad * .5); ctx.lineTo(px(selected), H - pad); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha = 1;
}

function drawCloud() {
  const canvas = document.getElementById('cloud');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 30;
  ctx.clearRect(0, 0, W, H);

  const step = allSteps[+slider.value];
  const active = conditions.filter(c => enabled.has(c));
  const clouds = active.map(condition => {
    const record = RUNS[condition].find(r => r.step === step);
    return [condition, (record && record.projection) || []];
  }).filter(entry => entry[1].length);

  if (!clouds.length) {
    ctx.fillStyle = '#999'; ctx.font = '13px sans-serif';
    ctx.fillText('no projection at this step', pad, H / 2);
    return;
  }

  const all = clouds.flatMap(entry => entry[1]);
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (xMax - xMin < 1e-9) { xMax += 1; xMin -= 1; }
  if (yMax - yMin < 1e-9) { yMax += 1; yMin -= 1; }
  const px = x => pad + (x - xMin) / (xMax - xMin) * (W - pad * 2);
  const py = y => H - pad - (y - yMin) / (yMax - yMin) * (H - pad * 2);

  clouds.forEach(([condition, points]) => {
    ctx.fillStyle = COLORS[condition] || '#888';
    ctx.globalAlpha = .5;
    points.forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(px(x), py(y), 2.5, 0, Math.PI * 2); ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
  ctx.fillStyle = getComputedStyle(document.body).color;
  ctx.font = '11px sans-serif';
  ctx.fillText('axes are per-step PCA; scale is not comparable across steps', pad, H - 8);
}

function redraw() {
  stepOut.textContent = allSteps[+slider.value] ?? 0;
  drawLineChart('probe', 'probe_accuracy');
  drawLineChart('rank', 'effective_rank');
  drawLineChart('std', 'mean_feature_std');
  drawCloud();
}

slider.oninput = redraw;
redraw();
</script>
</body>
</html>
"""


def build_html(runs: dict[str, list[dict]], title: str = "jepa-lens") -> str:
    """Render the self-contained interactive page with run data embedded."""
    return (
        TEMPLATE.replace("__TITLE__", title)
        .replace("__RUNS__", json.dumps(runs))
        .replace("__COLORS__", json.dumps(CONDITION_COLORS))
    )
