"""Render a self-contained interactive HTML report from run logs.

Run data is embedded inline, so the page needs no server and works offline. It
is the same artifact used for development and for sharing, which removes any
chance of the demo drifting from the tool.

STRUCTURE. The page is built around Phase A's result rather than being a
generic training dashboard, for a specific reason: the obvious dashboard --
standardized probe accuracy and an effective-rank curve -- is exactly the pair
of metrics that rate the fully collapsed control as healthy. A viewer shown
those charts alone would conclude the dashboard was broken. So the page leads
with those charts, asks which encoder is dead, and then reveals the two that
answer it. The misleading view is the point, and it is labelled as such.

The scorecard takes across-seed means; the curves come from one seed and say
so. Mixing the two silently would be the same error the project documents.
"""

from __future__ import annotations

import html
import json

from .runs import CONDITION_COLORS

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#16181d; --card:#fff; --line:#e3e6ea;
          --muted:#f4f5f7; --chip:#e0e3e8; --chipfg:#40454f; --grid:#d8dce2; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161a; --fg:#e8eaed; --card:#1c1f26; --line:#2b2f38;
            --muted:#22262e; --chip:#333845; --chipfg:#c3c8d2; --grid:#333845; }
  }
  body { margin:0; padding:2rem 1.25rem 4rem; background:var(--bg); color:var(--fg);
         font:15px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  .wrap { max-width: 1180px; margin: 0 auto; }
  h1 { font-size:1.5rem; margin:0 0 .3rem; letter-spacing:-.01em; }
  h2 { font-size:1.05rem; margin:0 0 .2rem; }
  .sub { opacity:.72; margin:0 0 1.6rem; font-size:.92rem; }
  section { margin: 0 0 2.4rem; }
  .lead { font-size:.95rem; max-width:62ch; margin:.2rem 0 1rem; }
  .note { background:var(--muted); border-radius:8px; padding:.85rem 1rem;
          font-size:.85rem; margin:1rem 0; }
  .chip { display:inline-block; font-size:.68rem; text-transform:uppercase;
          letter-spacing:.05em; padding:.12rem .42rem; border-radius:4px;
          background:var(--chip); color:var(--chipfg); margin-right:.4rem;
          vertical-align:.08em; font-weight:600; }
  .grid { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:1rem; min-width:0; }
  .card h3 { font-size:.92rem; margin:0 0 .15rem; }
  .card p.cap { font-size:.78rem; opacity:.7; margin:0 0 .6rem; }
  canvas { width:100%; height:auto; display:block; }
  button.reveal { font:inherit; font-weight:600; cursor:pointer; padding:.55rem 1rem;
    border-radius:8px; border:1px solid var(--line); background:var(--card);
    color:var(--fg); }
  button.reveal:hover { background:var(--muted); }
  #revealBlock[hidden] { display:none; }
  .controls { display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center; margin:1rem 0; }
  .toggles { display:flex; flex-wrap:wrap; gap:.65rem; }
  .toggle { display:flex; align-items:center; gap:.32rem; font-size:.82rem; cursor:pointer; }
  .swatch { width:10px; height:10px; border-radius:2px; display:inline-block; flex:none; }
  input[type=range] { width:250px; }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; font-size:.86rem; width:100%; min-width:520px; }
  th, td { text-align:left; padding:.42rem .7rem; border-bottom:1px solid var(--line); }
  th { font-weight:600; opacity:.75; font-size:.78rem; text-transform:uppercase;
       letter-spacing:.04em; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .v-correct { color:#2e7d4f; font-weight:600; }
  .v-unresolved { color:#a06a1b; font-weight:600; }
  .v-blind { color:#c1443f; font-weight:600; }
  @media (prefers-color-scheme: dark) {
    .v-correct { color:#63c48d; } .v-unresolved { color:#e0a84a; } .v-blind { color:#f0847f; }
  }
  footer { margin-top:3rem; font-size:.8rem; opacity:.65; }
</style>
</head>
<body>
<div class="wrap">

<h1>__TITLE__</h1>
<p class="sub">Can you trust a self-supervised scenario embedding? &mdash; CIFAR-10
calibration bench, 7 conditions.</p>

<section>
  <h2>1. Which of these encoders is dead?</h2>
  <p class="lead">Below are the two evaluations almost everyone actually runs: a
  standardized linear probe, and nearest-neighbour retrieval by cosine similarity. One of
  these seven conditions has no collapse prevention at all &mdash; its encoder has
  collapsed to a single point. Pick it out.</p>

  <div class="grid">
    <div class="card">
      <h3>Standardized linear probe</h3>
      <p class="cap">The standard SSL evaluation protocol.</p>
      <canvas id="probeStd" width="640" height="360"></canvas>
    </div>
    <div class="card">
      <h3>Retrieval P@10 (cosine)</h3>
      <p class="cap">The default metric in essentially every vector database.</p>
      <canvas id="retr" width="640" height="360"></canvas>
    </div>
  </div>

  <div class="note">
    <span class="chip">ours</span>You cannot, reliably. The collapsed control finishes
    <strong>second of all seven</strong> on the standardized probe &mdash; behind only the
    healthy baseline, above every other arm &mdash; and its retrieval is statistically
    indistinguishable from a partially-working encoder (paired difference +0.002 over five
    seeds).
  </div>

  <p><button class="reveal" id="revealBtn">Show the two diagnostics that answer it</button></p>
</section>

<section id="revealBlock" hidden>
  <h2>The answer, and what those two charts could not show you</h2>
  <p class="lead">Same runs, two different measurements. Total variance is on a log axis
  because the gap spans five orders of magnitude. The unscaled probe is the same logistic
  regression as above, minus the standardization step.</p>

  <div class="grid">
    <div class="card">
      <h3>Total variance (log scale)</h3>
      <p class="cap">Trace of the embedding covariance.</p>
      <canvas id="totvar" width="640" height="360"></canvas>
    </div>
    <div class="card">
      <h3>Unscaled linear probe</h3>
      <p class="cap">Identical probe, features left at the scale the encoder emits.</p>
      <canvas id="probeUns" width="640" height="360"></canvas>
    </div>
  </div>

  <div class="note">
    <span class="chip">established</span>Standardizing divides each feature by its standard
    deviation; cosine retrieval L2-normalizes each vector. Both remove scale <em>by
    construction</em> &mdash; that is what they are for, and it is predicted from the
    definitions, not discovered here.
    <span class="chip">ours</span>What Phase&nbsp;A measures is the magnitude: a collapsed
    encoder's residual numerical noise is still a deterministic function of the input, and
    rescaled back to unit variance a linear model reads it happily &mdash; to 0.413 on
    10-class CIFAR-10, against a 0.100 chance floor.
  </div>
</section>

<section>
  <h2>2. Diagnostic scorecard on the collapsed control</h2>
  <p class="lead">The question a collapse diagnostic exists to answer is not &ldquo;is the
  dead encoder worse than the best one&rdquo; &mdash; almost anything clears that. It is
  <em>does it rank the dead encoder below one that is merely weak</em>. So the verdict
  column compares the collapsed control against <code>none_stopgrad</code>, the worst
  condition that still genuinely trains, and requires the margin to clear twice the
  across-seed standard deviation &mdash; the same claim rule the report applies everywhere
  else. Five seeds, final checkpoint.</p>
  <div class="tablewrap"><table id="scorecard">
    <thead><tr><th>diagnostic</th><th class="num">collapsed control</th>
    <th class="num">weakest real run</th><th class="num">healthy baseline</th>
    <th>verdict</th></tr></thead>
    <tbody></tbody>
  </table></div>
  <div class="note">
    <span class="chip">ours</span>The participation ratio does not merely miss the collapse
    &mdash; it reads <em>higher</em> on the dead encoder than on either real run. RankMe
    gets this case right, but reads 39.6 on <code>sigreg_nostopgrad</code>, whose variance
    is 127&times; below baseline. Each rank measure is blind to the mode the other catches,
    which is why they should not be substituted for one another despite both being
    described as effective rank. Retrieval points the right way by 0.002 &mdash; well
    inside its own across-seed noise, so it cannot separate the two at all.
  </div>
  <div class="note">
    <span class="chip">read this before quoting the table</span>
    &ldquo;Cannot separate&rdquo; is not the same as &ldquo;useless&rdquo;. Four diagnostics
    land there largely because the <em>comparison target</em> is unstable:
    <code>none_stopgrad</code> is bimodal across seeds, with total variance running 7.5,
    15.4, 16.9, 17.1 and 50.9. Absolute detection still works &mdash; nobody would look at
    total variance 0.0002 and cosine 1.0000 and call that encoder healthy. What the table
    shows is that these diagnostics are unreliable for <em>ranking or comparing</em>
    configurations, which is the job they are usually given. The two marked BLIND are a
    strictly worse failure: they point the wrong way.
  </div>
</section>

<section>
  <h2>3. Explore the runs</h2>
  <p class="lead">All conditions, all logged metrics, one seed. Drag the step slider to move
  the embedding cloud through training.</p>

  <div class="controls">
    <label>metric
      <select id="metricPick"></select>
    </label>
    <label>step <output id="stepOut">0</output>
      <input type="range" id="stepSlider" min="0" max="0" value="0" step="1">
    </label>
  </div>
  <div class="toggles" id="toggles"></div>

  <div class="grid" style="margin-top:1rem">
    <div class="card">
      <h3 id="freeTitle">metric</h3>
      <p class="cap">Log axis is used automatically when the range spans &gt;100&times;.</p>
      <canvas id="free" width="640" height="360"></canvas>
    </div>
    <div class="card">
      <h3>Embedding cloud at selected step</h3>
      <p class="cap">2D PCA of the evaluation embeddings. Axes rescale per frame.</p>
      <canvas id="cloud" width="640" height="360"></canvas>
    </div>
  </div>
</section>

<footer>
  <span class="chip">provenance</span>__PROVENANCE__
</footer>
</div>

<script>
const RUNS = __RUNS__;
const COLORS = __COLORS__;
const SCORECARD = __SCORECARD__;
const conditions = Object.keys(RUNS);
const enabled = new Set(conditions);

/* ---------- scorecard ---------- */
const tbody = document.querySelector('#scorecard tbody');
SCORECARD.forEach(row => {
  const tr = document.createElement('tr');
  const verdict = document.createElement('td');
  verdict.className = 'v-' + row.status;
  verdict.textContent = row.verdict;
  const cells = [row.name, row.collapsed, row.weakest, row.healthy].map((v, i) => {
    const td = document.createElement('td');
    if (i > 0) td.className = 'num';
    td.textContent = v;
    return td;
  });
  cells.forEach(c => tr.append(c));
  tr.append(verdict);
  tbody.append(tr);
});

/* ---------- shared plotting ---------- */
const allSteps = [...new Set(conditions.flatMap(c => RUNS[c].map(r => r.step)))]
  .sort((a, b) => a - b);

function series(condition, key) {
  return RUNS[condition]
    .filter(r => r[key] !== undefined && r[key] !== null && isFinite(r[key]))
    .map(r => [r.step, r[key]]);
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawLineChart(canvasId, key, opts) {
  opts = opts || {};
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 52;
  ctx.clearRect(0, 0, W, H);

  const active = conditions.filter(c => enabled.has(c));
  const data = active.map(c => [c, series(c, key)]).filter(([, s]) => s.length > 0);
  if (!data.length) {
    ctx.fillStyle = css('--fg'); ctx.globalAlpha = .55;
    ctx.font = '13px sans-serif';
    ctx.fillText('no data for this metric', pad, H / 2);
    ctx.globalAlpha = 1;
    return;
  }

  const xs = data.flatMap(([, s]) => s.map(p => p[0]));
  let ys = data.flatMap(([, s]) => s.map(p => p[1]));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);

  // Log axis when the spread makes a linear one unreadable. Total variance
  // spans 1e-4 to 1e2 across conditions; on a linear axis every curve but one
  // is pinned to the bottom edge and the collapse is invisible.
  const positive = ys.every(v => v > 0);
  const log = opts.log !== undefined ? opts.log : (positive && yMax / Math.max(yMin, 1e-12) > 100);
  const fwd = log ? Math.log10 : (v => v);
  if (log) { yMin = Math.max(yMin, 1e-12); }
  let lo = fwd(yMin), hi = fwd(yMax);
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const padY = (hi - lo) * 0.08;
  const allNonNegative = ys.every(v => v >= 0);
  lo -= padY; hi += padY;
  // A gridline at -7.35 on a metric with a floor of 1 reads as a broken chart.
  if (!log && allNonNegative && lo < 0) lo = 0;

  const X = v => pad + (W - pad - 14) * (xMax === xMin ? 0.5 : (v - xMin) / (xMax - xMin));
  const Yp = v => (H - pad) - (H - pad - 14) * ((fwd(v) - lo) / (hi - lo));

  ctx.strokeStyle = css('--grid'); ctx.lineWidth = 1;
  ctx.fillStyle = css('--fg'); ctx.globalAlpha = .75;
  ctx.font = '11px ui-monospace, monospace';
  for (let i = 0; i <= 4; i++) {
    const t = lo + (hi - lo) * (i / 4);
    const y = Yp(log ? Math.pow(10, t) : t);
    ctx.globalAlpha = .18; ctx.beginPath();
    ctx.moveTo(pad, y); ctx.lineTo(W - 14, y); ctx.stroke();
    ctx.globalAlpha = .75;
    const val = log ? Math.pow(10, t) : t;
    const label = (Math.abs(val) >= 1000 || (val !== 0 && Math.abs(val) < 0.01))
      ? val.toExponential(1) : val.toFixed(Math.abs(val) < 10 ? 2 : 1);
    ctx.fillText(label, 4, y + 3.5);
  }
  ctx.globalAlpha = .75;
  ctx.fillText(String(xMin), pad, H - pad + 16);
  ctx.fillText(String(xMax), W - 48, H - pad + 16);
  ctx.globalAlpha = 1;

  data.forEach(([condition, s]) => {
    ctx.strokeStyle = COLORS[condition] || '#888';
    ctx.lineWidth = condition === 'none_nostopgrad' ? 3 : 1.8;
    ctx.beginPath();
    s.forEach(([x, y], i) => {
      const px = X(x), py = Yp(y);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
}

function drawCloud(stepIndex) {
  const canvas = document.getElementById('cloud');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 24;
  ctx.clearRect(0, 0, W, H);
  const step = allSteps[stepIndex];

  const clouds = conditions.filter(c => enabled.has(c)).map(c => {
    const rec = RUNS[c].find(r => r.step === step);
    return [c, (rec && rec.projection) || []];
  }).filter(([, pts]) => pts.length > 0);
  if (!clouds.length) return;

  const all = clouds.flatMap(([, pts]) => pts);
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  let xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  // A collapsed cloud is a single point; without a floor the transform divides by zero.
  if (xMax - xMin < 1e-12) { xMin -= 1; xMax += 1; }
  if (yMax - yMin < 1e-12) { yMin -= 1; yMax += 1; }

  clouds.forEach(([condition, pts]) => {
    ctx.fillStyle = COLORS[condition] || '#888';
    ctx.globalAlpha = .5;
    pts.forEach(([x, y]) => {
      const px = pad + (W - 2 * pad) * (x - xMin) / (xMax - xMin);
      const py = H - pad - (H - 2 * pad) * (y - yMin) / (yMax - yMin);
      ctx.beginPath(); ctx.arc(px, py, 2, 0, 6.284); ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
}

/* ---------- controls ---------- */
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

const SKIP = new Set(['step', 'condition', 'projection']);
const metricKeys = [...new Set(conditions.flatMap(
  c => RUNS[c].flatMap(r => Object.keys(r))))]
  .filter(k => !SKIP.has(k)).sort();
const metricPick = document.getElementById('metricPick');
metricKeys.forEach(k => {
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = k;
  metricPick.append(opt);
});
metricPick.value = metricKeys.includes('rankme') ? 'rankme' : metricKeys[0];
metricPick.onchange = redraw;

const slider = document.getElementById('stepSlider');
const stepOut = document.getElementById('stepOut');
slider.max = Math.max(allSteps.length - 1, 0);
stepOut.textContent = allSteps[0] ?? 0;
slider.oninput = () => {
  stepOut.textContent = allSteps[+slider.value] ?? 0;
  drawCloud(+slider.value);
};

const revealBtn = document.getElementById('revealBtn');
revealBtn.onclick = () => {
  const block = document.getElementById('revealBlock');
  block.hidden = false;
  revealBtn.remove();
  redraw();
  block.scrollIntoView({ behavior: 'smooth', block: 'start' });
};

function redraw() {
  drawLineChart('probeStd', 'probe_accuracy', { log: false });
  drawLineChart('retr', 'retrieval_p10', { log: false });
  drawLineChart('totvar', 'total_variance', { log: true });
  drawLineChart('probeUns', 'probe_accuracy_unscaled', { log: false });
  const key = metricPick.value;
  document.getElementById('freeTitle').textContent = key;
  drawLineChart('free', key);
  drawCloud(+slider.value);
}

redraw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
</script>
</body>
</html>
"""


def _embed_json(value: object) -> str:
    """Serialize for inline <script> embedding.

    `</script>` inside a string literal would close the block early, so the
    forward slash is escaped. The data is run output rather than user input,
    but a condition name containing it would break the page silently.
    """
    return json.dumps(value).replace("</", "<\\/")


def build_html(
    runs: dict[str, list[dict]],
    title: str = "jepa-lens",
    scorecard: list[dict] | None = None,
    provenance: str = "",
) -> str:
    """Render the report to a single self-contained HTML string.

    `scorecard` rows are dicts with name/collapsed/healthy/verdict/correct. When
    omitted the table renders empty rather than inventing numbers: the curves
    come from one seed and the scorecard from across-seed means, so the page
    must never derive one from the other.
    """
    safe_title = html.escape(title, quote=True)
    return (
        TEMPLATE.replace("__TITLE__", safe_title)
        .replace("__PROVENANCE__", html.escape(provenance, quote=True))
        .replace("__RUNS__", _embed_json(runs))
        .replace("__COLORS__", _embed_json(CONDITION_COLORS))
        .replace("__SCORECARD__", _embed_json(scorecard or []))
    )
