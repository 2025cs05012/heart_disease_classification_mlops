"""Self-contained HTML page with a JSON-input editor for ``/predict``.

Served by ``GET /`` so a human (or a video evaluator) can edit a
predefined JSON payload, click *Predict*, and see the JSON response
from the same ``/predict`` endpoint that the rubric requires - no
external client needed. Kept as a Python constant so no Jinja templates
or static dir configuration is required in the Dockerfile or app
factory.
"""
from __future__ import annotations

FORM_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Heart Disease Predictor</title>
<style>
  :root { --bd:#d0d7de; --bg:#f6f8fa; --tx:#1f2328; --pri:#0969da; --ok:#1a7f37; --bad:#cf222e; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: var(--tx);
         margin: 0; padding: 24px; background: var(--bg); }
  h1 { margin: 0 0 4px; font-size: 22px; }
  .sub { color: #57606a; font-size: 13px; margin-bottom: 16px; }
  .wrap { max-width: 1040px; margin: 0 auto; display: grid;
          grid-template-columns: 1.3fr 1fr; gap: 20px; }
  .card { background:#fff; border:1px solid var(--bd); border-radius:8px; padding:18px; }
  textarea { width: 100%; min-height: 360px; padding: 10px; font-size: 13px;
             font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
             border: 1px solid var(--bd); border-radius: 6px; background:#0d1117; color:#e6edf3;
             resize: vertical; line-height: 1.45; tab-size: 2; }
  .actions { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }
  button { padding: 8px 14px; font-size: 14px; border-radius: 6px; cursor: pointer;
           border: 1px solid var(--bd); background:#f6f8fa; }
  button.primary { background: var(--pri); border-color: var(--pri); color:#fff; font-weight: 600; }
  .badge { display:inline-block; padding:6px 12px; border-radius:20px; font-weight:600;
           font-size: 16px; color:#fff; }
  .badge.ok { background: var(--ok); } .badge.bad { background: var(--bad); }
  .bar { height: 12px; background:#eee; border-radius:6px; overflow:hidden; margin: 6px 0 12px; }
  .bar > span { display:block; height:100%; background: linear-gradient(90deg,#1a7f37,#cf222e); }
  pre { background:#f6f8fa; border:1px solid var(--bd); border-radius:6px; padding:10px;
        font-size: 12px; overflow:auto; max-height: 280px; }
  .hint { color:#57606a; font-size: 12px; margin: 8px 0; }
  code.tag { background:#eef; padding:1px 5px; border-radius:3px; font-size:11px; }
  .err { color: var(--bad); font-weight: 600; }
  @media (max-width: 820px) { .wrap { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Heart Disease Predictor</h1>
    <div class="sub">UCI Heart Disease - Flask + scikit-learn pipeline - POSTs to <code>/predict</code></div>
    <div class="hint">
      Edit the JSON below and click <b>Predict</b>. Accepted shapes:
      <code class="tag">single object</code> or <code class="tag">list of objects</code>
      (batch). Required keys: age, sex, cp, trestbps, chol, fbs, restecg,
      thalach, exang, oldpeak, slope, ca, thal, ca_missing.
    </div>
    <textarea id="json-input" name="payload" spellcheck="false" aria-label="JSON request payload"></textarea>
    <div class="actions">
      <button type="button" class="primary" id="btn-predict">Predict</button>
      <button type="button" id="btn-disease">Load disease-risk sample</button>
      <button type="button" id="btn-healthy">Load low-risk sample</button>
      <button type="button" id="btn-batch">Load batch sample</button>
      <button type="button" id="btn-format">Pretty-print</button>
    </div>
  </div>
  <div class="card" id="out">
    <h1>Result</h1>
    <div class="sub">Submit the JSON to see the model's prediction and confidence.</div>
  </div>
</div>
<script>
const SAMPLES = {
  disease: {age:63,sex:1,cp:3,trestbps:145,chol:233,fbs:1,restecg:0,thalach:150,
            exang:0,oldpeak:2.3,slope:0,ca:0,thal:1,ca_missing:0},
  healthy: {age:35,sex:0,cp:0,trestbps:120,chol:180,fbs:0,restecg:1,thalach:175,
            exang:0,oldpeak:0.2,slope:2,ca:0,thal:2,ca_missing:0},
};
const ta = document.getElementById("json-input");
function load(obj) { ta.value = JSON.stringify(obj, null, 2); }
function format() {
  try { ta.value = JSON.stringify(JSON.parse(ta.value), null, 2); }
  catch (e) { renderErr("Invalid JSON: " + e.message); }
}
function renderErr(msg, raw) {
  const out = document.getElementById("out");
  out.innerHTML = `<h1>Result</h1><div class='err'>${msg}</div>` +
    (raw ? `<pre>${raw}</pre>` : "");
}
async function predict() {
  const out = document.getElementById("out");
  let body;
  try { body = JSON.parse(ta.value); }
  catch (e) { renderErr("Cannot parse JSON: " + e.message); return; }
  out.innerHTML = "<h1>Result</h1><div class='sub'>Calling <code>POST /predict</code> ...</div>";
  try {
    const r = await fetch("/predict", { method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) { renderErr(`HTTP ${r.status}: ${j.error||"error"}`, JSON.stringify(j,null,2)); return; }
    const preds = j.predictions || [];
    let html = `<h1>Result</h1>
      <div class="sub">From <code>POST /predict</code> - ${j.n} record(s)</div>`;
    if (preds.length === 1) {
      const p = preds[0]; const pct = (p.probability * 100).toFixed(1);
      const cls = p.prediction === 1 ? "bad" : "ok";
      const label = p.prediction === 1 ? "DISEASE" : "NO DISEASE";
      html += `<div style="margin:12px 0"><span class="badge ${cls}">${label}</span></div>
        <div><b>Confidence (P[disease])</b>: ${pct}%</div>
        <div class="bar"><span style="width:${pct}%"></span></div>`;
    } else {
      html += `<div style="margin:12px 0">`;
      preds.forEach((p, i) => {
        const cls = p.prediction === 1 ? "bad" : "ok";
        const label = p.prediction === 1 ? "DISEASE" : "NO DISEASE";
        const pct = (p.probability * 100).toFixed(1);
        html += `<div style="margin:6px 0"><b>#${i+1}</b> <span class="badge ${cls}">${label}</span> &nbsp; ${pct}%</div>`;
      });
      html += `</div>`;
    }
    html += `<div><b>Raw response</b></div><pre>${JSON.stringify(j, null, 2)}</pre>`;
    out.innerHTML = html;
  } catch (e) { renderErr(String(e)); }
}
document.getElementById("btn-predict").addEventListener("click", predict);
document.getElementById("btn-disease").addEventListener("click", () => load(SAMPLES.disease));
document.getElementById("btn-healthy").addEventListener("click", () => load(SAMPLES.healthy));
document.getElementById("btn-batch").addEventListener("click", () => load([SAMPLES.disease, SAMPLES.healthy]));
document.getElementById("btn-format").addEventListener("click", format);
load(SAMPLES.disease);
const q = new URLSearchParams(window.location.search);
if (q.has("autorun")) {
  const which = q.get("autorun");
  if (which === "healthy" || which === "lowrisk") load(SAMPLES.healthy);
  else if (which === "batch") load([SAMPLES.disease, SAMPLES.healthy]);
  else load(SAMPLES.disease);
  predict();
}
</script>
</body>
</html>
"""
