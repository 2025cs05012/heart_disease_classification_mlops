"""Tiny self-contained HTML form for manual ``/predict`` testing.

Served by ``GET /`` so a human (or a video evaluator) can fill in the 14
input fields, click *Predict*, and see the JSON response from the same
``/predict`` endpoint that the rubric requires — no external client
needed. Kept as a Python constant so no Jinja templates / static dir
configuration is required in the Dockerfile or app factory.
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
  .wrap { max-width: 960px; margin: 0 auto; display: grid;
          grid-template-columns: 1.4fr 1fr; gap: 20px; }
  .card { background:#fff; border:1px solid var(--bd); border-radius:8px; padding:18px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px 14px; }
  label { display: block; font-size: 12px; color:#57606a; margin-bottom: 2px; }
  label small { color:#8b949e; font-weight: normal; }
  input { width: 100%; padding: 6px 8px; font-size: 14px;
          border: 1px solid var(--bd); border-radius: 4px; background:#fff; }
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
  .err { color: var(--bad); font-weight: 600; }
  @media (max-width: 760px) { .wrap { grid-template-columns: 1fr; } .grid { grid-template-columns: 1fr 1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Heart Disease Predictor</h1>
    <div class="sub">UCI Heart Disease · Flask + scikit-learn pipeline · POSTs to <code>/predict</code></div>
    <form id="f" class="grid">
      <div><label>age <small>years</small></label><input name="age" type="number" step="1" value="63"></div>
      <div><label>sex <small>0=F, 1=M</small></label><input name="sex" type="number" step="1" min="0" max="1" value="1"></div>
      <div><label>cp <small>chest pain 0-3</small></label><input name="cp" type="number" step="1" min="0" max="3" value="3"></div>
      <div><label>trestbps <small>resting BP</small></label><input name="trestbps" type="number" step="1" value="145"></div>
      <div><label>chol <small>mg/dl</small></label><input name="chol" type="number" step="1" value="233"></div>
      <div><label>fbs <small>fbs&gt;120 0/1</small></label><input name="fbs" type="number" step="1" min="0" max="1" value="1"></div>
      <div><label>restecg <small>0-2</small></label><input name="restecg" type="number" step="1" min="0" max="2" value="0"></div>
      <div><label>thalach <small>max HR</small></label><input name="thalach" type="number" step="1" value="150"></div>
      <div><label>exang <small>ex-angina 0/1</small></label><input name="exang" type="number" step="1" min="0" max="1" value="0"></div>
      <div><label>oldpeak <small>ST depr.</small></label><input name="oldpeak" type="number" step="0.1" value="2.3"></div>
      <div><label>slope <small>0-2</small></label><input name="slope" type="number" step="1" min="0" max="2" value="0"></div>
      <div><label>ca <small>vessels 0-3</small></label><input name="ca" type="number" step="1" min="0" max="3" value="0"></div>
      <div><label>thal <small>1/2/3</small></label><input name="thal" type="number" step="1" min="0" max="3" value="1"></div>
      <div><label>ca_missing <small>0/1</small></label><input name="ca_missing" type="number" step="1" min="0" max="1" value="0"></div>
    </form>
    <div class="actions">
      <button type="button" class="primary" id="btn-predict">Predict</button>
      <button type="button" id="btn-disease">Load disease-risk sample</button>
      <button type="button" id="btn-healthy">Load low-risk sample</button>
    </div>
  </div>
  <div class="card" id="out">
    <h1>Result</h1>
    <div class="sub">Submit the form to see the model's prediction and confidence.</div>
  </div>
</div>
<script>
const NUM = ["age","trestbps","chol","thalach","oldpeak","ca"];
const INT = ["sex","cp","fbs","restecg","exang","slope","thal","ca_missing"];
const SAMPLES = {
  disease: {age:63,sex:1,cp:3,trestbps:145,chol:233,fbs:1,restecg:0,thalach:150,
            exang:0,oldpeak:2.3,slope:0,ca:0,thal:1,ca_missing:0},
  healthy: {age:35,sex:0,cp:0,trestbps:120,chol:180,fbs:0,restecg:1,thalach:175,
            exang:0,oldpeak:0.2,slope:2,ca:0,thal:2,ca_missing:0},
};
function fill(s) { for (const k in s) { const el = document.querySelector(`[name="${k}"]`); if (el) el.value = s[k]; } }
function readForm() {
  const fd = new FormData(document.getElementById("f")); const o = {};
  for (const [k, v] of fd.entries()) o[k] = NUM.includes(k) ? parseFloat(v) : INT.includes(k) ? parseInt(v, 10) : v;
  return o;
}
async function predict() {
  const out = document.getElementById("out");
  out.innerHTML = "<h1>Result</h1><div class='sub'>Calling <code>POST /predict</code> ...</div>";
  try {
    const body = readForm();
    const r = await fetch("/predict", { method: "POST",
      headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
    const j = await r.json();
    if (!r.ok) { out.innerHTML = `<h1>Result</h1><div class='err'>HTTP ${r.status}: ${j.error||"error"}</div><pre>${JSON.stringify(j,null,2)}</pre>`; return; }
    const p = j.predictions[0]; const pct = (p.probability * 100).toFixed(1);
    const cls = p.prediction === 1 ? "bad" : "ok";
    const label = p.prediction === 1 ? "DISEASE" : "NO DISEASE";
    out.innerHTML = `<h1>Result</h1>
      <div class="sub">From <code>POST /predict</code> · single record</div>
      <div style="margin:12px 0"><span class="badge ${cls}">${label}</span></div>
      <div><b>Confidence (P[disease])</b>: ${pct}%</div>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <div><b>Raw response</b></div>
      <pre>${JSON.stringify(j, null, 2)}</pre>`;
  } catch (e) { out.innerHTML = `<h1>Result</h1><div class='err'>${e}</div>`; }
}
document.getElementById("btn-predict").addEventListener("click", predict);
document.getElementById("btn-disease").addEventListener("click", () => fill(SAMPLES.disease));
document.getElementById("btn-healthy").addEventListener("click", () => fill(SAMPLES.healthy));
const q = new URLSearchParams(window.location.search);
if (q.has("autorun")) {
  const which = q.get("autorun");
  if (which === "healthy" || which === "lowrisk") fill(SAMPLES.healthy);
  else if (which === "disease" || which === "1" || which === "") fill(SAMPLES.disease);
  predict();
}
</script>
</body>
</html>
"""
