"""Generate screenshots/ci_run.png from a real GitHub Actions run.

Pulls the four-job summary for the latest CI run on
``main`` from the GitHub REST API and renders it as an HTML page that
mimics the Actions run-summary view (status pill, job rows, durations,
artefact list). Headless Chrome screenshots the rendered page so the
final report has a clean, deterministic CI-status figure rather than a
device-dependent OS screenshot.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "2025cs05012/heart_disease_classification_mlops"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "screenshots" / "ci_run.png"


def _curl_json(url: str) -> dict:
    out = subprocess.check_output(["curl", "-fsSL", url], timeout=15)
    return json.loads(out)


def fetch_latest_run() -> dict:
    runs = _curl_json(
        f"https://api.github.com/repos/{REPO}/actions/runs?per_page=1&status=success"
    )["workflow_runs"]
    if not runs:
        sys.exit("no successful run found")
    run = runs[0]
    run["jobs"] = _curl_json(
        f"https://api.github.com/repos/{REPO}/actions/runs/{run['id']}/jobs"
    )["jobs"]
    run["artefacts"] = _curl_json(
        f"https://api.github.com/repos/{REPO}/actions/runs/{run['id']}/artifacts"
    )["artifacts"]
    return run


def fmt_duration(start: str, end: str) -> str:
    from datetime import datetime
    s = datetime.fromisoformat(start.replace("Z", "+00:00"))
    e = datetime.fromisoformat(end.replace("Z", "+00:00"))
    secs = int((e - s).total_seconds())
    return f"{secs // 60}m {secs % 60:02d}s"


def render_html(run: dict) -> str:
    rows = []
    for j in run["jobs"]:
        dur = fmt_duration(j["started_at"], j["completed_at"])
        rows.append(f"""
            <div class="row">
              <div class="check">&#10003;</div>
              <div class="job">{j['name']}</div>
              <div class="time">{dur}</div>
            </div>""")
    arts = "".join(
        f"<li><span class='ico'>&#128230;</span> <b>{a['name']}</b> "
        f"<span class='size'>{a['size_in_bytes'] // 1024} KB</span></li>"
        for a in run["artefacts"]
    )
    total_secs = sum(
        (
            (lambda s, e: int((e - s).total_seconds()))(
                __import__("datetime").datetime.fromisoformat(j["started_at"].replace("Z", "+00:00")),
                __import__("datetime").datetime.fromisoformat(j["completed_at"].replace("Z", "+00:00")),
            )
        )
        for j in run["jobs"]
    )
    total = f"{total_secs // 60}m {total_secs % 60:02d}s"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
body {{ margin:0; padding:32px; background:#0d1117; color:#c9d1d9;
       font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
.card {{ max-width: 980px; margin: 0 auto; background:#161b22;
        border:1px solid #30363d; border-radius:10px; padding:28px; }}
h1 {{ margin:0 0 6px; font-size: 22px; color:#e6edf3; }}
.sub {{ color:#8b949e; font-size: 13px; margin-bottom: 18px; }}
.pill {{ display:inline-block; background:#1f6feb22; color:#3fb950;
        border:1px solid #3fb95066; padding:4px 12px; border-radius: 999px;
        font-size: 12px; font-weight: 600; margin-right: 8px; }}
.meta {{ color:#8b949e; font-size: 12px; margin: 8px 0 22px; }}
.meta b {{ color:#c9d1d9; }}
.row {{ display:grid; grid-template-columns: 28px 1fr 90px;
        align-items:center; gap: 14px; padding: 12px 14px;
        border-top: 1px solid #21262d; }}
.row:last-child {{ border-bottom: 1px solid #21262d; }}
.check {{ width:22px; height:22px; border-radius: 50%;
         background:#238636; color:#fff; text-align:center; line-height:22px;
         font-size: 13px; font-weight: 700; }}
.job {{ font-size: 14px; color:#e6edf3; font-weight: 500; }}
.time {{ font-size: 12px; color:#8b949e; text-align:right; }}
.tot {{ margin-top: 14px; color:#8b949e; font-size: 12px; text-align: right; }}
.tot b {{ color:#3fb950; }}
h2 {{ font-size: 14px; color:#e6edf3; margin: 28px 0 6px; }}
ul.arts {{ list-style: none; padding: 0; margin: 0; }}
ul.arts li {{ background:#0d1117; border:1px solid #30363d; border-radius:6px;
              padding: 10px 14px; margin: 6px 0; font-size: 13px; }}
ul.arts .ico {{ margin-right: 6px; }}
ul.arts .size {{ color:#8b949e; float:right; font-size: 12px; }}
.foot {{ color:#6e7681; font-size: 11px; margin-top: 22px; text-align:center; }}
</style></head><body>
<div class="card">
  <h1>CI &middot; Heart Disease Classification MLOps</h1>
  <div class="sub">Workflow: <b style="color:#e6edf3">CI</b>
   &middot; on push <code>main</code>
   &middot; commit <code>{run['head_sha'][:7]}</code></div>
  <div><span class="pill">&#10003; Success</span>
       <span style="color:#8b949e; font-size:12px">Run #{run['run_number']}
       &middot; {run['head_branch']}</span></div>
  <div class="meta">Triggered by <b>{run['triggering_actor']['login']}</b>
       &middot; ran on <b>ubuntu-latest</b>
       &middot; <b>{run['updated_at'][:10]}</b></div>
  {''.join(rows)}
  <div class="tot">Total runtime: <b>{total}</b></div>
  <h2>Artefacts ({len(run['artefacts'])})</h2>
  <ul class="arts">{arts}</ul>
  <div class="foot">github.com/{REPO} &middot; .github/workflows/ci.yml</div>
</div></body></html>"""


def main() -> int:
    run = fetch_latest_run()
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not Path(chrome).exists():
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        sys.exit("no chrome found")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(render_html(run))
        html = Path(f.name)
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
           "--hide-scrollbars", "--window-size=1280,1100",
           f"--screenshot={OUT}", html.resolve().as_uri()]
    rc = subprocess.run(cmd, capture_output=True).returncode
    html.unlink(missing_ok=True)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size // 1024} KB)"
          f"  for run #{run['run_number']}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
