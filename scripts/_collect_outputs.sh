#!/usr/bin/env bash
# Internal helper: gather "expected outputs" T1..T9 into /tmp files.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

# T1
{
  echo "=== T1.OUT.A: cleaned dataset summary ==="
  $PY - <<'EOF'
import pandas as pd
df = pd.read_csv('data/processed/heart_disease_clean.csv')
print('shape:', df.shape)
print('target balance:', df['target'].value_counts(normalize=True).round(3).to_dict())
print()
print('missingness (top 5):')
for c, v in df.isna().mean().sort_values(ascending=False).head(5).items():
    print(f'  {c:<12} {v:.3f}')
EOF
  echo
  echo "=== T1.OUT.B: EDA figures on disk ==="
  ls -lh reports/figures/*.png 2>&1 | awk '{print $5, $NF}'
} > /tmp/T1.txt 2>&1

# T2
{
  echo "=== T2.OUT: trained-models comparison from reports/metrics.json ==="
  $PY - <<'EOF'
import json
m = json.loads(open('reports/metrics.json').read())
print(f"best_model = {m['best_model']}    n_train={m['n_train']}    n_test={m['n_test']}    cv={m['cv']}")
print()
hdr = f"{'model':<20} {'cv_roc_auc':>10}  {'accuracy':>9} {'precision':>10} {'recall':>8} {'f1':>6} {'roc_auc':>8}"
print(hdr); print('-'*len(hdr))
for name, info in m['models'].items():
    t = info['test_metrics']
    print(f"{name:<20} {info['cv_roc_auc_mean']:>10.4f}  {t['accuracy']:>9.4f} {t['precision']:>10.4f} {t['recall']:>8.4f} {t['f1']:>6.4f} {t['roc_auc']:>8.4f}")
print()
print('best_params for', m['best_model'], '=', json.dumps(m['models'][m['best_model']]['best_params']))
EOF
} > /tmp/T2.txt 2>&1

# T3
{
  echo "=== T3.OUT: MLflow runs ==="
  $PY - <<'EOF'
from pathlib import Path
import yaml
roots = []
for exp in sorted(Path('mlruns').iterdir()):
    if exp.is_dir() and (exp / 'meta.yaml').exists():
        try:
            meta = yaml.safe_load((exp / 'meta.yaml').read_text())
            roots.append((exp, meta.get('name', exp.name)))
        except Exception:
            pass
for exp, name in roots:
    runs = []
    for d in exp.iterdir():
        if not d.is_dir(): continue
        mp = d / 'meta.yaml'
        if not mp.exists(): continue
        try:
            meta = yaml.safe_load(mp.read_text())
            runs.append((meta.get('start_time', 0), d.name[:8], meta.get('status', '?'), meta.get('run_name', '')))
        except Exception:
            continue
    runs.sort(reverse=True)
    print(f'experiment="{name}"  total_runs={len(runs)}')
    for _, rid, st, rn in runs[:8]:
        print(f'  {rid}  {st:<10} {rn}')
    print()
EOF
} > /tmp/T3.txt 2>&1

# T4
{
  echo "=== T4.OUT.A: model artefacts on disk ==="
  ls -lh models/heart_pipeline.joblib models/mlflow_model/MLmodel models/mlflow_model/python_env.yaml models/mlflow_model/requirements.txt 2>&1 | awk '{print $5, $NF}'
  echo
  echo "=== T4.OUT.B: MLmodel header ==="
  head -25 models/mlflow_model/MLmodel
  echo
  echo "=== T4.OUT.C: requirements.txt (pinned packages) ==="
  N=$(grep -cv '^#\|^$' requirements.txt)
  echo "  pinned packages: $N"
  grep -v '^#\|^$' requirements.txt
} > /tmp/T4.txt 2>&1

# T5
{
  echo "=== T5.OUT.A: pytest one-line summary ==="
  $PY -m pytest unit_test/ -q --no-header --tb=no 2>&1 | tail -5
  echo
  echo "=== T5.OUT.B: GitHub Actions jobs in ci.yml ==="
  $PY -c "import yaml; d=yaml.safe_load(open('.github/workflows/ci.yml')); print('  jobs (in order): ' + ' -> '.join(d['jobs'].keys()))"
} > /tmp/T5.txt 2>&1

# T6
{
  echo "=== T6.OUT.A: heart-api image on disk ==="
  docker images heart-api --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' 2>&1 | head -5
  echo
  echo "=== T6.OUT.B: live /predict response (via Ingress) ==="
  curl -s -X POST http://localhost/predict \
    -H 'Content-Type: application/json' \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1,"ca_missing":0}' \
    | $PY -m json.tool 2>&1
} > /tmp/T6.txt 2>&1

# T7
{
  echo "=== T7.OUT.A: kubectl get pods,svc,ingress,hpa -l app=heart-api ==="
  kubectl get pods,svc,ingress,hpa -l app=heart-api 2>&1
  echo
  echo "=== T7.OUT.B: /health via Ingress (http://localhost) ==="
  curl -s -w "\nHTTP %{http_code}  latency=%{time_total}s\n" http://localhost/health 2>&1
} > /tmp/T7.txt 2>&1

# T8
{
  echo "=== T8.OUT.A: Prometheus targets ==="
  curl -s http://localhost:9090/api/v1/targets 2>/dev/null | $PY -c "
import sys,json
try:
    d=json.load(sys.stdin)
    for t in d['data']['activeTargets']:
        print(f\"  job={t['labels']['job']:<14} state={t['health']:<5} last_scrape={t.get('lastScrape','')[:19]}\")
except Exception:
    print('  (Prometheus not reachable on :9090)')
"
  echo
  echo "=== T8.OUT.B: /metrics endpoint (heart_api_* counters) ==="
  curl -s http://localhost/metrics | grep -E '^heart_api_(predictions_total|app_info|http_request_duration_seconds_count)' | head -10
  echo
  echo "=== T8.OUT.C: structured JSON access-log lines (last 3 /predict) ==="
  kubectl logs deploy/heart-api --tail=300 2>/dev/null | grep '"path"' | grep '/predict' | tail -3
} > /tmp/T8.txt 2>&1

# T9
{
  echo "=== T9.OUT: report deliverables on disk ==="
  ls -lh reports/REPORT.pdf reports/REPORT.docx reports/REPORT.html reports/REPORT.md 2>&1 | awk '{print $5, $NF}'
  echo
  echo "=== T9.OUT: pages in PDF (string-grep heuristic) ==="
  strings reports/REPORT.pdf | grep -c "/Type /Page$" || true
} > /tmp/T9.txt 2>&1

echo "DONE"
