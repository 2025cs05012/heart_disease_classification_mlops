# Screenshot capture cookbook

The 7 mandatory screenshots referenced from `screenshots/README.md`,
`reports/REPORT.md`, and the assignment rubric — with the **exact
command and the moment to capture** for each one.

Capture either with `Cmd+Shift+4` (selection) or `Cmd+Shift+5` (full
window) on macOS. Save into this folder with the exact filenames
listed below — they are referenced verbatim from other documents.

---

## 1. `mlflow_ui.png` — Task 3 evidence

```bash
cd Assignment
./.venv/bin/mlflow ui --backend-store-uri ./mlruns
# → http://127.0.0.1:5000
```

In the UI: open experiment `heart_disease_classification`, click on the
parent `grid_search_<timestamp>` row to expand the three nested runs
(`logreg`, `random_forest`, `gradient_boosting`). Capture the row with
metrics columns visible (CV ROC-AUC, test_roc_auc, etc.).

## 2. `ci_run.png` — Task 5 evidence

After pushing the repository to GitHub, open
`https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml`
and pick a green run. Capture the **summary view** showing the three
jobs `lint ✓ → test ✓ → train ✓` in sequence, plus the artefacts panel
on the right with `coverage-report` and `heart-pipeline-<n>`.

## 3. `docker_build.png` — Task 6 evidence

```bash
cd Assignment
docker build -f docker/Dockerfile -t heart-api:latest .
docker images heart-api
```

Capture the terminal showing the final `Successfully tagged
heart-api:latest` line **and** the `docker images heart-api` output
underneath (image size column visible).

## 4. `kubectl_get_pods.png` — Task 7 evidence (pods)

```bash
kubectl get pods -l app=heart-api -o wide
```

Capture the table with both Pods in `Running 1/1`, AGE, IP, and NODE
columns.

## 5. `kubectl_get_svc.png` — Task 7 evidence (service + ingress + hpa)

```bash
kubectl get svc,ingress,hpa -l app=heart-api
```

Capture the combined output:

- `service/heart-api          NodePort      ...   80:30050/TCP`
- `ingress.networking.k8s.io/heart-api  nginx  *  localhost  80`
- `horizontalpodautoscaler.autoscaling/heart-api  Deployment/heart-api ...`

> **Optional bonus:** also capture
> `kubectl describe ingress heart-api | head -30` showing the
> `Backends: heart-api:80 (10.244.0.x:5000,...)` line — proves the
> Ingress is wired to live Pods.

## 6. `predict_curl.png` — Task 7 verification

```bash
curl -i http://localhost/health
echo
curl -i -X POST http://localhost/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
       "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,
       "ca":0,"thal":1,"ca_missing":0}'
```

Capture both responses (HTTP 200 + JSON body) in one screenshot — this
proves the **end-to-end Ingress → Service → Pod → model** path works.

## 7. Task 8 evidence — Monitoring & Logging (capture **all three**)

The rubric explicitly accepts *"Prometheus + Grafana **or** API
metrics/logs dashboard"*. We ship **both**, so capture all three of:

### 7a. `metrics_endpoint.png` — raw `/metrics` exposition

After `bash scripts/demo_up.sh` and a few `/predict` calls:

```bash
curl -s http://localhost/metrics | grep -E '^heart_api_' | head -20
```

Make sure these labels are visible in the screenshot:

- `heart_api_predictions_total{label="disease"} ...`
- `heart_api_predictions_total{label="no_disease"} ...`
- `heart_api_http_request_total{method="POST",path="/predict",status="200"} ...`
- `heart_api_app_info{version="1.0.0"} 1.0`

### 7b. `access_logs.png` — structured JSON access log

```bash
kubectl logs -f deploy/heart-api -c heart-api | grep '"path":"/predict"'
```

Trigger ~3 predictions in another terminal so a few lines appear, then
capture the terminal showing 3-5 JSON access-log lines (each must
contain `ts`, `method`, `path`, `status`, `latency_ms`, `n_records`).

### 7c. `grafana_dashboard.png` — Prometheus + Grafana dashboard

Bring up the bundled monitoring stack and open the pre-loaded dashboard:

```bash
# (heart-api must already be up via scripts/demo_up.sh)
docker compose -f monitoring/docker-compose.yml up -d
sleep 25                     # let Prometheus scrape twice

# generate ~50 requests so panels are non-empty
for i in $(seq 1 50); do
  curl -s -o /dev/null -X POST http://localhost/predict \
    -H 'Content-Type: application/json' \
    -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
         "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,
         "ca":0,"thal":1,"ca_missing":0}' &
done; wait

open http://localhost:3000/d/heart-api/heart-api?orgId=1&refresh=10s
```

In Grafana: log in as `admin` / `admin` (or use the anonymous viewer
mode that's already enabled), set the time range to **Last 15 minutes**,
let it auto-refresh once, and capture the full dashboard window —
all six panels populated:

1. Request rate by path (timeseries)
2. p95 latency by path (timeseries)
3. Error ratio (stat, %)
4. Predicted class distribution (donut: disease / no_disease)
5. Total predictions served (stat)
6. Request rate by status (timeseries)

Tear-down (after capturing): `docker compose -f monitoring/docker-compose.yml down -v`

---

## Optional polish screenshots

| File | Capture |
|---|---|
| `mlflow_run_detail.png`  | Open any single nested run → Artifacts tab → expand `figures/` and click `roc_curves.png`. Capture the detail pane. |
| `kubectl_rollout.png`    | After `docker build -t heart-api:v2 . && kind load docker-image heart-api:v2 --name heart && kubectl set image deployment/heart-api heart-api=heart-api:v2`, capture `kubectl rollout status` showing zero-downtime progression. |
| `prometheus_targets.png` | `open http://localhost:9090/targets` — capture the **heart-api** target with state **UP**. |

PNG @ 1280×800 or larger keeps text legible after the report PDF export.
