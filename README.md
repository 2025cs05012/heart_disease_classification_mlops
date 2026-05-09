# MLOps Assignment-I — Heart Disease Classification

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml)

End-to-end MLOps pipeline for the **UCI Heart Disease** dataset: data acquisition → cleaning → EDA → modelling → MLflow tracking → packaging → CI/CD → Docker → Kubernetes (kind + Ingress) → monitoring.

**Stack:** Python 3.11, scikit-learn, MLflow, Flask, GitHub Actions, Docker, Kubernetes (`kind` + `ingress-nginx`), Prometheus.

📄 **Final report:** [`reports/REPORT.md`](reports/REPORT.md) (also exported as `reports/REPORT.pdf` / `reports/REPORT.docx`)  ·  🗺️ **Architecture:** [`reports/architecture.md`](reports/architecture.md) (rendered → `reports/figures/architecture.png`)  ·  🖼️ **Screenshots checklist:** [`screenshots/README.md`](screenshots/README.md)

## How to access the deployed API (Deliverable c)

This solution is deployed to a **local Kubernetes cluster (`kind`)** as
permitted by the rubric (*"GKE, EKS, AKS, or Minikube/Docker Desktop"*).
The API is exposed via an `ingress-nginx` Ingress on plain HTTP port 80
of the host running the cluster:

| What | Endpoint |
|---|---|
| Health | `GET  http://localhost/health` |
| Metadata | `GET  http://localhost/metadata` |
| Prediction | `POST http://localhost/predict`  (JSON body, see Task 6) |
| Prometheus metrics | `GET  http://localhost/metrics` |

> No public cloud URL is provided — the assignment explicitly allows
> *"local Kubernetes (… Minikube/Docker Desktop)"* and the
> deliverables list says *"Deployed API URL (if public) **or access
> instructions (for local testing)**"*.

### One-command bring-up (what to run after cloning)

Prerequisites: Docker (Engine or Desktop) running, plus `kind` and
`kubectl` on PATH (`brew install kind kubectl`).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>          # or cd <your-repo>/Assignment if pushed as a sub-folder

bash scripts/demo_up.sh
```

`demo_up.sh` is idempotent and does the full bring-up in 6 steps —
build the image, create the kind cluster, sideload the image, install
`ingress-nginx`, apply the manifests, and run a smoke test against
`/health` and `/predict`. Total time on a cold machine ≈ 5 min, on a
warm machine ≈ 90 s.

After it succeeds, the API is reachable on `http://localhost/`:

```bash
curl -s http://localhost/health | jq .
# → {"model_path":"/app/models/heart_pipeline.joblib","status":"ok"}

curl -s -X POST http://localhost/predict \
     -H 'Content-Type: application/json' \
     -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
          "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,
          "ca":0,"thal":1,"ca_missing":0}' | jq .
# → {"n":1,"predictions":[{"label":"disease","prediction":1,"probability":0.55}]}

curl -s http://localhost/metrics | grep -E '^heart_api_' | head -5
```

### Tear-down

```bash
bash scripts/demo_down.sh           # remove just the heart-api workload
bash scripts/demo_down.sh --full    # also delete the kind cluster
```

Manual step-by-step instructions for the same flow (useful for
troubleshooting or for capturing screenshots) are in
[Task 7 below](#task-7--production-deployment-local-kubernetes-via-kind--ingress).

---

## Project layout

```
Assignment/
├── data/
│   ├── raw/heart+disease/             UCI source files (4 processed.*.data subsets)
│   └── processed/heart_disease_clean.csv
├── src/
│   ├── data/
│   │   ├── download.py                Idempotent UCI fetch
│   │   └── preprocess.py              Cleaning + binary target + missingness flag
│   ├── models/                        (Task 2 — training)
│   ├── api/                           (Task 6 — Flask /predict)
│   └── utils/
├── notebooks/01_eda.ipynb             Exploratory data analysis
├── reports/figures/                   EDA plots (PNG)
├── unit_test/                         pytest suite (Task 5)
├── docker/                            Dockerfile (Task 6)
├── k8s/                               Kubernetes manifests (Task 7)
├── .github/workflows/                 CI/CD (Task 5)
├── requirements.txt
└── Mlops_Assignment1.pdf              Assignment brief
```

---

## Setup

```bash
cd Assignment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If `python` is shadowed by a shell alias, invoke the venv binary directly:
> `./.venv/bin/python -m src.data.preprocess`

---

## Reproducing the pipeline

```bash
# 1. (Optional) re-download the raw dataset from UCI
./.venv/bin/python -m src.data.download

# 2. Build the cleaned CSV (data/processed/heart_disease_clean.csv)
./.venv/bin/python -m src.data.preprocess

# 3. Run the EDA notebook end-to-end (figures -> reports/figures/)
./.venv/bin/jupyter nbconvert --to notebook --execute \
    notebooks/01_eda.ipynb --output 01_eda.ipynb

# 4. Train + evaluate models (writes models/heart_pipeline.joblib + reports/metrics.json
#    and logs the run to mlruns/ — pass --no-mlflow to disable tracking)
./.venv/bin/python -m src.models.train

# 5. Score new records using the saved model (CSV or JSON in -> CSV out)
./.venv/bin/python -m src.models.predict --input data/processed/heart_disease_clean.csv \
    --output reports/predictions.csv

# 6. Browse MLflow runs in the UI (http://127.0.0.1:5000)
./.venv/bin/mlflow ui --backend-store-uri ./mlruns

# 7. Run the test suite
./.venv/bin/pytest unit_test/ -v
```

---

## Task 1 — Data Acquisition & EDA (summary)

**Source:** UCI ML Repository, *Heart Disease Data Set* (id 45) — Cleveland, Hungarian, Switzerland and Long-Beach VA subsets, 14 attributes per record.

**Volume:** 920 patient records (303 + 294 + 123 + 200) × 17 columns after engineering.

**Cleaning rules applied** (`src/data/preprocess.py`):
- `?` tokens → `NaN`.
- Sentinel zeros in `chol` and `trestbps` → `NaN` (medically impossible). 173 records affected, mostly from the Switzerland subset.
- Integer-valued categoricals (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`, `num`) cast to nullable `Int64`.
- Continuous numerics (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) cast to `float`.
- **Binary target** added: `target = (num > 0).astype(int)`.
- **`ca_missing`** indicator added so the model can learn from the missingness pattern of `ca` (66 % NA).

**Class balance:** `target=1` 55.3 % · `target=0` 44.7 % → balanced; no resampling required.

**Missingness (post-cleanup):**

| Feature | % missing | Strategy (Task 2) |
|---|---:|---|
| `ca` | 66.4 % | median impute + `ca_missing` flag |
| `thal` | 52.8 % | most-frequent impute |
| `slope` | 33.6 % | most-frequent impute |
| `chol` | 22.0 % | median impute |
| `fbs` | 9.8 % | most-frequent impute |
| `oldpeak`, `trestbps`, `thalach`, `exang` | 6 – 7 % | median / most-frequent |
| `age`, `sex`, `cp`, `restecg`, `num` | ≤ 0.2 % | — |

**Top features by Spearman correlation with `target`:**

| Rank | Feature | ρ | Interpretation |
|---:|---|---:|---|
| 1 | `cp` | +0.51 | asymptomatic chest-pain type ↑ disease |
| 2 | `ca` | +0.48 | more vessels coloured ↑ disease |
| 3 | `thal` | +0.48 | reversible defect ↑ disease |
| 4 | `exang` | +0.46 | exercise-induced angina ↑ disease |
| 5 | `oldpeak` | +0.40 | larger ST depression ↑ disease |
| 6 | `thalach` | −0.40 | lower max HR ↑ disease |
| 7 | `slope` | +0.35 | flat / down slope ↑ disease |
| 8 | `sex` | +0.31 | male ↑ disease |
| 9 | `age` | +0.29 | older ↑ disease |

**Generated figures** (`reports/figures/`):
`class_balance.png`, `missingness_overall.png`, `missingness_per_source.png`, `histograms_by_target.png`, `disease_rate_by_category.png`, `correlation_heatmap.png`, `boxplots_numeric.png`.

---

## Task 2 — Feature Engineering & Model Development (summary)

**Feature pipeline** (`src/features/build_features.py`) is a single
`ColumnTransformer` reused at training **and** inference time:

| Group | Columns | Transform |
|---|---|---|
| Numeric | `age, trestbps, chol, thalach, oldpeak, ca` | `SimpleImputer(median)` → `StandardScaler` |
| Categorical | `sex, cp, fbs, restecg, exang, slope, thal` | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |
| Passthrough | `ca_missing` | as-is |
| Dropped | `source, num` | leakage / metadata |

Output: 26-column dense matrix.

**Models** (`src/models/train.py`): Logistic Regression, Random Forest, Gradient
Boosting — each wrapped in a single `Pipeline(preprocessor + estimator)` and
tuned with 5-fold stratified `GridSearchCV` (`scoring='roc_auc'`) on an 80 / 20
stratified train-test split (`random_state=42`).

**Held-out test results (n = 184):**

| Model | CV ROC-AUC | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.886 | 0.794 | 0.796 | 0.843 | 0.819 | 0.897 |
| **Random Forest** *(best)* | 0.882 | 0.804 | 0.800 | 0.863 | 0.830 | **0.914** |
| Gradient Boosting | 0.871 | 0.821 | 0.805 | 0.892 | 0.847 | 0.910 |

Best Random Forest hyper-parameters: `n_estimators=200, max_depth=8, min_samples_split=2`.

**Artefacts produced**
- `models/heart_pipeline.joblib` — full preprocessor + model (used by Task 4 / 6)
- `reports/metrics.json` — per-model CV + test metrics
- `reports/figures/roc_curves.png`, `reports/figures/confusion_matrix.png`

**Tests** (`unit_test/`) — 12 pytest cases, run in ~9 s:
- `test_preprocess.py` — sentinel-zero handling, target binary/balanced, `ca_missing` flag matches `ca.isna()`, CSV roundtrip.
- `test_build_features.py` — `split_xy` shape, transformer output 920×26 with no NaN, unknown-category robustness.
- `test_train.py` — persisted pipeline `predict_proba`, metrics-summary schema, end-to-end smoke run on a tiny grid.

---

## Task 3 — Experiment Tracking with MLflow (summary)

**Backend:** local file-store at `Assignment/mlruns/` (no external service needed).
Override with `MLFLOW_TRACKING_URI` or `--mlflow-tracking-uri` for a remote server.

**Experiment:** `heart_disease_classification` (helpers in `src/utils/mlflow_utils.py`).

**Run hierarchy** — every `python -m src.models.train` invocation creates:

- **Parent run** `grid_search_<YYYYMMDD_HHMMSS>` — logs the global session
  parameters (`cv`, `random_state`, `test_size`, `best_model`), the winning
  model's test metrics (`best_accuracy`, `best_precision`, `best_recall`,
  `best_f1`, `best_roc_auc`), the artefacts (`model/`, `figures/roc_curves.png`,
  `figures/confusion_matrix.png`, `metrics.json`) and the serialised sklearn
  pipeline via `mlflow.sklearn.log_model`.
- **Nested runs** (one per candidate: `logreg`, `random_forest`,
  `gradient_boosting`) — log the best hyper-parameters from `GridSearchCV`,
  the CV ROC-AUC mean (`cv_roc_auc_mean`) and per-test metrics
  (`test_accuracy`, …, `test_roc_auc`).

**CLI flags** (defaults shown):

| Flag | Default | Meaning |
|---|---|---|
| `--mlflow / --no-mlflow` | `--mlflow` | Toggle MLflow tracking |
| `--mlflow-tracking-uri URI` | `file://Assignment/mlruns` | Override tracking backend |

**Browse runs:**
```bash
./.venv/bin/mlflow ui --backend-store-uri ./mlruns   # → http://127.0.0.1:5000
```

**Tests** (`unit_test/test_mlflow.py`) verify, against a temporary tracking URI:
experiment creation, exactly one parent + one nested run with the correct
parent-child link, that all expected params/metrics are persisted, and that
`model/`, `figures/`, and `metrics.json` artefacts are uploaded to the parent run.

---

## Task 4 — Model Packaging & Reproducibility (summary)

Every successful `python -m src.models.train` produces **two** locally
re-loadable artefacts of the same fitted pipeline:

| Artefact | Path | Loader | Use case |
|---|---|---|---|
| Joblib pickle | `models/heart_pipeline.joblib` | `joblib.load` | Fast in-process loading (Flask API, tests) |
| MLflow model | `models/mlflow_model/` | `mlflow.pyfunc.load_model` | Portable, signed schema, `mlflow models serve` |

The MLflow flavour bundles `MLmodel`, `model.pkl`, `conda.yaml`,
`python_env.yaml`, `requirements.txt`, plus an `input_example.json` and an
inferred input/output **signature** so consumers can validate payloads.

**Score new records** (CSV or JSON in, predictions out):

```bash
# CSV input -> stdout
./.venv/bin/python -m src.models.predict --input data/processed/heart_disease_clean.csv

# JSON input -> file
./.venv/bin/python -m src.models.predict --input sample.json --output preds.csv

# Use the portable MLflow model instead of the joblib
./.venv/bin/python -m src.models.predict --model models/mlflow_model --input sample.csv
```

The output schema is always `prediction, probability, label` where
`label ∈ {disease, no_disease}` and `probability` is the positive-class
confidence (degenerates to 0/1 when scoring through `mlflow.pyfunc`, which
exposes `predict` only).

**Public Python API** (used by tests today, by the Flask `/predict` endpoint
in Task 6 tomorrow):

```python
from src.models.predict import load_model, predict

model = load_model("models/heart_pipeline.joblib")     # or models/mlflow_model
predictions = predict(model, [{"age": 63, "sex": 1, ...}])
```

**Reproducibility guarantees:**

- Python interpreter pinned via `Assignment/.python-version` → **3.11**.
- All 16 runtime dependencies pinned with exact `==` versions in
  `requirements.txt` (numpy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2,
  mlflow 2.13.2, flask 3.0.3, joblib 1.4.2, …).
- The MLflow model directory carries its own self-describing
  `conda.yaml` + `python_env.yaml` so it can be deserialised on a fresh host
  with `mlflow models serve -m models/mlflow_model`.

**Tests** (`unit_test/test_predict.py`, 9 cases): load from joblib, load from
MLflow dir, missing-path error, scoring shape + columns, list-of-dicts input,
schema validation (missing column raises `KeyError`), label/prediction
consistency, CLI CSV round-trip, CLI JSON input.

---

## Task 5 — CI/CD & Automated Testing (summary)

GitHub Actions workflow at `.github/workflows/ci.yml` runs on every push,
pull request to `main` / `master`, and on manual `workflow_dispatch`. The
pipeline is staged so a failure short-circuits later jobs:

| Job | Runs | What it does |
|---|---|---|
| **lint** | always | `ruff check src unit_test` (rules `E`, `W`, `F`, `I`; cfg in `pyproject.toml`) |
| **test** | after lint | `pip install -r requirements.txt`, build cleaned CSV via `python -m src.data.preprocess`, then `pytest unit_test/ --cov=src --cov-report=xml --junitxml=pytest-report.xml`. Uploads `coverage.xml` + `pytest-report.xml` as the *coverage-report* artefact. |
| **train** | after test | Re-trains all 3 candidates with `python -m src.models.train --no-mlflow` and uploads `models/heart_pipeline.joblib`, `models/mlflow_model/`, `reports/metrics.json`, and the ROC + confusion-matrix figures as the *heart-pipeline-{run}* artefact (14-day retention). |

**Local equivalent of what CI runs:**

```bash
./.venv/bin/pip install -r requirements.txt
./.venv/bin/ruff check src unit_test
./.venv/bin/python -m src.data.preprocess
./.venv/bin/pytest unit_test/ --cov=src --cov-report=term --cov-report=xml \
    --junitxml=pytest-report.xml
./.venv/bin/python -m src.models.train --no-mlflow
```

**Latest local run:** 23 / 23 tests pass, **74 % line coverage**
(`mlflow_utils 100 % · predict 95 % · train 87 % · preprocess 84 % ·
build_features 71 % · download 0 %` — `download.py` is excluded as it makes a
live UCI HTTP request).

**Activating the badge:** the workflow file uses `Assignment/` as the repo
root. Either submit the `Assignment/` folder as its own GitHub repository or
move `.github/` to the outer repository's root, then replace
`<your-username>/<your-repo>` in the badge URL above with the real path.

---

## Task 6 — Containerisation (summary)

The trained model is served by a small Flask app
(`src/api/app.py`) packaged into a slim Docker image
(`docker/Dockerfile`). The container loads
`models/heart_pipeline.joblib` once at start-up and serves predictions
via gunicorn (2 workers).

### API contract

| Method | Path | Body | Success | Failure |
|---|---|---|---|---|
| GET  | `/health`   | — | `200 {"status":"ok","model_path":"…"}` | — |
| GET  | `/metadata` | — | `200 {"feature_cols":[…],"output_schema":[…]}` | — |
| POST | `/predict`  | JSON object **or** list of objects | `200 {"n":N,"predictions":[{"prediction":0\|1,"probability":<float>,"label":"disease"\|"no_disease"},…]}` | `400` non-JSON / empty body · `422` missing feature column / invalid value |

Required feature keys (any payload must contain all 14):
`age, trestbps, chol, thalach, oldpeak, ca, sex, cp, fbs, restecg, exang,
slope, thal, ca_missing`.

### Run locally — without Docker

```bash
# Flask development server (single process, debug off)
./.venv/bin/python -m src.api.app          # → http://127.0.0.1:5000

# Gunicorn (matches the container exactly)
./.venv/bin/gunicorn --bind 127.0.0.1:5000 --workers 2 src.api.app:app
```

### Build & run the container

```bash
cd Assignment

# Build (run from Assignment/ so requirements.txt + src/ + models/ are in the
# build context). Final image is ~600 MB on python:3.11-slim.
docker build -f docker/Dockerfile -t heart-api:latest .

# Run (publishes port 5000 on the host)
docker run --rm -p 5000:5000 --name heart-api heart-api:latest

# Verify
curl -s http://localhost:5000/health | jq .
curl -s http://localhost:5000/metadata | jq .

curl -s -X POST http://localhost:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
          "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,
          "ca":0,"thal":6,"ca_missing":0}' | jq .
# → {"n":1,"predictions":[{"prediction":0,"probability":0.30,"label":"no_disease"}]}
```

### Image hygiene

- Base: `python:3.11-slim` + `libgomp1` (numpy/scipy runtime) + `curl` (HEALTHCHECK).
- `requirements.txt` is copied **before** `src/` so layer caching survives code edits.
- Runs as non-root `appuser` (uid 1000).
- `HEALTHCHECK` hits `/health` every 30 s with a 15 s start-up grace.
- `.dockerignore` excludes `.venv`, `data/`, `notebooks/`, `unit_test/`, `mlruns/`,
  `models/mlflow_model/`, and CI scratch files so the build context stays small.

### Tests

`unit_test/test_api.py` (9 cases, run in ~2 s) exercises the Flask app via the
test client without spinning up gunicorn or Docker:

- `/health` → 200 with `status: ok`
- `/metadata` → exposes the exact 14 feature columns + output schema
- `/predict` accepts a single dict and a list of dicts
- `/predict` returns `422` when required columns are missing
- `/predict` returns `400` for non-JSON bodies, empty bodies, or non-object payloads
- `label` field always agrees with the `prediction` integer

---

## Task 7 — Production Deployment (local Kubernetes via `kind` + Ingress)

The image built in Task 6 is deployed to a local **`kind`** (Kubernetes
in Docker) cluster and exposed through an **`ingress-nginx`** Ingress.
Manifests live in `k8s/`:

| File | Kind | Purpose |
|---|---|---|
| `k8s/configmap.yaml`  | ConfigMap | `MODEL_PATH`, `HOST`, `PORT` env vars |
| `k8s/deployment.yaml` | Deployment | 2 replicas, rolling update, startup/readiness/liveness probes on `/health`, CPU+memory requests/limits, non-root securityContext, `imagePullPolicy: Never` (image is sideloaded into the kind node) |
| `k8s/service.yaml`    | Service (NodePort 30050) | ClusterIP entry that the Ingress targets; NodePort kept as a fallback for direct access |
| `k8s/ingress.yaml`    | Ingress (`ingressClassName: nginx`) | Routes `http://localhost/*` to `heart-api:80` — **this is the public exposure layer** |
| `k8s/hpa.yaml`        | HorizontalPodAutoscaler | Scales 2 → 5 replicas at 70 % CPU |
| `k8s/servicemonitor.yaml` | ServiceMonitor | Optional Prometheus Operator scrape config |

The cluster bootstrap config is committed at
`k8s/setup/kind-cluster-config.yaml` — it maps host ports `80`, `443`
and `30050` to the control-plane node and labels it
`ingress-ready=true` so the kind variant of `ingress-nginx` schedules
on it.

### Prerequisites

1. Docker (Engine or Desktop) running.
2. `kind` and `kubectl` on PATH:

   ```bash
   brew install kind kubectl
   ```

3. The `heart-api:latest` image must be built locally (it gets
   sideloaded into the kind node — no registry needed):

   ```bash
   docker build -f docker/Dockerfile -t heart-api:latest .
   ```

### One-time cluster setup

```bash
cd Assignment

# A. Create the kind cluster with port mappings + ingress-ready label
kind create cluster --config k8s/setup/kind-cluster-config.yaml

# B. Sideload the image into the kind node
kind load docker-image heart-api:latest --name heart

# C. Install ingress-nginx (kind variant) and wait for the controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=180s
```

### Deploy the workload

```bash
kubectl config use-context kind-heart            # confirm context
kubectl apply -f k8s/configmap.yaml \
              -f k8s/deployment.yaml \
              -f k8s/service.yaml \
              -f k8s/hpa.yaml \
              -f k8s/ingress.yaml

kubectl rollout status deployment/heart-api      # wait until 2/2 ready
kubectl get pods,svc,ingress,hpa -l app=heart-api
```

Expected output (used for the Task 7 screenshots):

```
pod/heart-api-xxx-aaa            1/1   Running
pod/heart-api-xxx-bbb            1/1   Running
service/heart-api                 NodePort   10.96.x.x   <none>      80:30050/TCP
ingress.networking.k8s.io/heart-api  nginx  *  localhost   80
horizontalpodautoscaler.autoscaling/heart-api ...
```

### Verify (via Ingress on plain HTTP port 80 — no NodePort needed)

```bash
curl -s http://localhost/health   | jq .
curl -s http://localhost/metadata | jq .

curl -s -X POST http://localhost/predict \
     -H 'Content-Type: application/json' \
     -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,
          "restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,
          "ca":0,"thal":6,"ca_missing":0}' | jq .
```

Alternative paths (kept for completeness):

```bash
# NodePort — direct hit, bypasses the Ingress
curl http://localhost:30050/health

# Port-forward — works regardless of Ingress / NodePort
kubectl port-forward svc/heart-api 8080:80
curl http://localhost:8080/health
```

### Rolling update demo

```bash
docker build -f docker/Dockerfile -t heart-api:v2 .
kind load docker-image heart-api:v2 --name heart    # sideload the new tag
kubectl set image deployment/heart-api heart-api=heart-api:v2
kubectl rollout status deployment/heart-api
kubectl rollout undo  deployment/heart-api          # rollback if needed
```

### Teardown

```bash
# Just remove the workload (cluster + ingress-nginx stay)
kubectl delete -f k8s/ingress.yaml -f k8s/hpa.yaml \
               -f k8s/service.yaml -f k8s/deployment.yaml \
               -f k8s/configmap.yaml

# Or destroy the cluster entirely
kind delete cluster --name heart
```

### Notes

- The startupProbe (12 × 5 s) gives the gunicorn workers up to a minute to
  load the joblib pipeline before the liveness probe takes over.
- The HPA requires the **metrics-server**. kind does not ship one by default;
  install with
  `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
  and patch with `--kubelet-insecure-tls`.
- The Service is kept as `NodePort` so that both the Ingress
  (`localhost:80`) **and** a direct port (`localhost:30050`) are
  available — the rubric explicitly accepts either Load Balancer or
  Ingress; we deliver Ingress as the primary exposure.

---

## Task 8 — Monitoring & Logging

The Flask app emits two complementary observability streams:

### 1. Structured JSON access log

Every request produces one line on the `heart_api.access` logger
(stdout in the container, captured by `kubectl logs`):

```json
{"ts":"2026-05-07T15:23:01Z","method":"POST","path":"/predict",
 "status":200,"latency_ms":4.81,"remote_addr":"10.1.0.42","n_records":3}
```

Tail it from a running Pod:

```bash
kubectl logs -f deploy/heart-api -c heart-api | grep '"path":"/predict"'
```

### 2. Prometheus metrics — `GET /metrics`

`prometheus-flask-exporter` registers default request counters and latency
histograms (prefixed `heart_api_*`); a custom counter is added for
predicted-class volume:

| Metric | Type | Labels | Description |
|---|---|---|---|
| `heart_api_http_request_total`         | Counter   | `method, status, path` | Total HTTP requests |
| `heart_api_http_request_duration_seconds` | Histogram | `method, status, path` | Request latency (seconds) |
| `heart_api_predictions_total`          | Counter   | `label`                | Predictions returned per class (`disease` / `no_disease`) |
| `heart_api_app_info`                   | Gauge     | `version`              | Build info |

Quick local verification:

```bash
curl -s http://localhost:30050/metrics | grep -E '^heart_api_(predictions|http_request_total)'
# heart_api_predictions_total{label="no_disease"} 17.0
# heart_api_predictions_total{label="disease"}    11.0
# heart_api_http_request_total{method="POST",path="/predict",status="200"} 12.0
```

### Prometheus scraping

Two paths are supported out of the box:

1. **Annotation-based discovery** — `deployment.yaml` already declares
   ```yaml
   prometheus.io/scrape: "true"
   prometheus.io/port:   "5000"
   prometheus.io/path:   "/metrics"
   ```
   so any vanilla Prometheus configured with the standard
   `kubernetes_sd_configs.role: pod` job picks the Pods up automatically.

2. **Prometheus Operator** — apply the bundled
   `k8s/servicemonitor.yaml` (CRD `ServiceMonitor`); adjust the
   `release:` label to match your Operator release.

If you don't have a Prometheus deployed yet, scrape locally:

```bash
kubectl port-forward svc/heart-api 8080:80
curl http://localhost:8080/metrics
```

### Grafana dashboard (bundled stack)

A self-contained Prometheus + Grafana stack with a pre-loaded
**Heart API** dashboard ships under `monitoring/`. Bring it up with one
command (after `scripts/demo_up.sh` has the API live on `localhost:80`):

```bash
docker compose -f monitoring/docker-compose.yml up -d
# Prometheus  http://localhost:9090   (Status → Targets: heart-api UP)
# Grafana     http://localhost:3000   (anonymous Viewer; admin/admin to edit)
#             → Dashboards → "Heart API"
```

The dashboard renders six panels driven by the PromQL we publish:

| Panel | PromQL |
|---|---|
| Request rate by path           | `sum(rate(heart_api_http_request_total[1m])) by (path)` |
| p95 latency by path            | `histogram_quantile(0.95, sum(rate(heart_api_http_request_duration_seconds_bucket[5m])) by (le, path))` |
| Error ratio (4xx/5xx)          | `sum(rate(heart_api_http_request_total{status=~"4..\|5.."}[5m])) / sum(rate(heart_api_http_request_total[5m]))` |
| Predicted class distribution   | `sum by (label) (heart_api_predictions_total)` |
| Total predictions served       | `sum(heart_api_predictions_total)` |
| Request rate by status         | `sum(rate(heart_api_http_request_total[1m])) by (status)` |

All assets are committed to the repo (no external Helm chart needed):

```
monitoring/
├── docker-compose.yml
├── prometheus/prometheus.yml
└── grafana/
    ├── provisioning/datasources/prometheus.yml
    ├── provisioning/dashboards/dashboards.yml
    └── dashboards/heart_api.json
```

Tear-down: `docker compose -f monitoring/docker-compose.yml down -v`.

### Tests

`unit_test/test_api.py` covers the monitoring surface (3 cases on top of the
9 contract tests):

- `/metrics` returns 200 with `text/plain` and contains
  `heart_api_predictions_total`.
- The custom counter increments by **N** after a `/predict` call with
  N records.
- Each request emits one JSON access-log line containing `ts`,
  `method`, `path`, `status`, `latency_ms`, and `n_records`.

---

## Roadmap

| Task | Marks | Status |
|---|---:|---|
| 1. Data Acquisition & EDA | 5 | ✅ |
| 2. Feature Engineering & Model Development | 8 | ✅ |
| 3. Experiment Tracking with MLflow | 5 | ✅ |
| 4. Model Packaging & Reproducibility | 7 | ✅ |
| 5. CI/CD & Automated Testing | 8 | ✅ |
| 6. Containerisation (Flask + Docker) | 5 | ✅ |
| 7. Production Deployment (Docker Desktop K8s) | 7 | ✅ |
| 8. Monitoring & Logging | 3 | ✅ |
| 9. Documentation & Reporting | 2 | ✅ |
| **Total** | **50** | **✅** |
