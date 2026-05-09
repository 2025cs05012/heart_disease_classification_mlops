# Architecture — Heart Disease MLOps Pipeline

GitHub renders Mermaid blocks natively. Export to PNG/SVG with the
[Mermaid CLI](https://github.com/mermaid-js/mermaid-cli):

```bash
mmdc -i reports/architecture.md -o reports/figures/architecture.png \
     -t neutral -b transparent
```

## End-to-end flow

```mermaid
flowchart LR
    %% --- Data layer ---
    subgraph DATA["Data layer"]
        UCI[UCI Heart Disease<br/>4 .data files]
        RAW[(data/raw/)]
        CLEAN[(data/processed/<br/>heart_disease_clean.csv)]
        UCI -->|src/data/download.py| RAW
        RAW -->|src/data/preprocess.py| CLEAN
    end

    %% --- Modelling layer ---
    subgraph MODEL["Modelling & tracking"]
        TRAIN[src/models/train.py<br/>GridSearchCV cv=5]
        MLFLOW[(mlruns/<br/>parent + 3 nested)]
        JOBLIB[(models/<br/>heart_pipeline.joblib)]
        MLM[(models/<br/>mlflow_model/)]
        CLEAN --> TRAIN
        TRAIN --> MLFLOW
        TRAIN --> JOBLIB
        TRAIN --> MLM
    end

    %% --- Serving layer ---
    subgraph SERVE["Serving"]
        PRED[src/models/predict.py<br/>load_model + predict]
        FLASK[src/api/app.py<br/>Flask + gunicorn]
        IMG[(heart-api:latest<br/>docker/Dockerfile)]
        JOBLIB --> PRED
        MLM --> PRED
        PRED --> FLASK
        FLASK --> IMG
    end

    %% --- K8s layer ---
    subgraph K8S["Kubernetes (kind)"]
        ING[Ingress<br/>ingress-nginx<br/>http://localhost/*]
        DEP[Deployment<br/>2 replicas + RollingUpdate]
        SVC[Service<br/>NodePort 30050]
        HPA[HPA<br/>2..5 replicas @ 70% CPU]
        IMG -->|kind load docker-image| DEP
        ING --> SVC
        SVC --> DEP
        DEP --- HPA
    end

    %% --- Observability layer ---
    subgraph OBS["Observability"]
        METRICS[/metrics<br/>Prometheus exposition/]
        LOGS[(stdout JSON access log)]
        PROM[Prometheus]
        GRAF[Grafana]
        DEP --> METRICS
        DEP --> LOGS
        METRICS --> PROM --> GRAF
    end

    %% --- CI/CD layer ---
    subgraph CI["CI/CD"]
        GHA[GitHub Actions<br/>lint -> test -> train]
        ARTS[(metrics.json<br/>figures + joblib<br/>artefacts)]
        GHA --> ARTS
    end

    CI -.->|every push / PR| TRAIN
    CI -.->|every push / PR| FLASK

    %% --- Client ---
    USER([Client / curl])
    USER -->|POST /predict| ING
    USER -->|GET /health| ING
```

## Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant I as Ingress<br/>(ingress-nginx)
    participant K as K8s Service<br/>(heart-api)
    participant P as Pod<br/>(gunicorn worker)
    participant M as Model<br/>(joblib pipeline)
    participant Pr as Prometheus

    C->>I: POST http://localhost/predict {"age":63,...}
    I->>K: forward to heart-api Service
    K->>P: route to ready Pod
    P->>P: before_request → start timer
    P->>P: validate JSON + feature columns
    P->>M: predict(records)
    M-->>P: prediction, probability
    P->>P: increment heart_api_predictions_total{label}
    P->>P: after_request → JSON access log
    P-->>K: 200 {n, predictions[]}
    K-->>I: 200 {n, predictions[]}
    I-->>C: 200 {n, predictions[]}
    Pr-->>P: GET /metrics  (every 30s)
    P-->>Pr: heart_api_* counters & histograms
```
