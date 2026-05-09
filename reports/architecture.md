# Architecture — Heart Disease MLOps Pipeline

GitHub renders Mermaid blocks natively. Export to PNG/SVG with the
[Mermaid CLI](https://github.com/mermaid-js/mermaid-cli):

```bash
mmdc -i reports/architecture.md -o reports/figures/architecture.png \
     -t neutral -b transparent
```

## End-to-end flow

```mermaid
flowchart TD
    classDef stage fill:#dbe9ff,stroke:#1f4ea8,stroke-width:3px,color:#0a2440,font-weight:bold;
    classDef store fill:#fff1d6,stroke:#a35a00,stroke-width:3px,color:#3a1a00,font-weight:bold;
    classDef tool  fill:#ecdcfb,stroke:#5a2ea6,stroke-width:3px,color:#2b0e60,font-weight:bold;
    classDef ext   fill:#d6f5dc,stroke:#1a7f37,stroke-width:3px,color:#08381a,font-weight:bold;

    subgraph BUILD["BUILD"]
        direction LR
        DATA[Data Pipeline<br/>UCI &rarr; clean CSV]:::stage
        TRAIN[Training<br/>GridSearchCV<br/>3 candidates]:::stage
        REG[(Model Registry<br/>MLflow + joblib)]:::store
        DATA --> TRAIN --> REG
    end

    subgraph SERVE["SERVE"]
        direction LR
        APP[Flask API<br/>/predict /health /metrics]:::stage
        DOCK[(Container<br/>heart-api image)]:::store
        K8S[Kubernetes<br/>kind + ingress-nginx]:::stage
        APP --> DOCK --> K8S
    end

    subgraph OPERATE["OPERATE"]
        direction LR
        OBS[Observability<br/>Prometheus + Grafana]:::tool
        USER([Client / curl / browser]):::ext
    end

    REG --> APP
    K8S --> OBS
    USER --> K8S

    CI[CI/CD<br/>GitHub Actions]:::tool
    CI -.->|every push| TRAIN
    CI -.->|every push| DOCK
```
