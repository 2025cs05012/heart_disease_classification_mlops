# Screenshots — submission checklist

Drop the following PNGs into this folder before submitting. The
filenames are referenced verbatim from `reports/REPORT.md` and
`README.md`, so keep them stable.

| File | What to capture | Source command / view |
|---|---|---|
| `mlflow_ui.png`         | MLflow UI showing the parent run (`grid_search_…`) expanded with three nested runs and their metrics. | `mlflow ui --backend-store-uri file://$PWD/mlruns` → http://localhost:5000 |
| `ci_run.png`            | A green GitHub Actions run for `ci.yml` — lint ✓ test ✓ train ✓, with the artefacts list visible. | https://github.com/`<your-username>`/`<your-repo>`/actions |
| `docker_build.png`      | Terminal showing `docker build -f docker/Dockerfile -t heart-api:latest .` succeeding, with the final image size. | local terminal |
| `kubectl_get_pods.png`  | `kubectl get pods -l app=heart-api` showing both replicas in `Running 1/1`. | `kubectl get pods -l app=heart-api -o wide` |
| `kubectl_get_svc.png`   | `kubectl get svc heart-api` showing `NodePort 80:30050/TCP`. | `kubectl get svc,hpa -l app=heart-api` |
| `predict_curl.png`      | Terminal output of a successful `POST /predict` against the NodePort, with the JSON response. | `curl -s -X POST http://localhost:30050/predict -H 'Content-Type: application/json' -d @sample.json` |
| `grafana_dashboard.png` | Grafana dashboard with the four suggested panels (request rate, p95 latency, error ratio, class distribution). | Grafana → Heart API dashboard |

Optional but worth ~½ mark of polish:

| File | What to capture |
|---|---|
| `mlflow_run_detail.png`   | Single run detail view showing `roc_curves.png` + `confusion_matrix.png` artefacts. |
| `kubectl_rollout.png`     | `kubectl rollout status deployment/heart-api` after a `set image` to a new tag, demonstrating zero-downtime update. |
| `prometheus_targets.png`  | Prometheus → Status → Targets with the `heart-api` Pods marked UP. |

PNG is preferred (lossless, small for terminal captures); 1280×800 or
larger keeps the text legible in the report PDF export.
