"""Flask micro-service exposing the trained Heart Disease pipeline.

Endpoints
---------
GET  /health     -> liveness probe (always 200 once the model is loaded)
GET  /metadata   -> feature column contract + model artefact path
GET  /metrics    -> Prometheus exposition (default Flask metrics + custom)
POST /predict    -> body = single record (dict) **or** list of records
                    response = list of {prediction, probability, label}

Observability (Task 8)
----------------------
- ``prometheus_flask_exporter`` registers default Flask request counters and
  latency histograms at ``/metrics``.
- A custom counter ``heart_api_predictions_total{label="disease|no_disease"}``
  is incremented per predicted record.
- Every request emits one structured JSON access-log line on the
  ``heart_api.access`` logger (timestamp, method, path, status, latency_ms,
  remote_addr, n_records).

The pipeline is loaded once at import time so each worker pays the cost only
on cold start; subsequent requests are served from the in-memory model.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from flask import Flask, Response, g, jsonify, request
from prometheus_client import Counter
from prometheus_flask_exporter import PrometheusMetrics

from src.api.form import FORM_HTML
from src.features.build_features import FEATURE_COLS
from src.models.predict import load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "heart_pipeline.joblib"

log = logging.getLogger(__name__)
access_log = logging.getLogger("heart_api.access")

PREDICTIONS_TOTAL = Counter(
    "heart_api_predictions_total",
    "Total number of predictions returned, labelled by predicted class.",
    ["label"],
)


def create_app(model_path: str | Path | None = None) -> Flask:
    """Application factory; eases unit testing with a custom model path."""
    app = Flask(__name__)
    app.config["MODEL_PATH"] = str(
        model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH
    )

    log.info("Loading model from %s", app.config["MODEL_PATH"])
    app.config["MODEL"] = load_model(app.config["MODEL_PATH"])

    metrics = PrometheusMetrics(app, defaults_prefix="heart_api")
    try:
        metrics.info(
            "heart_api_app",
            "Heart Disease prediction API",
            version=os.environ.get("APP_VERSION", "1.0.0"),
        )
    except ValueError:
        # Already registered in the global Prometheus registry (e.g. when
        # create_app is invoked multiple times during testing).
        pass
    app.config["METRICS"] = metrics

    @app.before_request
    def _start_timer():
        g._t0 = time.perf_counter()

    @app.after_request
    def _emit_access_log(response):
        latency_ms = round((time.perf_counter() - getattr(g, "_t0", time.perf_counter())) * 1000, 2)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "method": request.method,
            "path": request.path,
            "status": response.status_code,
            "latency_ms": latency_ms,
            "remote_addr": request.remote_addr,
            "n_records": getattr(g, "n_records", None),
        }
        access_log.info(json.dumps(record))
        return response

    @app.get("/")
    def index():
        return Response(FORM_HTML, mimetype="text/html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_path": app.config["MODEL_PATH"]})

    @app.get("/metadata")
    def metadata():
        return jsonify({
            "feature_cols": list(FEATURE_COLS),
            "model_path": app.config["MODEL_PATH"],
            "output_schema": ["prediction", "probability", "label"],
        })

    @app.post("/predict")
    def predict_endpoint():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        try:
            payload = request.get_json()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Malformed JSON: {exc}"}), 400
        if payload is None:
            return jsonify({"error": "Empty request body"}), 400

        records = payload if isinstance(payload, list) else [payload]
        if not records or not all(isinstance(r, dict) for r in records):
            return jsonify({
                "error": "Body must be a JSON object or a list of JSON objects.",
            }), 400

        try:
            df_out = predict(app.config["MODEL"], records)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 422
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid input values: {exc}"}), 422

        for label in df_out["label"].tolist():
            PREDICTIONS_TOTAL.labels(label=label).inc()
        g.n_records = len(df_out)

        return jsonify({
            "n": len(df_out),
            "predictions": df_out.to_dict(orient="records"),
        })

    return app


app = create_app()


def main() -> int:
    """Local development entry point: ``python -m src.api.app``."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
