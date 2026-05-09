"""Tests for ``src.api.app`` — Flask service serving the trained pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.features.build_features import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "heart_pipeline.joblib"

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL.is_file(),
    reason="trained pipeline missing; run `python -m src.models.train` first.",
)


@pytest.fixture(scope="module")
def client():
    from src.api.app import create_app

    app = create_app(DEFAULT_MODEL)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def sample_records(cleaned_df):
    return cleaned_df.head(3)[list(FEATURE_COLS)].to_dict(orient="records")


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "model_path" in body


def test_index_returns_html_form(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/html")
    body = resp.get_data(as_text=True)
    assert "Heart Disease Predictor" in body
    assert 'name="age"' in body and 'name="thal"' in body


def test_metadata_lists_feature_cols(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["feature_cols"] == list(FEATURE_COLS)
    assert body["output_schema"] == ["prediction", "probability", "label"]


def test_predict_single_record(client, sample_records):
    resp = client.post("/predict", json=sample_records[0])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n"] == 1
    pred = body["predictions"][0]
    assert pred["prediction"] in (0, 1)
    assert 0.0 <= pred["probability"] <= 1.0
    assert pred["label"] in ("disease", "no_disease")


def test_predict_list_of_records(client, sample_records):
    resp = client.post("/predict", json=sample_records)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["n"] == len(sample_records)
    assert len(body["predictions"]) == len(sample_records)


def test_predict_missing_required_columns_returns_422(client):
    resp = client.post("/predict", json={"age": 50, "sex": 1})
    assert resp.status_code == 422
    assert "missing required feature columns" in resp.get_json()["error"]


def test_predict_rejects_non_json(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_rejects_empty_body(client):
    resp = client.post("/predict", json=None)
    assert resp.status_code == 400


def test_predict_rejects_non_object_payload(client):
    resp = client.post("/predict", json="hello")
    assert resp.status_code == 400


def test_predict_label_matches_prediction(client, sample_records):
    resp = client.post("/predict", json=sample_records)
    assert resp.status_code == 200
    for pred in resp.get_json()["predictions"]:
        expected = "disease" if pred["prediction"] == 1 else "no_disease"
        assert pred["label"] == expected


# ---- Task 8: monitoring & logging ------------------------------------------

def test_metrics_endpoint_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.content_type.startswith("text/plain")
    body = resp.get_data(as_text=True)
    assert "heart_api_predictions_total" in body
    assert "heart_api_app_info" in body or "heart_api" in body


def test_predictions_counter_increments(client, sample_records):
    import re

    def total(text: str) -> float:
        matches = re.findall(
            r"^heart_api_predictions_total\{label=\"[^\"]+\"\} ([0-9.eE+-]+)$",
            text,
            flags=re.MULTILINE,
        )
        return sum(float(m) for m in matches)

    before = total(client.get("/metrics").get_data(as_text=True))
    resp = client.post("/predict", json=sample_records)
    assert resp.status_code == 200
    after = total(client.get("/metrics").get_data(as_text=True))
    assert after - before == len(sample_records)


def test_access_log_emits_json(client, sample_records, caplog):
    import json
    import logging

    with caplog.at_level(logging.INFO, logger="heart_api.access"):
        resp = client.post("/predict", json=sample_records[0])
    assert resp.status_code == 200
    log_lines = [r.message for r in caplog.records if r.name == "heart_api.access"]
    assert log_lines, "no access log line emitted"
    record = json.loads(log_lines[-1])
    for key in ("ts", "method", "path", "status", "latency_ms"):
        assert key in record
    assert record["method"] == "POST"
    assert record["path"] == "/predict"
    assert record["status"] == 200
    assert record["n_records"] == 1
