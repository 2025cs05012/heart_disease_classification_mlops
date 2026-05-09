"""Smoke tests for ``src.models.train``."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.build_features import split_xy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "heart_pipeline.joblib"
DEFAULT_METRICS = PROJECT_ROOT / "reports" / "metrics.json"


@pytest.mark.skipif(
    not DEFAULT_MODEL.is_file(),
    reason="trained pipeline missing; run `python -m src.models.train` first.",
)
def test_persisted_pipeline_predicts_probabilities(cleaned_df):
    pipe = joblib.load(DEFAULT_MODEL)
    X, _ = split_xy(cleaned_df)
    proba = pipe.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2)
    assert (proba.sum(axis=1).round(6) == 1.0).all()


@pytest.mark.skipif(
    not DEFAULT_METRICS.is_file(),
    reason="metrics.json missing; run training first.",
)
def test_metrics_summary_has_all_models():
    summary = json.loads(DEFAULT_METRICS.read_text())
    assert summary["best_model"] in summary["models"]
    for name in ("logreg", "random_forest", "gradient_boosting"):
        m = summary["models"][name]["test_metrics"]
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0


def test_train_end_to_end_with_tiny_grid(tmp_path, monkeypatch, cleaned_df):
    """End-to-end smoke run with a single tiny model so CI stays fast."""
    csv = tmp_path / "clean.csv"
    cleaned_df.to_csv(csv, index=False)

    import src.models.train as tm

    monkeypatch.setattr(tm, "candidates", lambda rs: [
        tm.Candidate(
            name="logreg",
            estimator=LogisticRegression(max_iter=500, random_state=rs),
            grid={"clf__C": [1.0]},
        )
    ])
    summary = tm.train(
        data_path=csv,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        cv=3,
        random_state=0,
        test_size=0.2,
    )
    assert summary["best_model"] == "logreg"
    assert (tmp_path / "models" / "heart_pipeline.joblib").is_file()
    assert (tmp_path / "reports" / "metrics.json").is_file()
    assert (tmp_path / "reports" / "figures" / "roc_curves.png").is_file()
