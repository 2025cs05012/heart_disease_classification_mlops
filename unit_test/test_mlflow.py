"""Verify MLflow instrumentation in ``src.models.train``.

Runs a minimal training pass with ``use_mlflow=True`` against a temporary
tracking URI and asserts the parent + nested runs, params, metrics, and
artifacts (model, figures, metrics.json) are persisted.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

mlflow = pytest.importorskip("mlflow")


def _tiny_candidate_factory(tm):
    return lambda rs: [
        tm.Candidate(
            name="logreg",
            estimator=LogisticRegression(max_iter=500, random_state=rs),
            grid={"clf__C": [1.0]},
        )
    ]


def test_train_with_mlflow_creates_parent_and_nested_runs(
    tmp_path, monkeypatch, cleaned_df,
):
    csv = tmp_path / "clean.csv"
    cleaned_df.to_csv(csv, index=False)
    tracking_uri = (tmp_path / "mlruns").as_uri()

    import src.models.train as tm

    monkeypatch.setattr(tm, "candidates", _tiny_candidate_factory(tm))
    summary = tm.train(
        data_path=csv,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        cv=3,
        random_state=0,
        test_size=0.2,
        use_mlflow=True,
        mlflow_tracking_uri=tracking_uri,
    )

    assert summary["best_model"] == "logreg"

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("heart_disease_classification")
    assert experiment is not None, "experiment was not created"

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 2, "expected at least one parent + one nested run"

    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == 1, f"expected 1 parent run, got {len(parents)}"
    assert len(children) == 1, f"expected 1 nested run, got {len(children)}"
    assert children[0].data.tags["mlflow.parentRunId"] == parents[0].info.run_id

    parent = parents[0]
    assert parent.data.params["best_model"] == "logreg"
    assert parent.data.params["cv"] == "3"
    assert parent.data.params["random_state"] == "0"
    for key in ("best_accuracy", "best_precision", "best_recall",
                "best_f1", "best_roc_auc"):
        assert key in parent.data.metrics, f"missing metric {key} on parent run"

    child = children[0]
    # log_candidate_run prefixes the chosen-best params with "best_"; sklearn
    # autolog also writes the same value under "best_clf__C".
    assert child.data.params["best_C"] == "1.0"
    assert "cv_roc_auc_mean" in child.data.metrics
    for key in ("test_accuracy", "test_precision", "test_recall",
                "test_f1", "test_roc_auc"):
        assert key in child.data.metrics, f"missing metric {key} on nested run"

    artifacts = {a.path for a in client.list_artifacts(parent.info.run_id)}
    assert "model" in artifacts
    assert "figures" in artifacts
    assert "metrics.json" in artifacts

    figure_files = {a.path for a in client.list_artifacts(parent.info.run_id, "figures")}
    assert "figures/roc_curves.png" in figure_files
    assert "figures/confusion_matrix.png" in figure_files


def test_setup_mlflow_uses_explicit_uri(tmp_path):
    from src.utils.mlflow_utils import setup_mlflow

    uri = (tmp_path / "mlruns").as_uri()
    resolved = setup_mlflow(experiment_name="unit_test_exp", tracking_uri=uri)
    assert resolved == uri
    assert mlflow.get_tracking_uri() == uri
