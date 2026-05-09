"""MLflow tracking helpers for the Heart Disease project.

Centralises the experiment name, tracking URI resolution, and the small
helper that opens a nested per-candidate run.  Defaults to a local file-store
backend at ``Assignment/mlruns/`` so that no external service is required.
"""
from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path

import mlflow
import mlflow.sklearn  # noqa: F401  (registers autolog hooks on import)

DEFAULT_EXPERIMENT = "heart_disease_classification"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()

log = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = DEFAULT_EXPERIMENT,
                 tracking_uri: str | None = None) -> str:
    """Configure MLflow tracking and (re)select the experiment.

    Resolution order for the tracking URI:
        1. explicit ``tracking_uri`` argument
        2. ``MLFLOW_TRACKING_URI`` environment variable
        3. ``Assignment/mlruns`` file-store

    Also enables ``mlflow.sklearn.autolog`` so every ``fit()`` call inside
    an active run captures the full pipeline ``get_params()``, training
    metrics, and the ``GridSearchCV.cv_results_`` table automatically.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    try:
        mlflow.sklearn.autolog(
            log_models=False,        # we save the winner ourselves
            log_datasets=False,
            silent=True,
            max_tuning_runs=0,       # don't fan out a child run per grid combo
        )
    except Exception as exc:         # pragma: no cover - best effort
        log.debug("sklearn autolog setup skipped: %s", exc)
    log.info("MLflow tracking URI: %s | experiment: %s", uri, experiment_name)
    return uri


def parent_run_ctx(name: str, enabled: bool = True):
    """Return a context manager: parent ``mlflow.start_run`` or ``nullcontext``."""
    return mlflow.start_run(run_name=name) if enabled else nullcontext()


def candidate_run_ctx(name: str):
    """Open a nested run for one candidate so autolog attaches to it."""
    return mlflow.start_run(run_name=name, nested=True)


def _jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def log_candidate_run(name: str, best_params: dict, cv_score: float,
                      test_metrics: dict, cv_results: dict | None = None) -> None:
    """Log curated extras into the currently-active MLflow run.

    Caller must already be inside an open run (e.g. via
    ``candidate_run_ctx``) so that sklearn autolog has logged the full
    estimator params there too.  Adds:

      * ``best_<param>`` for every key chosen by GridSearchCV
      * ``cv_roc_auc_mean`` and ``test_<metric>`` numbers
      * ``cv_results.json`` artifact (full grid-search results table)
    """
    mlflow.log_params({f"best_{k.replace('clf__', '')}": v
                       for k, v in best_params.items()})
    mlflow.log_metric("cv_roc_auc_mean", float(cv_score))
    for k, v in test_metrics.items():
        mlflow.log_metric(f"test_{k}", float(v))
    if cv_results is not None:
        try:
            jsonable = {k: [_jsonable(x) for x in v] for k, v in cv_results.items()}
            mlflow.log_dict(jsonable, "cv_results.json")
        except Exception as exc:     # pragma: no cover - best effort
            log.debug("cv_results dump skipped: %s", exc)
