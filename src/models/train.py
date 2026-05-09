"""Train and evaluate Heart Disease classifiers.

Builds a single ``Pipeline(preprocessor + estimator)`` for each candidate
model, tunes hyperparameters via 5-fold stratified ``GridSearchCV``
(``scoring='roc_auc'``), evaluates the best estimator of each family on a
held-out 20 % test split, persists the overall winner to
``models/heart_pipeline.joblib`` **and** the portable MLflow flavour at
``models/mlflow_model/``, and writes ``reports/metrics.json`` plus ROC and
confusion-matrix figures under ``reports/figures/``.

Usage:
    python -m src.models.train [--cv 5] [--random-state 42] [--test-size 0.2]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.features.build_features import build_preprocessor, split_xy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

log = logging.getLogger(__name__)


_ANSI = {
    "reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
    "bright_green": "\033[92m", "bright_red": "\033[91m",
    "bright_cyan": "\033[96m",
}

_LEVEL_COLOR = {
    logging.DEBUG: _ANSI["dim"] + _ANSI["cyan"],
    logging.INFO: _ANSI["green"],
    logging.WARNING: _ANSI["yellow"],
    logging.ERROR: _ANSI["bright_red"],
    logging.CRITICAL: _ANSI["bold"] + _ANSI["bright_red"],
}

_BRACKET_RE = re.compile(r"\[([a-zA-Z_][\w\-]*)\]")
_KV_RE = re.compile(r"([a-zA-Z_][\w]*)=([+-]?\d+(?:\.\d+)?)")
_PATH_RE = re.compile(r"(/[\w./\-]+\.(?:json|joblib|png|csv|yaml|yml))")


class ColoredFormatter(logging.Formatter):
    """Logging formatter that paints level/timestamp/message with ANSI colors.

    Highlights ``[model_name]`` brackets in cyan and ``key=number`` metric
    pairs in bright green so train logs are scannable at a glance. Falls
    back to plain text when stderr is not a TTY or ``NO_COLOR`` is set.
    """

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%H:%M:%S")
        msg = record.getMessage()
        if not self.use_color:
            return f"{ts} {record.levelname:<7s} {record.name}: {msg}"
        lvl_color = _LEVEL_COLOR.get(record.levelno, "")
        msg = _BRACKET_RE.sub(
            lambda m: f"{_ANSI['bright_cyan']}{_ANSI['bold']}[{m.group(1)}]{_ANSI['reset']}", msg)
        msg = _KV_RE.sub(
            lambda m: f"{_ANSI['bold']}{_ANSI['bright_green']}{m.group(1)}={m.group(2)}{_ANSI['reset']}", msg)
        msg = _PATH_RE.sub(
            lambda m: f"{_ANSI['magenta']}{m.group(1)}{_ANSI['reset']}", msg)
        return (
            f"{_ANSI['dim']}{ts}{_ANSI['reset']} "
            f"{lvl_color}{_ANSI['bold']}{record.levelname:<7s}{_ANSI['reset']} "
            f"{_ANSI['dim']}{record.name}{_ANSI['reset']}: {msg}"
        )


def _setup_logging(level: int = logging.INFO) -> None:
    if os.environ.get("NO_COLOR") is not None:
        use_color = False
    elif os.environ.get("FORCE_COLOR") is not None:
        use_color = True
    else:
        use_color = sys.stderr.isatty() and os.environ.get("TERM", "") != "dumb"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColoredFormatter(use_color=use_color))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


@dataclass
class Candidate:
    name: str
    estimator: object
    grid: dict


def candidates(random_state: int) -> list[Candidate]:
    return [
        Candidate(
            name="logreg",
            estimator=LogisticRegression(max_iter=2000, random_state=random_state),
            grid={"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l2"]},
        ),
        Candidate(
            name="random_forest",
            estimator=RandomForestClassifier(random_state=random_state, n_jobs=-1),
            grid={"clf__n_estimators": [200], "clf__max_depth": [None, 8],
                  "clf__min_samples_split": [2, 5]},
        ),
        Candidate(
            name="gradient_boosting",
            estimator=GradientBoostingClassifier(random_state=random_state),
            grid={"clf__n_estimators": [150], "clf__learning_rate": [0.05, 0.1],
                  "clf__max_depth": [3]},
        ),
    ]


def _evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def train(data_path: Path, models_dir: Path, reports_dir: Path,
          cv: int, random_state: int, test_size: float,
          use_mlflow: bool = False,
          mlflow_tracking_uri: str | None = None) -> dict:
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from contextlib import nullcontext

    if use_mlflow:
        import mlflow

        from src.utils.mlflow_utils import (
            candidate_run_ctx,
            log_candidate_run,
            parent_run_ctx,
            setup_mlflow,
        )
        setup_mlflow(tracking_uri=mlflow_tracking_uri)
        run_name = f"grid_search_{datetime.now():%Y%m%d_%H%M%S}"
        parent_ctx = parent_run_ctx(run_name, enabled=True)
    else:
        parent_ctx = nullcontext()

    df = pd.read_csv(data_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    log.info("Train: %s  Test: %s  positive rate: %.3f",
             X_train.shape, X_test.shape, y_train.mean())

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}

    with parent_ctx:
        if use_mlflow:
            mlflow.set_tags({"random_state": random_state, "test_size": test_size,
                             "cv": cv, "n_train": len(X_train), "n_test": len(X_test)})
            mlflow.log_params({"random_state": random_state,
                               "test_size": test_size, "cv": cv})

        for cand in candidates(random_state):
            pipe = Pipeline([("pre", build_preprocessor()), ("clf", cand.estimator)])
            gs = GridSearchCV(pipe, cand.grid, cv=skf, scoring="roc_auc",
                              n_jobs=-1, refit=True)
            log.info("[%s] grid=%s", cand.name, cand.grid)
            # gs.fit() must run INSIDE the nested run so sklearn autolog
            # attaches the full pipeline params + training metrics to it.
            cand_ctx = candidate_run_ctx(cand.name) if use_mlflow else nullcontext()
            with cand_ctx:
                gs.fit(X_train, y_train)
                metrics = _evaluate(gs.best_estimator_, X_test, y_test)
                results[cand.name] = {"best_params": gs.best_params_,
                                      "cv_roc_auc_mean": float(gs.best_score_),
                                      "test_metrics": {k: float(v) for k, v in metrics.items()}}
                fitted[cand.name] = gs.best_estimator_
                log.info("[%s] cv_roc_auc=%.4f  test=%s",
                         cand.name, gs.best_score_,
                         {k: round(v, 4) for k, v in metrics.items()})
                if use_mlflow:
                    log_candidate_run(cand.name, gs.best_params_, gs.best_score_,
                                      metrics, cv_results=gs.cv_results_)

        best_name = max(results, key=lambda k: results[k]["test_metrics"]["roc_auc"])
        best_pipe = fitted[best_name]
        log.info("Best model: %s (test ROC-AUC=%.4f)",
                 best_name, results[best_name]["test_metrics"]["roc_auc"])

        fig, ax = plt.subplots(figsize=(7, 6))
        for name, pipe in fitted.items():
            RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        ax.set_title("ROC curves on held-out test set")
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curves.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay.from_estimator(best_pipe, X_test, y_test,
                                              display_labels=["no_disease", "disease"],
                                              cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix — {best_name}")
        fig.tight_layout()
        fig.savefig(figures_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close(fig)

        model_path = models_dir / "heart_pipeline.joblib"
        joblib.dump(best_pipe, model_path)
        log.info("Saved best pipeline -> %s", model_path)

        mlflow_model_dir = models_dir / "mlflow_model"
        if mlflow_model_dir.exists():
            shutil.rmtree(mlflow_model_dir)
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train.head(5),
                                    best_pipe.predict(X_train.head(5)))
        mlflow.sklearn.save_model(
            sk_model=best_pipe,
            path=str(mlflow_model_dir),
            signature=signature,
            input_example=X_train.head(2),
        )
        log.info("Saved MLflow model -> %s", mlflow_model_dir)

        summary = {"best_model": best_name, "models": results,
                   "n_train": int(len(X_train)), "n_test": int(len(X_test)),
                   "cv": cv, "random_state": random_state, "test_size": test_size}
        metrics_path = reports_dir / "metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2))
        log.info("Wrote metrics summary -> %s", metrics_path)

        if use_mlflow:
            mlflow.log_param("best_model", best_name)
            for k, v in results[best_name]["test_metrics"].items():
                mlflow.log_metric(f"best_{k}", float(v))
            mlflow.log_artifact(str(figures_dir / "roc_curves.png"), "figures")
            mlflow.log_artifact(str(figures_dir / "confusion_matrix.png"), "figures")
            mlflow.log_artifact(str(metrics_path))
            mlflow.sklearn.log_model(best_pipe, artifact_path="model")
            log.info("Logged best model to MLflow run %s",
                     mlflow.active_run().info.run_id)

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--mlflow", dest="use_mlflow", action=argparse.BooleanOptionalAction,
                   default=True, help="Enable/disable MLflow tracking (default: enabled).")
    p.add_argument("--mlflow-tracking-uri", default=None,
                   help="Override MLflow tracking URI (else uses Assignment/mlruns).")
    return p


def main(argv: list[str] | None = None) -> int:
    _setup_logging(logging.INFO)
    args = _build_arg_parser().parse_args(argv)
    train(args.data, args.models_dir, args.reports_dir,
          args.cv, args.random_state, args.test_size,
          use_mlflow=args.use_mlflow, mlflow_tracking_uri=args.mlflow_tracking_uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
