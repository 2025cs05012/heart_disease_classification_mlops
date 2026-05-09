"""Load the trained Heart Disease pipeline and score new records.

Supports both artefact formats produced by :mod:`src.models.train`:

* ``models/heart_pipeline.joblib`` (default, fastest)
* ``models/mlflow_model/``         (portable MLflow flavour, ``mlflow.pyfunc``)

The same module is used by the unit tests and will be imported by the Flask
``/predict`` endpoint in Task 6, so the public API is intentionally small.

Usage:
    python -m src.models.predict --input sample.csv --output preds.csv
    python -m src.models.predict --input sample.json --json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping

import joblib
import pandas as pd

from src.features.build_features import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    PASSTHROUGH_COLS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "heart_pipeline.joblib"

log = logging.getLogger(__name__)


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load a trained pipeline from a ``.joblib`` file or MLflow model dir."""
    p = Path(path)
    if p.is_dir() and (p / "MLmodel").is_file():
        import mlflow.pyfunc
        log.info("Loading MLflow pyfunc model from %s", p)
        return mlflow.pyfunc.load_model(str(p))
    if not p.is_file():
        raise FileNotFoundError(
            f"Model artefact not found at {p}. "
            "Run `python -m src.models.train` first."
        )
    log.info("Loading joblib pipeline from %s", p)
    return joblib.load(p)


def _coerce_to_frame(records: pd.DataFrame | Iterable[Mapping]) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(list(records))
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Input is missing required feature columns: {missing}. "
            f"Expected columns: {list(FEATURE_COLS)}"
        )
    X = df[list(FEATURE_COLS)].copy()
    for col in NUMERIC_COLS + CATEGORICAL_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in PASSTHROUGH_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)
    return X


def predict(model, records: pd.DataFrame | Iterable[Mapping]) -> pd.DataFrame:
    """Score ``records`` and return a DataFrame with ``prediction`` + ``probability``.

    ``probability`` is the model's confidence in the positive ('disease') class.
    Works for both the joblib sklearn pipeline (uses ``predict_proba``) and an
    MLflow ``pyfunc`` model (falls back to ``predict`` only).
    """
    X = _coerce_to_frame(records)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)
    else:
        raw = model.predict(X)
        preds = pd.Series(raw).astype(int).to_numpy()
        proba = preds.astype(float)
    return pd.DataFrame({
        "prediction": preds,
        "probability": [float(p) for p in proba],
        "label": ["disease" if p == 1 else "no_disease" for p in preds],
    })


def _load_input(path: Path, as_json: bool) -> pd.DataFrame:
    if as_json or path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping):
            payload = [payload]
        return pd.DataFrame(list(payload))
    return pd.read_csv(path)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Path to the joblib file or MLflow model directory.")
    p.add_argument("--input", type=Path, required=True,
                   help="CSV or JSON file containing records to score.")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write predictions (CSV). Defaults to stdout.")
    p.add_argument("--json", dest="as_json", action="store_true",
                   help="Force JSON parsing of the input file.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    model = load_model(args.model)
    df_in = _load_input(args.input, args.as_json)
    log.info("Scoring %d record(s)", len(df_in))
    preds = predict(model, df_in)

    if args.output is None:
        print(preds.to_csv(index=False), end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(args.output, index=False)
        log.info("Wrote predictions -> %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
