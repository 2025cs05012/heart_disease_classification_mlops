"""Feature-engineering pipeline for the Heart Disease classifier.

Builds the sklearn ``ColumnTransformer`` that turns the cleaned dataframe
produced by :mod:`src.data.preprocess` into the matrix consumed by every
downstream model:

    numeric (incl. ``ca``)  -> SimpleImputer(median)        -> StandardScaler
    categorical             -> SimpleImputer(most_frequent) -> OneHotEncoder
    ca_missing flag         -> passthrough
    {source, num, target}   -> dropped

The transformer is deliberately decoupled from any specific estimator so it
can be reused at training time, in MLflow logging, in unit tests, and when
the Flask ``/predict`` endpoint serves a single incoming JSON record.

Usage:
    python -m src.features.build_features                  # verify on processed CSV
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS: tuple[str, ...] = (
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
)
CATEGORICAL_COLS: tuple[str, ...] = (
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal",
)
PASSTHROUGH_COLS: tuple[str, ...] = ("ca_missing",)

FEATURE_COLS: tuple[str, ...] = NUMERIC_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS
TARGET_COL: str = "target"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"

log = logging.getLogger(__name__)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` using only the configured feature columns.

    Casts pandas nullable ``Int64`` columns to float so that ``SimpleImputer``
    treats ``<NA>`` consistently as ``NaN``.
    """
    required = set(FEATURE_COLS + (TARGET_COL,))
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Required columns missing from dataframe: {sorted(missing)}")

    X = df[list(FEATURE_COLS)].copy()
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in CATEGORICAL_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in PASSTHROUGH_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype(int)

    y = df[TARGET_COL].astype(int).copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """sklearn ``ColumnTransformer`` performing impute + scale / one-hot."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, list(NUMERIC_COLS)),
            ("cat", categorical_pipe, list(CATEGORICAL_COLS)),
            ("pass", "passthrough", list(PASSTHROUGH_COLS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Names of the columns emitted by a fitted ``ColumnTransformer``."""
    return list(preprocessor.get_feature_names_out())


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                   help="Path to the cleaned CSV produced by src.data.preprocess.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    df = pd.read_csv(args.data)
    X, y = split_xy(df)
    preprocessor = build_preprocessor()
    Xt = preprocessor.fit_transform(X)
    names = get_feature_names(preprocessor)

    log.info("Input shape   : %s  (target balance: %s)",
             X.shape, y.value_counts(normalize=True).round(3).to_dict())
    log.info("Output shape  : %s", Xt.shape)
    log.info("Output columns: %s", names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
