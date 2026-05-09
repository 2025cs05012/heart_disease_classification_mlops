"""Preprocess the UCI Heart Disease dataset.

Loads the four ``processed.*.data`` files, normalises types, replaces sentinel
"missing" tokens, derives the binary classification target, and writes a tidy
CSV to ``data/processed/heart_disease_clean.csv``.

Usage:
    python -m src.data.preprocess
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS: tuple[str, ...] = (
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
)

# Mapping of UCI source file -> short label written into the ``source`` column.
SOURCES: dict[str, str] = {
    "processed.cleveland.data": "cleveland",
    "processed.hungarian.data": "hungarian",
    "processed.switzerland.data": "switzerland",
    "processed.va.data": "long_beach_va",
}

# Columns where 0 is biologically impossible and is used by UCI as a sentinel
# for "missing" (most prominently in the Switzerland subset, where the entire
# ``chol`` column is recorded as 0).
SENTINEL_ZERO_COLS: tuple[str, ...] = ("trestbps", "chol")

CATEGORICAL_INT_COLS: tuple[str, ...] = (
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "num",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "heart+disease"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease_clean.csv"

log = logging.getLogger(__name__)


def _load_one(path: Path, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=list(COLUMNS),
        na_values=["?"],
        skipinitialspace=True,
    )
    df.insert(0, "source", source_label)
    return df


def load_raw(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Concatenate the four processed UCI subsets into a single frame."""
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for filename, label in SOURCES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Expected UCI subset not found: {path}")
        frames.append(_load_one(path, label))
    df = pd.concat(frames, axis=0, ignore_index=True)
    log.info("Loaded %d rows from %d sources.", len(df), len(frames))
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data-cleaning rules and add the binary target column."""
    out = df.copy()

    # Sentinel zeros -> NaN (cholesterol of 0 mg/dl is medically impossible).
    for col in SENTINEL_ZERO_COLS:
        zeros = (out[col] == 0).sum()
        if zeros:
            log.info("Replacing %d sentinel zeros in '%s' with NaN.", zeros, col)
            out.loc[out[col] == 0, col] = np.nan

    # Type normalisation: integer-valued categoricals -> nullable Int64.
    for col in CATEGORICAL_INT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    # Continuous numerics -> float.
    for col in ("age", "trestbps", "chol", "thalach", "oldpeak"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    # Binary classification target: any disease (num > 0).
    out["target"] = (out["num"].fillna(0) > 0).astype(int)

    # Explicit missingness indicator for the high-NA ``ca`` feature so that
    # downstream models can learn from the pattern of missingness itself.
    out["ca_missing"] = out["ca"].isna().astype(int)

    return out


def preprocess(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_path: Path = DEFAULT_OUT_PATH,
) -> pd.DataFrame:
    """End-to-end: load raw subsets, clean, persist, return the clean frame."""
    df = load_raw(raw_dir)
    cleaned = clean(df)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    log.info("Wrote cleaned dataset (%d rows, %d cols) -> %s",
             len(cleaned), cleaned.shape[1], out_path)
    return cleaned


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Directory containing the processed.*.data files.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH,
                   help="Destination CSV path for the cleaned dataset.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)
    preprocess(raw_dir=args.raw_dir, out_path=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
