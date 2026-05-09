"""Unit tests for ``src.data.preprocess``."""
from __future__ import annotations

import pandas as pd

from src.data.preprocess import COLUMNS, SOURCES, load_raw, preprocess


def test_load_raw_returns_expected_shape():
    df = load_raw()
    # 14 canonical UCI columns + the synthetic ``source`` column.
    assert df.shape == (920, 1 + len(COLUMNS))
    assert set(df["source"].unique()) == set(SOURCES.values())


def test_chol_sentinel_zeros_replaced(cleaned_df):
    assert (cleaned_df["chol"] == 0).sum() == 0
    assert cleaned_df["chol"].isna().sum() >= 100  # 173 zeros + 30 raw NA


def test_trestbps_sentinel_zero_replaced(cleaned_df):
    assert (cleaned_df["trestbps"] == 0).sum() == 0


def test_target_is_binary_and_balanced(cleaned_df):
    assert set(cleaned_df["target"].unique()).issubset({0, 1})
    positive_rate = cleaned_df["target"].mean()
    assert 0.4 < positive_rate < 0.7


def test_ca_missing_matches_ca_nan(cleaned_df):
    assert "ca_missing" in cleaned_df.columns
    assert set(cleaned_df["ca_missing"].unique()).issubset({0, 1})
    assert (cleaned_df["ca_missing"] == cleaned_df["ca"].isna().astype(int)).all()


def test_preprocess_writes_csv(tmp_path):
    out = tmp_path / "clean.csv"
    df = preprocess(out_path=out)
    assert out.is_file()
    reread = pd.read_csv(out)
    assert reread.shape == df.shape
    assert "target" in reread.columns
