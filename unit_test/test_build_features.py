"""Unit tests for ``src.features.build_features``."""
from __future__ import annotations

import numpy as np

from src.features.build_features import (
    FEATURE_COLS,
    build_preprocessor,
    get_feature_names,
    split_xy,
)


def test_split_xy_returns_configured_columns(cleaned_df):
    X, y = split_xy(cleaned_df)
    assert list(X.columns) == list(FEATURE_COLS)
    assert len(X) == len(y) == 920
    assert set(y.unique()).issubset({0, 1})


def test_preprocessor_output_shape_and_no_nan(cleaned_df):
    X, _ = split_xy(cleaned_df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape == (920, 26)
    assert not np.isnan(Xt).any()
    assert len(get_feature_names(pre)) == Xt.shape[1]


def test_preprocessor_ignores_unseen_category(cleaned_df):
    X, _ = split_xy(cleaned_df)
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row.loc[:, "sex"] = 9.0  # value never seen during fit
    out = pre.transform(row)
    # OneHotEncoder(handle_unknown='ignore') -> all-zero block, no crash.
    assert out.shape == (1, 26)
