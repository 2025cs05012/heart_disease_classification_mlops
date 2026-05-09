"""Tests for ``src.models.predict`` — load + score API and CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLS
from src.models.predict import load_model, predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "heart_pipeline.joblib"
DEFAULT_MLFLOW_MODEL = PROJECT_ROOT / "models" / "mlflow_model"


@pytest.fixture(scope="module")
def trained_pipeline():
    if not DEFAULT_MODEL.is_file():
        pytest.skip("trained pipeline missing; run `python -m src.models.train` first.")
    return joblib.load(DEFAULT_MODEL)


@pytest.fixture(scope="module")
def sample_records(cleaned_df):
    return cleaned_df.head(5).copy()


def test_load_model_from_joblib(trained_pipeline):
    model = load_model(DEFAULT_MODEL)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_load_model_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "does_not_exist.joblib")


@pytest.mark.skipif(not DEFAULT_MLFLOW_MODEL.is_dir(),
                    reason="MLflow model dir missing; run training first.")
def test_load_model_from_mlflow_dir():
    model = load_model(DEFAULT_MLFLOW_MODEL)
    assert hasattr(model, "predict")


def test_predict_dataframe_shape_and_columns(trained_pipeline, sample_records):
    out = predict(trained_pipeline, sample_records)
    assert list(out.columns) == ["prediction", "probability", "label"]
    assert len(out) == len(sample_records)
    assert set(out["prediction"].unique()).issubset({0, 1})
    assert ((out["probability"] >= 0) & (out["probability"] <= 1)).all()
    assert set(out["label"].unique()).issubset({"disease", "no_disease"})


def test_predict_accepts_list_of_dicts(trained_pipeline, sample_records):
    records = sample_records[list(FEATURE_COLS)].to_dict(orient="records")
    out = predict(trained_pipeline, records)
    assert len(out) == len(records)


def test_predict_missing_required_column_raises(trained_pipeline, sample_records):
    bad = sample_records.drop(columns=["age"])
    with pytest.raises(KeyError, match="missing required feature columns"):
        predict(trained_pipeline, bad)


def test_predict_label_matches_prediction(trained_pipeline, sample_records):
    out = predict(trained_pipeline, sample_records)
    for _, row in out.iterrows():
        expected = "disease" if row["prediction"] == 1 else "no_disease"
        assert row["label"] == expected


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(),
                    reason="trained pipeline missing; run training first.")
def test_predict_cli_csv_roundtrip(tmp_path, cleaned_df):
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    cleaned_df.head(4).to_csv(in_csv, index=False)

    result = subprocess.run(
        [sys.executable, "-m", "src.models.predict",
         "--input", str(in_csv), "--output", str(out_csv)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out_csv.is_file()
    df = pd.read_csv(out_csv)
    assert list(df.columns) == ["prediction", "probability", "label"]
    assert len(df) == 4


@pytest.mark.skipif(not DEFAULT_MODEL.is_file(),
                    reason="trained pipeline missing; run training first.")
def test_predict_cli_json_input(tmp_path, cleaned_df):
    in_json = tmp_path / "in.json"
    payload = cleaned_df.head(2)[list(FEATURE_COLS)].to_dict(orient="records")
    in_json.write_text(json.dumps(payload))

    result = subprocess.run(
        [sys.executable, "-m", "src.models.predict",
         "--input", str(in_json)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    lines = [ln for ln in result.stdout.strip().splitlines() if ln]
    assert lines[0] == "prediction,probability,label"
    assert len(lines) == 3
