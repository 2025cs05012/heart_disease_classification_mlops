"""Pytest configuration: make the project root importable and share fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def cleaned_df():
    """Cleaned heart-disease dataframe; loaded once per test session."""
    from src.data.preprocess import clean, load_raw

    return clean(load_raw())
