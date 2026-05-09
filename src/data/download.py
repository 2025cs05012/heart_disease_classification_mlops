"""Download the UCI Heart Disease dataset into ``data/raw/``.

Idempotent: skips the download if the expected source files already exist.

Usage:
    python -m src.data.download                # default destination
    python -m src.data.download --force        # re-download even if present
"""
from __future__ import annotations

import argparse
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import requests

UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
ARCHIVE_NAME = "heart+disease.zip"
EXTRACT_DIR_NAME = "heart+disease"

REQUIRED_FILES: tuple[str, ...] = (
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
    "heart-disease.names",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

log = logging.getLogger(__name__)


def _all_present(extract_dir: Path, files: Iterable[str]) -> bool:
    return extract_dir.is_dir() and all((extract_dir / f).is_file() for f in files)


def _download(url: str, dest: Path, timeout: int = 60) -> None:
    log.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            shutil.copyfileobj(resp.raw, fh)


def _extract(zip_path: Path, target_dir: Path) -> None:
    log.info("Extracting %s -> %s", zip_path, target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)


def download_heart_disease(raw_dir: Path = DEFAULT_RAW_DIR, force: bool = False) -> Path:
    """Ensure the UCI heart disease files are available under ``raw_dir``.

    Returns the path to the extracted dataset directory.
    """
    raw_dir = Path(raw_dir)
    extract_dir = raw_dir / EXTRACT_DIR_NAME
    archive_path = raw_dir / ARCHIVE_NAME

    if not force and _all_present(extract_dir, REQUIRED_FILES):
        log.info("Dataset already present at %s; skipping download.", extract_dir)
        return extract_dir

    if force or not archive_path.is_file():
        _download(UCI_URL, archive_path)
    else:
        log.info("Archive already present at %s; reusing.", archive_path)

    _extract(archive_path, extract_dir)

    missing = [f for f in REQUIRED_FILES if not (extract_dir / f).is_file()]
    if missing:
        raise RuntimeError(f"Required files missing after extraction: {missing}")

    log.info("Dataset ready at %s", extract_dir)
    return extract_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Destination raw-data directory.")
    p.add_argument("--force", action="store_true",
                   help="Re-download even when the dataset is already present.")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)
    download_heart_disease(raw_dir=args.raw_dir, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
