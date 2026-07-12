"""Utility helpers shared across the project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_path(filename: str) -> Path:
    """Return the preferred location of a dataset file."""
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path

    root_path = ROOT_DIR / filename
    if root_path.exists():
        return root_path

    raise FileNotFoundError(f"Could not find dataset file: {filename}")


def read_csv_with_fallback(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read CSV files while tolerating common encoding issues."""
    encodings: Iterable[str] = ("utf-8", "latin-1", "cp1252")
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(f"Unable to read {path.name} with supported encodings") from last_error


def normalize_text(value: object, fallback: str = "Unknown") -> str:
    """Clean and standardize text values used by the app and model."""
    if pd.isna(value):
        return fallback

    text = str(value).strip()
    return " ".join(text.split()) if text else fallback


def safe_year(value: object) -> int | None:
    """Convert raw publication years into a validated integer range."""
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    return year if 1800 <= year <= 2026 else None

