"""Loading and validation of the German Credit data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import EXCLUDED_FEATURES, TARGET, TARGET_RAW

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "germancredit.csv"


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load the raw CSV, encode the target (bad = 1) and drop excluded features."""
    df = pd.read_csv(path)
    validate(df)
    df = df.copy()
    df[TARGET] = (df[TARGET_RAW] == "bad").astype(int)
    df = df.drop(columns=[TARGET_RAW] + [c for c in EXCLUDED_FEATURES if c in df.columns])
    return df


def validate(df: pd.DataFrame) -> None:
    """Fail fast on data problems instead of training on them."""
    if TARGET_RAW not in df.columns:
        raise ValueError(f"missing target column '{TARGET_RAW}'")
    labels = set(df[TARGET_RAW].unique())
    if not labels <= {"good", "bad"}:
        raise ValueError(f"unexpected target labels: {sorted(labels - {'good', 'bad'})}")
    if df.isna().any().any():
        cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"missing values in columns: {cols}")
    if len(df) < 100:
        raise ValueError("too few rows to build a scorecard")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET]
