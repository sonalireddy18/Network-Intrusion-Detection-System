from typing import Iterable, List
import numpy as np
import pandas as pd

LEAKAGE_IDENTIFIERS = [
    "flow id", "source ip", "src ip", "destination ip", "dst ip",
    "timestamp", "source port", "src port", "destination port", "dst port"
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and drop duplicate header collisions."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "Fwd Header Length.1" in df.columns and "Fwd Header Length" in df.columns:
        if df["Fwd Header Length.1"].equals(df["Fwd Header Length"]):
            df = df.drop(columns=["Fwd Header Length.1"])

    duplicated_cols = df.columns[df.columns.duplicated()].unique()
    for col in duplicated_cols:
        positions = [i for i, name in enumerate(df.columns) if name == col]
        first = df.iloc[:, positions[0]]
        for pos in reversed(positions[1:]):
            if df.iloc[:, pos].equals(first):
                df = df.drop(columns=[df.columns[pos]], errors="ignore")
    return df


def find_label_column(columns: Iterable[str]) -> str:
    """Locate the target column regardless of letter casing."""
    matches = [str(c) for c in columns if str(c).strip().casefold() == "label"]
    if len(matches) != 1:
        raise ValueError(f"Expected 1 target column named 'Label'; found {matches!r}.")
    return matches[0]


def drop_leakage_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove network socket identifiers and temporal markers to prevent shortcut learning."""
    df = df.copy()
    cols_to_drop = [
        c for c in df.columns
        if str(c).strip().casefold() in LEAKAGE_IDENTIFIERS
        or str(c).strip().casefold().startswith("unnamed:")
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def sanitize_numeric_values(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Coerce features to numeric, drop missing labels, remove duplicates, and impute medians."""
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Target column {label_col!r} missing.")

    df[label_col] = df[label_col].astype("string").str.strip()
    df = df.dropna(subset=[label_col])
    df = df[df[label_col] != ""].drop_duplicates()

    num_cols = [c for c in df.columns if c != label_col]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    for c in num_cols:
        if df[c].isna().all():
            df = df.drop(columns=[c])
        elif df[c].isna().any():
            median_val = df[c].median()
            df[c] = df[c].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Remove zero-variance (constant) features
    constant_cols = [c for c in df.columns if c != label_col and df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)

    return df





