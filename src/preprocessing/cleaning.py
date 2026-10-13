"""
cleaning.py - Data Sanitization & Leakage Prevention
Primary Owner: Member 1 (Random Forest Lead)
"""
from typing import List
import numpy as np
import pandas as pd

# Identifier headers that cause shortcut learning and target leakage (Sarhan et al., 2022)
LEAKAGE_IDENTIFIERS = [
    "flow id", "source ip", "src ip", "destination ip", "dst ip",
    "timestamp", "source port", "src port", "destination port", "dst port"
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and unify casing across raw CSV headers."""
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_label_column(columns: List[str]) -> str:
    """Locate the target column regardless of casing or extra padding."""
    for c in columns:
        if str(c).strip().lower() == "label":
            return c
    raise KeyError(f"Could not locate 'Label' column in: {list(columns)[:10]}")


def drop_leakage_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop socket metadata and timestamp markers to prevent network overfitting."""
    cols_to_drop = [
        c for c in df.columns 
        if str(c).strip().lower() in LEAKAGE_IDENTIFIERS
    ]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def sanitize_numeric_values(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    1. Drop rows missing the target label.
    2. Remove duplicate flow records.
    3. Coerce numeric columns, map +/- inf to NaN, and impute with column medians.
    """
    df = df.dropna(subset=[label_col]).copy()
    df = df.drop_duplicates()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in num_cols:
        num_cols.remove(label_col)

    # Cast object representations to numeric if corrupted during raw export
    for c in df.columns:
        if c != label_col and c not in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
    num_cols = [c for c in df.columns if c != label_col]

    # Replace inf and -inf with NaN
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

    # Impute missing values with column median
    for col in num_cols:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    # Drop zero-variance columns (constant values across all flows)
    zero_var_cols = [c for c in num_cols if df[c].std() == 0]
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)

    return df