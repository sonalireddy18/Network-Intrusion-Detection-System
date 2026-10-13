"""
anomaly_preparation.py - Unsupervised & Held-Out Attack Extraction
Primary Owner: Member 3 (Isolation Forest Lead)
"""
from typing import Tuple, Dict, Any, Optional
import pandas as pd


def isolate_held_out_attack(
    df: pd.DataFrame,
    label_col: str,
    held_out_attack: str = "PortScan"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Completely separates the held-out attack prior to training/testing splits (Objective 4).
    Ensures baseline models never encounter this attack family during training.
    """
    labels_clean = df[label_col].astype(str).str.strip()
    match_mask = labels_clean.str.lower() == held_out_attack.strip().lower()
    
    if match_mask.sum() == 0:
        print(f"[AnomalyPrep] Warning: Held-out attack '{held_out_attack}' not found.")
        return df.copy(), None

    df_heldout = df[match_mask].copy()
    df_remaining = df[~match_mask].copy()

    print(f"[AnomalyPrep] Isolated {len(df_heldout):,} samples of '{held_out_attack}' for zero-day emulation.")
    print(f"[AnomalyPrep] Retained {len(df_remaining):,} records for standard splits.")
    return df_remaining, df_heldout


def extract_pure_benign(
    X_train: pd.DataFrame,
    y_train_raw: pd.Series,
    normal_label: str = "BENIGN"
) -> pd.DataFrame:
    """Extracts strictly normal flows for training Isolation Forest & Autoencoder."""
    benign_mask = y_train_raw.astype(str).str.strip() == normal_label
    X_benign = X_train.loc[benign_mask].copy()
    print(f"[AnomalyPrep] Extracted {len(X_benign):,} benign flows for anomaly baseline training.")
    return X_benign


def make_binary_anomaly_labels(
    y_series_raw: pd.Series, 
    normal_label: str = "BENIGN"
) -> pd.Series:
    """Maps multi-class target labels to binary: 0 = Benign, 1 = Attack/Anomaly."""
    is_anomaly = (y_series_raw.astype(str).str.strip() != normal_label).astype(int)
    is_anomaly.name = "Anomaly"
    return is_anomaly