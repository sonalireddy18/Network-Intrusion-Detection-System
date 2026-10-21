from typing import Tuple, Optional
import pandas as pd


def isolate_held_out_attack(
    df: pd.DataFrame,
    label_col: str,
    held_out_attack: str = "PortScan"
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Isolates the held-out attack prior to any training splits (Objective 4)."""
    labels_clean = df[label_col].astype(str).str.strip().str.lower()
    target_clean = held_out_attack.strip().lower()

    heldout_mask = labels_clean == target_clean
    if heldout_mask.sum() == 0:
        print(f"[AnomalyPrep] Warning: '{held_out_attack}' not found.")
        return df.copy(), None

    df_heldout = df.loc[heldout_mask].copy()
    df_remaining = df.loc[~heldout_mask].copy()
    print(f"[AnomalyPrep] Isolated {len(df_heldout):,} samples of '{held_out_attack}' for zero-day evaluation.")
    return df_remaining, df_heldout


def extract_pure_benign(
    X_train: pd.DataFrame,
    y_raw_train: pd.Series,
    normal_label: str = "BENIGN"
) -> pd.DataFrame:
    """Extracts strictly normal flows for baseline training (Isolation Forest & Autoencoder)."""
    benign_mask = y_raw_train.astype(str).str.strip() == normal_label
    X_train_benign = X_train.loc[benign_mask].copy()
    print(f"[AnomalyPrep] Pure benign training set size: {len(X_train_benign):,} flows.")
    return X_train_benign


def make_binary_anomaly_labels(
    y_raw: pd.Series,
    normal_label: str = "BENIGN"
) -> pd.Series:
    """Maps multiclass labels to binary: 0 = Normal, 1 = Attack."""
    is_anomaly = (y_raw.astype(str).str.strip() != normal_label).astype("int8")
    is_anomaly.name = "Anomaly"
    return is_anomaly
