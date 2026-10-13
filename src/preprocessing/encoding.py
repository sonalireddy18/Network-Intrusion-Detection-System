"""
encoding.py - Target Normalization & Deterministic Label Encoding
Primary Owner: Member 2 (XGBoost Lead)
"""
import json
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd


def standardize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Normalize label strings by trimming whitespace and unifying hyphens."""
    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.replace("", "-", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return df


def encode_labels(
    y: pd.Series, 
    mapping_save_path: Path = None
) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Encodes multiclass labels to integers, enforcing 'BENIGN' as 0.
    Persists the label map to JSON for downstream evaluation and Streamlit.
    """
    unique_classes = sorted(y.unique().tolist())
    
    # Guarantee BENIGN receives class ID 0
    if "BENIGN" in unique_classes:
        unique_classes.remove("BENIGN")
        ordered_classes = ["BENIGN"] + unique_classes
    else:
        ordered_classes = unique_classes

    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(ordered_classes)}
    encoded_y = y.map(label_to_idx).astype(int)

    if mapping_save_path:
        mapping_save_path = Path(mapping_save_path)
        mapping_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_save_path, "w", encoding="utf-8") as f:
            json.dump(label_to_idx, f, indent=4)
        print(f"[Encoding] Saved class label mapping to: {mapping_save_path}")

    return encoded_y, label_to_idx