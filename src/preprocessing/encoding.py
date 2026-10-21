import json
from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd


def standardize_labels(df: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
    """Fix corrupted dashes (e.g. 'Web Attack � XSS') and normalize whitespace."""
    df = df.copy()
    df[label_column] = (
        df[label_column]
        .astype("string")
        .str.replace(r"\s*[^\x20-\x7E]+\s*", " - ", regex=True)  # only non-ASCII chars -> " - "
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df.dropna(subset=[label_column])


def merge_web_attacks(df: pd.DataFrame, label_column: str = "Label") -> pd.DataFrame:
    """Merge Brute Force / XSS / Sql Injection into one 'Web Attack' class (rare-class fix)."""
    df = df.copy()
    is_web = df[label_column].str.startswith("Web Attack", na=False)
    df.loc[is_web, label_column] = "Web Attack"
    return df


def encode_labels(
    y: pd.Series,
    mapping_save_path: Optional[Path] = None
) -> Tuple[pd.Series, Dict[str, int]]:
    """Encodes attack labels into integer class IDs, setting BENIGN as 0."""
    y = y.astype("string").str.strip()
    unique_classes = sorted(y.unique().tolist())

    if "BENIGN" in unique_classes:
        unique_classes.remove("BENIGN")
        ordered_classes = ["BENIGN"] + unique_classes
    else:
        ordered_classes = unique_classes

    label_to_idx = {label: idx for idx, label in enumerate(ordered_classes)}
    encoded_y = y.map(label_to_idx).astype("int32")

    if mapping_save_path:
        mapping_path = Path(mapping_save_path)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_to_idx, f, indent=4)
        print(f"[Encoding] Saved class mapping ({len(label_to_idx)} classes) to: {mapping_path}")

    return encoded_y, label_to_idx