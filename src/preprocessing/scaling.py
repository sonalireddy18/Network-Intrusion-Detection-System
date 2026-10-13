"""
scaling.py - Feature Normalization & Serialization
Primary Owner: Member 4 (Deep Autoencoder Lead)
"""
import json
from pathlib import Path
from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_feature_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    """Retrieve all numeric feature columns, excluding the target."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c != label_col]


def fit_and_scale_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_save_path: Optional[Path] = None,
    features_save_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Fits MinMaxScaler strictly on X_train to prevent leakage.
    Scales and clips features to [0, 1] for neural network stability.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit exclusively on training data
    scaler.fit(X_train)
    
    feature_cols = X_train.columns.tolist()

    X_train_scaled = pd.DataFrame(
        np.clip(scaler.transform(X_train), 0.0, 1.0),
        columns=feature_cols,
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        np.clip(scaler.transform(X_val), 0.0, 1.0),
        columns=feature_cols,
        index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        np.clip(scaler.transform(X_test), 0.0, 1.0),
        columns=feature_cols,
        index=X_test.index
    )

    if scaler_save_path:
        scaler_save_path = Path(scaler_save_path)
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"[Scaling] Fitted scaler saved to: {scaler_save_path}")

    if features_save_path:
        features_save_path = Path(features_save_path)
        features_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(features_save_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=4)
        print(f"[Scaling] Feature list ({len(feature_cols)} features) saved to: {features_save_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def scale_external_data(
    X_df: pd.DataFrame,
    scaler: MinMaxScaler,
    feature_cols: List[str]
) -> pd.DataFrame:
    """Transforms unseen evaluation subsets (e.g., held-out attack) using the pre-fitted scaler."""
    scaled = np.clip(scaler.transform(X_df[feature_cols]), 0.0, 1.0)
    return pd.DataFrame(scaled, columns=feature_cols, index=X_df.index)