import json
from pathlib import Path
from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_feature_columns(df: pd.DataFrame, label_col: str) -> List[str]:
    """Retrieve all numeric feature columns excluding the target."""
    return [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != label_col]


def get_log_columns(X_train: pd.DataFrame) -> List[str]:
    """Columns safe for log1p: non-negative in train (skips e.g. Init_Win_bytes_* which contain -1)."""
    return [c for c in X_train.columns if X_train[c].min() >= 0]


def apply_log1p(X: pd.DataFrame, log_cols: List[str]) -> pd.DataFrame:
    """log1p on heavy-tailed columns. Lower-clips at 0 so unseen negatives can't produce NaN."""
    X = X.copy()
    X[log_cols] = np.log1p(X[log_cols].clip(lower=0))
    return X


def _transform(scaler: MinMaxScaler, X: pd.DataFrame, clip: bool) -> pd.DataFrame:
    arr = scaler.transform(X)
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return pd.DataFrame(arr.astype("float32"), columns=X.columns, index=X.index)


def fit_and_scale_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_save_path: Optional[Path] = None,
    features_save_path: Optional[Path] = None,
    clip: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Fits MinMaxScaler strictly on X_train to prevent leakage."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)

    X_train_scaled = _transform(scaler, X_train, clip)
    X_val_scaled = _transform(scaler, X_val, clip)
    X_test_scaled = _transform(scaler, X_test, clip)

    feature_cols = X_train.columns.tolist()

    if scaler_save_path:
        scaler_save_path = Path(scaler_save_path)
        scaler_save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"[Scaling] Scaler saved to: {scaler_save_path}")

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
    feature_cols: List[str],
    clip: bool = True,
) -> pd.DataFrame:
    """Transforms unseen evaluation subsets (e.g., held-out attack) using the fitted scaler."""
    return _transform(scaler, X_df[feature_cols], clip)