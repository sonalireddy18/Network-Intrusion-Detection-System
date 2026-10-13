"""
data_pipeline.py - Master Pipeline Execution
Orchestrates cleaning, encoding, splitting, anomaly isolation, and scaling.
Run from root: python -m src.preprocessing.data_pipeline
"""
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.cleaning import (
    clean_column_names,
    find_label_column,
    drop_leakage_identifiers,
    sanitize_numeric_values,
)
from src.preprocessing.encoding import standardize_labels, encode_labels
from src.preprocessing.anomaly_preparation import (
    isolate_held_out_attack,
    extract_pure_benign,
    make_binary_anomaly_labels,
)
from src.preprocessing.scaling import (
    get_feature_columns,
    fit_and_scale_splits,
    scale_external_data,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def run_pipeline(held_out_attack: str = "PortScan", sample_fraction: float = None):
    start_time = time.time()
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {RAW_DIR}. Place raw files first.")

    print(f"[*] Found {len(csv_files)} CSV files. Loading and sanitizing...")
    frames = []
    for f in csv_files:
        print(f"    - Loading: {f.name}")
        df_part = pd.read_csv(f, low_memory=False)
        df_part = clean_column_names(df_part)
        frames.append(df_part)

    df = pd.concat(frames, ignore_index=True)
    label_col = find_label_column(df.columns)
    
    # 1. Clean & Sanitize
    print("[*] Sanitizing strings, removing leakage headers and handling infs/NaNs...")
    df = standardize_labels(df, label_col)
    df = drop_leakage_identifiers(df)
    df = sanitize_numeric_values(df, label_col)
    
    # Optional subsampling for development on resource-constrained machines
    if sample_fraction and 0.0 < sample_fraction < 1.0:
        print(f"[*] Subsampling {sample_fraction * 100:.1f}% stratified by class...")
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=42)
        )

    print(f"[+] Total cleaned records: {len(df):,}")

    # 2. Extract Zero-Day / Held-Out Attack (Objective 4)
    df_main, df_heldout = isolate_held_out_attack(
        df, label_col=label_col, held_out_attack=held_out_attack
    )

    # 3. Supervised Encoding
    feature_names = get_feature_columns(df_main, label_col)
    X = df_main[feature_names]
    y_raw = df_main[label_col]

    y_supervised, _ = encode_labels(
        y_raw, mapping_save_path=PROCESSED_DIR / "label_mapping.json"
    )

    # 4. Stratified Splits (70% Train, 15% Validation, 15% Test)
    print("[*] Generating stratified train/val/test splits...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_supervised, test_size=0.30, random_state=42, stratify=y_supervised
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    y_raw_train = y_raw.loc[X_train.index]
    y_raw_val = y_raw.loc[X_val.index]
    y_raw_test = y_raw.loc[X_test.index]

    # 5. Leak-Free Scaling
    print("[*] Scaling features to [0, 1] bounded range...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fit_and_scale_splits(
        X_train,
        X_val,
        X_test,
        scaler_save_path=PROCESSED_DIR / "scaler.joblib",
        features_save_path=PROCESSED_DIR / "feature_names.json",
    )

    # 6. Save Supervised Splits (M1 & M2: RF & XGBoost)
    print("[*] Saving Supervised Parquet files...")
    X_train_scaled.to_parquet(PROCESSED_DIR / "X_train_supervised.parquet")
    y_train.to_frame(name="Label").to_parquet(PROCESSED_DIR / "y_train_supervised.parquet")
    X_val_scaled.to_parquet(PROCESSED_DIR / "X_val_supervised.parquet")
    y_val.to_frame(name="Label").to_parquet(PROCESSED_DIR / "y_val_supervised.parquet")
    X_test_scaled.to_parquet(PROCESSED_DIR / "X_test_supervised.parquet")
    y_test.to_frame(name="Label").to_parquet(PROCESSED_DIR / "y_test_supervised.parquet")

    # 7. Save Anomaly Detection Splits (M3 & M4: Isolation Forest & Autoencoder)
    print("[*] Saving Anomaly Detection Parquet files...")
    X_train_benign = extract_pure_benign(X_train_scaled, y_raw_train, normal_label="BENIGN")
    X_train_benign.to_parquet(PROCESSED_DIR / "X_train_anomaly_benign.parquet")

    X_val_scaled.to_parquet(PROCESSED_DIR / "X_val_anomaly.parquet")
    y_val_binary = make_binary_anomaly_labels(y_raw_val, normal_label="BENIGN")
    y_val_binary.to_frame().to_parquet(PROCESSED_DIR / "y_val_anomaly_binary.parquet")

    X_test_scaled.to_parquet(PROCESSED_DIR / "X_test_anomaly.parquet")
    y_test_binary = make_binary_anomaly_labels(y_raw_test, normal_label="BENIGN")
    y_test_binary.to_frame().to_parquet(PROCESSED_DIR / "y_test_anomaly_binary.parquet")

    # 8. Transform & Save Held-Out Evaluation Set
    if df_heldout is not None and not df_heldout.empty:
        print(f"[*] Scaling held-out subset ({held_out_attack})...")
        X_heldout = df_heldout[feature_names]
        X_heldout_scaled = scale_external_data(X_heldout, scaler, feature_names)
        y_heldout = pd.Series(1, index=df_heldout.index, name="Anomaly")

        X_heldout_scaled.to_parquet(PROCESSED_DIR / f"X_heldout_{held_out_attack.lower()}.parquet")
        y_heldout.to_frame().to_parquet(PROCESSED_DIR / f"y_heldout_{held_out_attack.lower()}.parquet")

    elapsed = time.time() - start_time
    print(f"\n[✓] Phase 3 preprocessing completed successfully in {elapsed:.1f} seconds!")
    print(f"[✓] Artifacts created in: {PROCESSED_DIR.resolve()}")


if __name__ == "__main__":
    # If system RAM is under 16 GB, pass sample_fraction=0.3 to run on 30% of data
    run_pipeline(held_out_attack="PortScan", sample_fraction=None)