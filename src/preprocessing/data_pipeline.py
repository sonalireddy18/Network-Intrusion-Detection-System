import json
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
from src.preprocessing.encoding import standardize_labels, merge_web_attacks, encode_labels
from src.preprocessing.anomaly_preparation import (
    isolate_held_out_attack,
    extract_pure_benign,
    make_binary_anomaly_labels,
)
from src.preprocessing.scaling import (
    get_feature_columns,
    get_log_columns,
    apply_log1p,
    fit_and_scale_splits,
    scale_external_data,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TWIN_CACHE = PROCESSED_DIR / "twin_drop_index.json"


def _row_hashes(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """64-bit hash of each row's features + label."""
    return pd.util.hash_pandas_object(X.assign(_y=y.to_numpy()), index=False)


def remove_cross_split_twins(
    X_train, y_train, X_val, y_val, X_test, y_test,
    cache_path: Path, reuse_cache: bool,
):
    """
    Drop val/test rows whose (features AND label) already appear in an earlier split.
    Rows with identical features but a DIFFERENT label are kept (genuine label noise).

    reuse_cache=False -> detect twins and save the dropped indices to cache_path.
    reuse_cache=True  -> reuse those indices so every variant has identical val/test rows.
    """
    if reuse_cache:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"{cache_path} not found. Run the non-log1p variant first."
            )
        with open(cache_path, "r", encoding="utf-8") as f:
            drop = json.load(f)
        val_keep = ~X_val.index.isin(drop["val"])
        test_keep = ~X_test.index.isin(drop["test"])
    else:
        ref = set(_row_hashes(X_train, y_train))
        val_keep = ~_row_hashes(X_val, y_val).isin(ref).to_numpy()
        ref |= set(_row_hashes(X_val[val_keep], y_val[val_keep]))
        test_keep = ~_row_hashes(X_test, y_test).isin(ref).to_numpy()
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"val": X_val.index[~val_keep].tolist(),
                 "test": X_test.index[~test_keep].tolist()},
                f,
            )

    n_val, n_test = int((~val_keep).sum()), int((~test_keep).sum())
    return (
        X_val[val_keep], y_val[val_keep],
        X_test[test_keep], y_test[test_keep],
        n_val, n_test,
    )


def run_pipeline(
    held_out_attack: str = "PortScan",
    sample_fraction: float = None,
    merge_web: bool = True,
    use_log1p: bool = False,
    clip: bool = True,
):
    """
    use_log1p=False -> data/processed/          (RF, XGBoost, Isolation Forest, baseline AE)
    use_log1p=True  -> data/processed/log1p/    (anomaly-model ablation)
    Run the non-log1p variant FIRST: it creates twin_drop_index.json used by the log1p variant.
    """
    start_time = time.time()
    out_dir = PROCESSED_DIR / "log1p" if use_log1p else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {RAW_DIR}. Place raw files first.")

    print(f"[*] Found {len(csv_files)} CSV files. Loading and sanitizing...")
    frames = [
        clean_column_names(pd.read_csv(f, low_memory=False, encoding="cp1252"))
        for f in csv_files
    ]
    df = pd.concat(frames, ignore_index=True)
    del frames
    label_col = find_label_column(df.columns)

    # 1. Clean & Sanitize
    print("[*] Sanitizing labels, removing leakage headers and handling infs/NaNs...")
    df = standardize_labels(df, label_col)
    if merge_web:
        df = merge_web_attacks(df, label_col)
    df = drop_leakage_identifiers(df)
    df = sanitize_numeric_values(df, label_col)

    # float32 halves memory, but rounding can create new duplicates -> dedup again
    feat_cols_all = get_feature_columns(df, label_col)
    df[feat_cols_all] = df[feat_cols_all].astype("float32")
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"[+] Removed {n_before - len(df):,} duplicates created by float32 rounding")

    if sample_fraction and 0.0 < sample_fraction < 1.0:
        print(f"[*] Subsampling {sample_fraction * 100:.1f}% stratified by class...")
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=42)
        )

    print(f"[+] Total cleaned records: {len(df):,}")

    # 2. Extract Zero-Day / Held-Out Attack
    df_main, df_heldout = isolate_held_out_attack(df, label_col=label_col, held_out_attack=held_out_attack)
    del df

    # 3. Supervised Encoding
    feature_names = get_feature_columns(df_main, label_col)
    X = df_main[feature_names]
    y_raw = df_main[label_col]

    y_supervised, _ = encode_labels(y_raw, mapping_save_path=out_dir / "label_mapping.json")

    counts = y_supervised.value_counts()
    too_small = counts[counts < 3]
    if len(too_small):
        raise ValueError(f"Classes with <3 samples cannot be stratified: {too_small.to_dict()}")

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

    # 4b. Optional log1p on skewed columns (columns chosen from TRAIN only)
    X_heldout = df_heldout[feature_names] if df_heldout is not None and not df_heldout.empty else None
    if use_log1p:
        log_cols = get_log_columns(X_train)
        print(f"[*] Applying log1p to {len(log_cols)}/{len(feature_names)} columns...")
        X_train = apply_log1p(X_train, log_cols)
        X_val = apply_log1p(X_val, log_cols)
        X_test = apply_log1p(X_test, log_cols)
        if X_heldout is not None:
            X_heldout = apply_log1p(X_heldout, log_cols)
        with open(out_dir / "log1p_columns.json", "w", encoding="utf-8") as f:
            json.dump(log_cols, f, indent=4)

    # 5. Leak-Free Scaling
    print("[*] Scaling features to [0, 1] range...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = fit_and_scale_splits(
        X_train, X_val, X_test,
        scaler_save_path=out_dir / "scaler.joblib",
        features_save_path=out_dir / "feature_names.json",
        clip=clip,
    )

    # 5b. Remove val/test rows identical (features AND label) to rows in an earlier split
    X_val_scaled, y_val, X_test_scaled, y_test, n_val, n_test = remove_cross_split_twins(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test,
        cache_path=TWIN_CACHE, reuse_cache=use_log1p,
    )
    y_raw_val = y_raw_val.loc[y_val.index]
    y_raw_test = y_raw_test.loc[y_test.index]
    print(f"[+] Dropped {n_val:,} val and {n_test:,} test rows that duplicate earlier splits")

    # 6. Save Supervised Splits (RF & XGBoost)
    print("[*] Saving Supervised Parquet files...")
    X_train_scaled.to_parquet(out_dir / "X_train_supervised.parquet")
    y_train.to_frame(name="Label").to_parquet(out_dir / "y_train_supervised.parquet")
    X_val_scaled.to_parquet(out_dir / "X_val_supervised.parquet")
    y_val.to_frame(name="Label").to_parquet(out_dir / "y_val_supervised.parquet")
    X_test_scaled.to_parquet(out_dir / "X_test_supervised.parquet")
    y_test.to_frame(name="Label").to_parquet(out_dir / "y_test_supervised.parquet")

    # 7. Save Anomaly Detection Splits (Isolation Forest & Autoencoder)
    print("[*] Saving Anomaly Detection Parquet files...")
    X_train_benign = extract_pure_benign(X_train_scaled, y_raw_train, normal_label="BENIGN")
    X_train_benign.to_parquet(out_dir / "X_train_anomaly_benign.parquet")

    X_val_scaled.to_parquet(out_dir / "X_val_anomaly.parquet")
    make_binary_anomaly_labels(y_raw_val, "BENIGN").to_frame().to_parquet(out_dir / "y_val_anomaly_binary.parquet")

    X_test_scaled.to_parquet(out_dir / "X_test_anomaly.parquet")
    make_binary_anomaly_labels(y_raw_test, "BENIGN").to_frame().to_parquet(out_dir / "y_test_anomaly_binary.parquet")

    # 8. Transform & Save Held-Out Evaluation Set
    if X_heldout is not None:
        print(f"[*] Scaling held-out subset ({held_out_attack})...")
        X_heldout_scaled = scale_external_data(X_heldout, scaler, feature_names, clip=clip)
        y_heldout = pd.Series(1, index=X_heldout.index, name="Anomaly")

        X_heldout_scaled.to_parquet(out_dir / f"X_heldout_{held_out_attack.lower()}.parquet")
        y_heldout.to_frame().to_parquet(out_dir / f"y_heldout_{held_out_attack.lower()}.parquet")

    elapsed = time.time() - start_time
    print(f"\n[✓] Preprocessing completed in {elapsed:.1f} seconds.")
    print(f"[✓] Artifacts created in: {out_dir.resolve()}")


if __name__ == "__main__":
    # Order matters: the base run creates twin_drop_index.json for the log1p run.
    run_pipeline(held_out_attack="PortScan", merge_web=True, use_log1p=False)
    run_pipeline(held_out_attack="PortScan", merge_web=True, use_log1p=True)