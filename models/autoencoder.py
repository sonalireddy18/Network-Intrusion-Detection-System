"""
models/autoencoder.py - Deep Autoencoder training pipeline (Base & Log1p variants)
Owner: Member 4 (Deep Autoencoder Lead)
Location: models/autoencoder.py
Framework: TensorFlow / Keras

Run:
    python models/autoencoder.py                    # both variants, seed 42
    python models/autoencoder.py --seeds 42 43 44   # + extra seeds for mean +/- std
    python models/autoencoder.py --variants base    # only one variant

Design rules (match the team data-usage guide):
  * Train on benign-only flows (X_train_anomaly_benign).
  * Tune latent size + threshold on VALIDATION only. Test set is never touched here.
  * Never mix files across the base / log1p folders.
  * Anomaly score = per-flow MSE reconstruction error (higher = more anomalous).
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_auc_score
from tensorflow.keras import callbacks, layers, models

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
BASE_SEED = 42
try:
    tf.config.experimental.enable_op_determinism()
except Exception:  # older TF versions
    pass


def set_seed(seed: int) -> None:
    """Seeds python, numpy and TensorFlow in one call."""
    tf.keras.utils.set_random_seed(seed)


set_seed(BASE_SEED)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODELS_PARENT_DIR = Path(__file__).resolve().parent
ROOT = MODELS_PARENT_DIR.parent

DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = MODELS_PARENT_DIR / "autoencoder"
RESULTS_DIR = ROOT / "results" / "autoencoder"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_LATENT_DIMS = [8, 16, 32]
MAX_EPOCHS = 50  # early stopping decides the real length
BATCH_SIZE = 512


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_autoencoder(input_dim: int, latent_dim: int = 16) -> models.Model:
    """
    Symmetric deep autoencoder. Sigmoid output assumes features are in [0, 1]
    (MinMax scaled). Latent layer is LINEAR (avoids dead ReLU units in the bottleneck).
    """
    inputs = layers.Input(shape=(input_dim,), name="flow_input")

    # Encoder
    x = layers.Dense(64, activation="relu", name="enc_1")(inputs)
    x = layers.BatchNormalization(name="enc_bn_1")(x)
    x = layers.Dropout(0.05, name="enc_drop_1")(x)
    x = layers.Dense(32, activation="relu", name="enc_2")(x)
    x = layers.BatchNormalization(name="enc_bn_2")(x)

    # Bottleneck (linear)
    latent = layers.Dense(latent_dim, activation="linear", name="latent_bottleneck")(x)

    # Decoder
    x = layers.Dense(32, activation="relu", name="dec_1")(latent)
    x = layers.BatchNormalization(name="dec_bn_1")(x)
    x = layers.Dense(64, activation="relu", name="dec_2")(x)
    x = layers.BatchNormalization(name="dec_bn_2")(x)

    outputs = layers.Dense(input_dim, activation="sigmoid", name="reconstruction")(x)

    ae = models.Model(inputs=inputs, outputs=outputs, name=f"Autoencoder_latent_{latent_dim}")
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return ae


def compute_reconstruction_error(model: models.Model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Per-flow MSE reconstruction error (the anomaly score)."""
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def calibrate_threshold(val_errors: np.ndarray, y_val: np.ndarray):
    """
    Picks the threshold that maximises F1 on VALIDATION data.
    NOTE: this uses attack labels from validation => semi-supervised calibration.
    precision/recall have one more entry than thresholds, so the last point is dropped.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_errors)
    p, r = precisions[:-1], recalls[:-1]
    f1 = (2 * p * r) / (p + r + 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


# --------------------------------------------------------------------------- #
# Data loading + sanity checks
# --------------------------------------------------------------------------- #
def _row_hashes(arr: np.ndarray) -> set:
    return set(pd.util.hash_pandas_object(pd.DataFrame(arr), index=False).tolist())


def load_variant_data(variant_name: str, data_path: Path):
    X_train = pd.read_parquet(data_path / "X_train_anomaly_benign.parquet").to_numpy(dtype=np.float32)
    X_val = pd.read_parquet(data_path / "X_val_anomaly.parquet").to_numpy(dtype=np.float32)
    y_val = pd.read_parquet(data_path / "y_val_anomaly_binary.parquet").squeeze().to_numpy(dtype=np.int8)

    heldout_file = data_path / "X_heldout_portscan.parquet"
    X_heldout = pd.read_parquet(heldout_file).to_numpy(dtype=np.float32) if heldout_file.exists() else None

    # Shuffle ONCE so Keras' validation_split (which takes the LAST 15%) is not an ordered slice.
    rng = np.random.default_rng(BASE_SEED)
    X_train = X_train[rng.permutation(len(X_train))]

    # Sanity check 1: sigmoid output needs features in [0, 1]
    tmin, tmax = float(X_train.min()), float(X_train.max())
    print(f"[i] Train feature range: [{tmin:.4f}, {tmax:.4f}] | Val max: {float(X_val.max()):.4f}")
    if tmin < -1e-6 or tmax > 1 + 1e-6:
        print("[!] WARNING: training features are outside [0,1]. Sigmoid output is NOT appropriate; "
              "check that MinMax scaling is the LAST step (after log1p for the log1p variant).")
    if float(X_val.max()) > 1.0:
        print("[i] Val values exceed 1 (scaler fit on train). Expected, but they cannot be reconstructed exactly.")

    # Sanity check 2: no held-out PortScan rows inside the validation set
    if X_heldout is not None:
        overlap = len(_row_hashes(X_val) & _row_hashes(X_heldout))
        print(f"[i] Held-out PortScan rows: {len(X_heldout):,} | identical rows also in val: {overlap:,}")
        if overlap > 0:
            print("[!] WARNING: some held-out rows also appear in the validation set. "
                  "The threshold may have seen PortScan; the held-out result could be inflated. "
                  "(Identical benign/attack feature vectors can also cause small overlaps.)")
    return X_train, X_val, y_val, X_heldout


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def fit_model(X_train: np.ndarray, latent_dim: int, seed: int):
    """Seeds, builds and fits one autoencoder. Returns (model, history_dict, train_time_sec)."""
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_autoencoder(X_train.shape[1], latent_dim)
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0),
    ]
    t0 = time.time()
    history = model.fit(
        X_train, X_train,
        validation_split=0.15,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        shuffle=True,
        verbose=0,
    )
    return model, history.history, time.time() - t0


def save_artifacts(model, meta: dict, variant: str, suffix: str = ""):
    model.save(MODEL_DIR / f"best_autoencoder_{variant}{suffix}.keras")
    with open(MODEL_DIR / f"autoencoder_metadata_{variant}{suffix}.json", "w") as f:
        json.dump(meta, f, indent=4)


def plot_loss(history: dict, variant: str, latent: int):
    plt.figure(figsize=(7, 4))
    plt.plot(history["loss"], label="Train Loss (MSE)")
    plt.plot(history["val_loss"], label="Val Loss (MSE)")
    plt.title(f"Autoencoder Reconstruction Loss ({variant.upper()} | Latent = {latent})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"training_loss_curve_{variant}.png", dpi=200)
    plt.close()


def build_metadata(model, variant, seed, latent, input_dim, threshold, val_auc, val_f1,
                   train_time, history, X_train):
    train_err = compute_reconstruction_error(model, X_train)
    return {
        "variant": variant,
        "seed": int(seed),
        "input_dim": int(input_dim),
        "best_latent_dim": int(latent),
        "calibrated_threshold": float(threshold),
        "val_roc_auc": float(val_auc),
        "val_f1": float(val_f1),
        "train_time_sec": float(train_time),
        "epochs_run": int(len(history["loss"])),
        "max_epochs": MAX_EPOCHS,
        "final_val_loss": float(history["val_loss"][-1]),
        "param_count": int(model.count_params()),
        # benign-train error percentiles, used by metrics.py for threshold-sensitivity checks
        "benign_train_error_p95": float(np.percentile(train_err, 95)),
        "benign_train_error_p99": float(np.percentile(train_err, 99)),
    }


# --------------------------------------------------------------------------- #
# Main per-variant routine
# --------------------------------------------------------------------------- #
def train_single_variant(variant_name: str, data_path: Path, seeds):
    print("\n" + "=" * 70)
    print(f"[*] TRAINING AUTOENCODER VARIANT: [{variant_name.upper()}]")
    print(f"[*] Reading data from: {data_path}")
    print("=" * 70)

    X_train, X_val, y_val, _ = load_variant_data(variant_name, data_path)
    input_dim = X_train.shape[1]
    print(f"[+] Benign training flows: {len(X_train):,} ({input_dim} features) | Val flows: {len(X_val):,}")

    # ---- 1. Hyperparameter search (latent size) on VALIDATION only ---------
    print(f"[*] Tuning latent dimension {CANDIDATE_LATENT_DIMS} (seed {BASE_SEED})...")
    best = {"auc": -1.0}
    records = []
    for l_dim in CANDIDATE_LATENT_DIMS:
        model, hist, t_train = fit_model(X_train, l_dim, BASE_SEED)
        val_err = compute_reconstruction_error(model, X_val)
        val_auc = roc_auc_score(y_val, val_err)
        thr, val_f1 = calibrate_threshold(val_err, y_val)
        epochs_run = len(hist["loss"])
        print(f"    Latent {l_dim:2d} -> Val ROC-AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | "
              f"Epochs: {epochs_run}/{MAX_EPOCHS} | Time: {t_train:.1f}s")
        if epochs_run >= MAX_EPOCHS:
            print(f"    [!] Latent {l_dim} hit the epoch ceiling; consider raising MAX_EPOCHS.")
        records.append({
            "variant": variant_name, "latent_dim": l_dim,
            "val_roc_auc": float(val_auc), "val_f1": float(val_f1),
            "calibrated_threshold": float(thr), "train_time_sec": float(t_train),
            "epochs_run": epochs_run, "param_count": int(model.count_params()),
        })
        if val_auc > best["auc"]:
            best = {"auc": val_auc, "f1": val_f1, "thr": thr, "latent": l_dim,
                    "time": t_train, "hist": hist, "weights": model.get_weights()}

    pd.DataFrame(records).to_csv(RESULTS_DIR / f"hyperparameter_tuning_{variant_name}.csv", index=False)
    print(f"[✓] Best latent dimension: {best['latent']} "
          f"(Val ROC-AUC {best['auc']:.4f}, Val F1 {best['f1']:.4f})")

    # ---- 2. Rebuild best model from stored weights, save as the MAIN model --
    tf.keras.backend.clear_session()
    set_seed(BASE_SEED)
    main_model = build_autoencoder(input_dim, best["latent"])
    main_model.set_weights(best["weights"])
    meta = build_metadata(main_model, variant_name, BASE_SEED, best["latent"], input_dim,
                          best["thr"], best["auc"], best["f1"], best["time"], best["hist"], X_train)
    save_artifacts(main_model, meta, variant_name)
    plot_loss(best["hist"], variant_name, best["latent"])
    print(f"[+] Saved main model: {MODEL_DIR / f'best_autoencoder_{variant_name}.keras'}")

    # ---- 3. Extra seeds (same latent dim) for mean +/- std ------------------
    for s in [s for s in seeds if s != BASE_SEED]:
        model, hist, t_train = fit_model(X_train, best["latent"], s)
        val_err = compute_reconstruction_error(model, X_val)
        val_auc = roc_auc_score(y_val, val_err)
        thr, val_f1 = calibrate_threshold(val_err, y_val)
        m = build_metadata(model, variant_name, s, best["latent"], input_dim,
                           thr, val_auc, val_f1, t_train, hist, X_train)
        save_artifacts(model, m, variant_name, suffix=f"_seed{s}")
        print(f"[+] Seed {s}: Val ROC-AUC {val_auc:.4f} | Val F1 {val_f1:.4f} | saved")


def run_pipeline(variants, seeds):
    print("=" * 70)
    print("PHASE 4: TRAINING BASE & LOG1P AUTOENCODER MODELS (MEMBER 4)")
    print(f"Project Root:        {ROOT}")
    print(f"Model Destination:   {MODEL_DIR}")
    print(f"Results Destination: {RESULTS_DIR}")
    print(f"Seeds:               {seeds}")
    print("=" * 70)

    if "base" in variants:
        train_single_variant("base", DATA_PROCESSED_DIR, seeds)

    if "log1p" in variants:
        log1p_dir = DATA_PROCESSED_DIR / "log1p"
        if log1p_dir.exists():
            train_single_variant("log1p", log1p_dir, seeds)
        else:
            print(f"\n[!] {log1p_dir} not found. Skipping log1p training.")

    print("\n[✓] Autoencoder training complete. Next: python metrics.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["base", "log1p"], choices=["base", "log1p"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[BASE_SEED],
                    help="Seed 42 is always the main model; extra seeds are saved for variance reporting.")
    args = ap.parse_args()
    run_pipeline(args.variants, args.seeds)