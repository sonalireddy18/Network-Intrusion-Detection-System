"""
src/models/autoencoder.py - Deep Autoencoder Training & Evaluation Pipeline
Primary Owner: Member 4 (Deep Learning Anomaly Lead)
Framework: TensorFlow / Keras

Key Workflow:
  1. Train unsupervised on pure normal traffic (X_train_anomaly_benign.parquet).
  2. Tune bottleneck latent dimensions (8, 16, 32) on validation data.
  3. Calibrate decision threshold (tau) using F1-score optimization on X_val_anomaly.
  4. Benchmark test performance (ROC-AUC, PR-AUC, F1, FPR, Inference Latency).
  5. Evaluate Objective 4: Held-out unseen attack detection (X_heldout_portscan.parquet).
  6. Save weights to models/autoencoder/ and reports to results/autoencoder/.
"""

import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# File Paths (ROOT is hybrid-nids/)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "autoencoder"       # Root-level models directory
RESULTS_DIR = ROOT / "results" / "autoencoder"   # Root-level results directory

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_autoencoder(input_dim: int, latent_dim: int = 16) -> models.Model:
    """
    Constructs a symmetric Deep Autoencoder architecture.
    Output activation is 'sigmoid' to match [0, 1] MinMaxScaler features.
    """
    inputs = layers.Input(shape=(input_dim,), name="flow_input")

    # Encoder
    encoded = layers.Dense(64, activation="relu", name="enc_dense_1")(inputs)
    encoded = layers.BatchNormalization(name="enc_bn_1")(encoded)
    encoded = layers.Dropout(0.05, name="enc_dropout_1")(encoded)

    encoded = layers.Dense(32, activation="relu", name="enc_dense_2")(encoded)
    encoded = layers.BatchNormalization(name="enc_bn_2")(encoded)

    # Bottleneck Latent Space
    latent = layers.Dense(latent_dim, activation="relu", name="bottleneck_latent")(encoded)

    # Decoder
    decoded = layers.Dense(32, activation="relu", name="dec_dense_1")(latent)
    decoded = layers.BatchNormalization(name="dec_bn_1")(decoded)

    decoded = layers.Dense(64, activation="relu", name="dec_dense_2")(decoded)
    decoded = layers.BatchNormalization(name="dec_bn_2")(decoded)

    # Output matches input dimensions
    outputs = layers.Dense(input_dim, activation="sigmoid", name="flow_reconstruction")(decoded)

    autoencoder = models.Model(inputs=inputs, outputs=outputs, name=f"Autoencoder_latent_{latent_dim}")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    return autoencoder


def compute_reconstruction_error(model: models.Model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Calculates sample-wise Mean Squared Error (MSE) reconstruction loss."""
    reconstructions = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - reconstructions), axis=1)


def calibrate_threshold(val_errors: np.ndarray, y_val: np.ndarray):
    """
    Calibrates decision threshold (tau) on validation data to maximize F1-score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_errors)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    # Cap index within valid threshold bounds
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    best_val_f1 = f1_scores[best_idx]
    
    return float(best_threshold), float(best_val_f1)


def run_training_pipeline():
    print("=" * 65)
    print("PHASE 4: DEEP AUTOENCODER TRAINING & TUNING (MEMBER 4)")
    print(f"Data Source:         {DATA_DIR}")
    print(f"Model Destination:   {MODEL_DIR}")
    print(f"Results Destination: {RESULTS_DIR}")
    print("=" * 65)

    # 1. Load Data Splits
    print("\n[*] Loading anomaly data splits...")
    X_train_benign = pd.read_parquet(DATA_DIR / "X_train_anomaly_benign.parquet").to_numpy(dtype=np.float32)
    X_val = pd.read_parquet(DATA_DIR / "X_val_anomaly.parquet").to_numpy(dtype=np.float32)
    y_val = pd.read_parquet(DATA_DIR / "y_val_anomaly_binary.parquet").squeeze().to_numpy(dtype=np.int8)

    X_test = pd.read_parquet(DATA_DIR / "X_test_anomaly.parquet").to_numpy(dtype=np.float32)
    y_test = pd.read_parquet(DATA_DIR / "y_test_anomaly_binary.parquet").squeeze().to_numpy(dtype=np.int8)

    input_dim = X_train_benign.shape[1]
    print(f"[+] Loaded {len(X_train_benign):,} benign training flows.")
    print(f"[+] Validation flows: {len(X_val):,} | Test flows: {len(X_test):,} ({input_dim} features)")

    # 2. Hyperparameter Search over Bottleneck Latent Dimensions
    candidate_latent_dims = [8, 16, 32]
    best_model = None
    best_latent = None
    best_val_auc = -1.0
    best_history = None
    best_threshold = None
    best_val_f1 = 0.0
    tuning_records = []

    print("\n[*] Starting Latent Bottleneck Tuning across [8, 16, 32]...")
    for l_dim in candidate_latent_dims:
        print(f"\n---> Training Candidate Architecture: Latent Dimension = {l_dim}")
        model = build_autoencoder(input_dim=input_dim, latent_dim=l_dim)

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0)
        ]

        t0 = time.time()
        # Unsupervised learning: target is X itself (pure normal traffic)
        history = model.fit(
            X_train_benign, X_train_benign,
            validation_split=0.15,
            epochs=25,
            batch_size=512,
            callbacks=cb,
            shuffle=True,
            verbose=1
        )
        train_time = time.time() - t0

        # Evaluate separation on Validation split
        val_errors = compute_reconstruction_error(model, X_val)
        val_roc_auc = roc_auc_score(y_val, val_errors)
        threshold, val_f1 = calibrate_threshold(val_errors, y_val)

        print(f"     Latent {l_dim:2d} -> Val ROC-AUC: {val_roc_auc:.4f} | Val F1: {val_f1:.4f} | Time: {train_time:.1f}s")
        tuning_records.append({
            "latent_dim": l_dim,
            "val_roc_auc": float(val_roc_auc),
            "val_f1": float(val_f1),
            "calibrated_threshold": float(threshold),
            "train_time_sec": float(train_time)
        })

        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc
            best_model = model
            best_latent = l_dim
            best_history = history
            best_threshold = threshold
            best_val_f1 = val_f1

    # Persist tuning comparison
    pd.DataFrame(tuning_records).to_csv(RESULTS_DIR / "hyperparameter_tuning.csv", index=False)
    print(f"\n[✓] Optimal Hyperparameter: Latent Dimension = {best_latent} (Val ROC-AUC: {best_val_auc:.4f})")

    # 3. Save Optimal Model
    best_model_path = MODEL_DIR / "best_autoencoder.keras"
    best_model.save(best_model_path)
    print(f"[+] Saved optimal model to: {best_model_path}")

    # Plot Convergence Curve
    plt.figure(figsize=(7, 4))
    plt.plot(best_history.history["loss"], label="Train Loss (MSE)")
    plt.plot(best_history.history["val_loss"], label="Val Loss (MSE)")
    plt.title(f"Deep Autoencoder Reconstruction Loss (Latent Dim = {best_latent})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss_curve.png", dpi=200)
    plt.close()

    # 4. Final Evaluation on Benchmark Test Split (Phase 5)
    print("\n[*] Evaluating Best Autoencoder on Test Split (X_test_anomaly)...")
    t_inf_start = time.time()
    test_errors = compute_reconstruction_error(best_model, X_test)
    inference_time = (time.time() - t_inf_start) / len(X_test)

    y_pred_test = (test_errors >= best_threshold).astype(int)

    test_roc_auc = roc_auc_score(y_test, test_errors)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_errors)
    test_pr_auc = auc(recall_curve, precision_curve)
    test_f1 = f1_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print("-" * 50)
    print("AUTOENCODER TEST METRICS (OFFICIAL BENCHMARK)")
    print("-" * 50)
    print(f"ROC-AUC:                  {test_roc_auc:.4f}")
    print(f"PR-AUC:                   {test_pr_auc:.4f}")
    print(f"Anomaly F1-Score:         {test_f1:.4f}")
    print(f"Precision:                {test_prec:.4f}")
    print(f"Recall (Detection Rate):  {test_rec:.4f}")
    print(f"False Positive Rate (FPR):{test_fpr:.4%}")
    print(f"Inference Latency/Flow:   {inference_time * 1e6:.2f} microseconds")

    # 5. Held-Out / Zero-Day Experiment (Objective 4 & Phase 6)
    heldout_file = DATA_DIR / "X_heldout_portscan.parquet"
    if heldout_file.exists():
        print("\n[*] Evaluating on Held-Out Unseen Threat (X_heldout_portscan.parquet)...")
        X_heldout = pd.read_parquet(heldout_file).to_numpy(dtype=np.float32)
        heldout_errors = compute_reconstruction_error(best_model, X_heldout)
        heldout_detected = np.sum(heldout_errors >= best_threshold)
        heldout_recall = heldout_detected / len(heldout_errors)
        print(f"[+] Held-Out PortScan Detection Rate: {heldout_recall:.2%} ({heldout_detected:,}/{len(heldout_errors):,} flagged)")
    else:
        heldout_recall = None

    # 6. Save Model Metadata & Artifacts
    metadata = {
        "model_name": "Deep Autoencoder",
        "input_dim": int(input_dim),
        "best_latent_dim": int(best_latent),
        "calibrated_threshold": float(best_threshold),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "test_f1": float(test_f1),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_fpr": float(test_fpr),
        "inference_latency_seconds_per_sample": float(inference_time),
        "heldout_portscan_recall": float(heldout_recall) if heldout_recall is not None else "N/A"
    }

    with open(MODEL_DIR / "autoencoder_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    pd.DataFrame([metadata]).to_csv(RESULTS_DIR / "autoencoder_test_metrics.csv", index=False)
    print(f"\n[✓] Completed successfully! Saved model to {MODEL_DIR} and results to {RESULTS_DIR}")


if __name__ == "__main__":
    run_training_pipeline()