"""
metrics.py - Standalone evaluation for the Deep Autoencoder (Base vs Log1p)
Location: project root (metrics.py)
Owner: Member 4

Run:  python metrics.py

Does NOT retrain anything. Loads saved models + thresholds, scores the TEST split once,
and writes everything needed for (a) the AE vs Isolation Forest comparison and
(b) the hybrid NIDS.

Convention shared with Isolation Forest: score is HIGHER = MORE ANOMALOUS.
(sklearn's IsolationForest.score_samples is the opposite, so use -score_samples.)
The generic helpers `evaluate_scores` and `threshold_sensitivity` take only scores,
so Member 3 can import them for an identical metric definition:
    from metrics import evaluate_scores
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "autoencoder"
RESULTS_DIR = ROOT / "results" / "autoencoder"
PLOTS_DIR = RESULTS_DIR / "plots"
SCORES_DIR = RESULTS_DIR / "scores"

for d in (RESULTS_DIR, PLOTS_DIR, SCORES_DIR):
    d.mkdir(parents=True, exist_ok=True)

BENIGN_NAMES = {"benign", "normal"}


# --------------------------------------------------------------------------- #
# Generic, model-agnostic metric helpers (reusable for Isolation Forest)
# --------------------------------------------------------------------------- #
def evaluate_scores(y_true, scores, threshold, heldout_scores=None):
    """All headline metrics from anomaly scores (higher = more anomalous)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "test_roc_auc": roc_auc_score(y_true, scores),
        "test_ap": average_precision_score(y_true, scores),  # proper AP, not trapezoid PR-AUC
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }

    if heldout_scores is not None and len(heldout_scores) > 0:
        benign_scores = scores[y_true == 0]
        detected = int(np.sum(heldout_scores >= threshold))
        out["heldout_recall"] = detected / len(heldout_scores)
        out["heldout_flagged"] = f"{detected}/{len(heldout_scores)}"
        # Threshold-independent: PortScan vs benign TEST flows
        labels = np.r_[np.zeros(len(benign_scores)), np.ones(len(heldout_scores))]
        out["heldout_roc_auc"] = roc_auc_score(labels, np.r_[benign_scores, heldout_scores])
    else:
        out["heldout_recall"] = None
        out["heldout_flagged"] = "0/0"
        out["heldout_roc_auc"] = None
    return out, y_pred, cm


def threshold_sensitivity(y_true, scores, thresholds: dict, heldout_scores=None):
    """Re-evaluates the same scores under several thresholds."""
    rows = []
    for name, thr in thresholds.items():
        if thr is None:
            continue
        m, _, _ = evaluate_scores(y_true, scores, thr, heldout_scores)
        rows.append({
            "threshold_rule": name, "threshold": float(thr),
            "recall": m["test_recall"], "precision": m["test_precision"],
            "f1": m["test_f1"], "fpr": m["test_fpr"],
            "heldout_recall": m["heldout_recall"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# AE-specific helpers
# --------------------------------------------------------------------------- #
def reconstruction_error(model, X, batch_size=1024):
    recon = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def timed_scoring(model, X, batch_size=1024):
    """Warm-up first (graph tracing), then time full scoring. Reports amortized time/sample."""
    model.predict(X[: min(len(X), batch_size)], batch_size=batch_size, verbose=0)  # warm-up
    t0 = time.perf_counter()
    errors = reconstruction_error(model, X, batch_size)
    elapsed = time.perf_counter() - t0
    return errors, elapsed, elapsed / len(X) * 1e6


def per_attack_recall(y_test, y_pred, data_path: Path):
    """
    Per-attack-type recall using the supervised test labels, ONLY if those rows are
    provably the same rows as the anomaly test set. Returns (DataFrame|None, note).
    """
    ys_file = data_path / "y_test_supervised.parquet"
    lm_file = data_path / "label_mapping.json"
    if not ys_file.exists() or not lm_file.exists():
        return None, "supervised label files not found"
    try:
        y_sup = pd.read_parquet(ys_file).squeeze().to_numpy()
        if len(y_sup) != len(y_test):
            return None, (f"row counts differ (anomaly test={len(y_test):,}, "
                          f"supervised test={len(y_sup):,}): NOT the same rows. "
                          "Use a shared row id to build a common test set for the hybrid.")
        with open(lm_file) as f:
            mapping = json.load(f)
        if all(isinstance(v, (int, float)) for v in mapping.values()):
            id2name = {int(v): k for k, v in mapping.items()}
        else:
            id2name = {int(k): v for k, v in mapping.items()}
        names = pd.Series(y_sup).map(id2name) if y_sup.dtype != object else pd.Series(y_sup)
        is_benign = names.astype(str).str.lower().isin(BENIGN_NAMES).to_numpy()
        if not np.array_equal(~is_benign, np.asarray(y_test) == 1):
            return None, "binary labels disagree with supervised labels: rows are NOT aligned."
        df = pd.DataFrame({"class": names, "pred": y_pred})
        res = (df.groupby("class")["pred"]
                 .agg(n_flows="size", flagged="sum")
                 .assign(recall_or_flag_rate=lambda d: d["flagged"] / d["n_flows"])
                 .reset_index())
        return res, "aligned with supervised test set (same rows, same order)"
    except Exception as e:  # label format surprises
        return None, f"could not parse supervised labels: {e}"


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(variant, y_test, scores, heldout_scores, threshold, cm, ap):
    tag = variant.upper()

    fpr, tpr, _ = roc_curve(y_test, scores)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"AE ({tag})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Autoencoder ({tag})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"roc_curve_{variant}.png", dpi=200); plt.close()

    p, r, _ = precision_recall_curve(y_test, scores)
    plt.figure(figsize=(5, 4.5))
    plt.plot(r, p, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve - Autoencoder ({tag})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"pr_curve_{variant}.png", dpi=200); plt.close()

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Benign", "Attack"]); plt.yticks([0, 1], ["Benign", "Attack"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"confusion_matrix_{variant}.png", dpi=200); plt.close()

    # The key explanatory figure: error distributions
    groups = {"Benign (test)": scores[y_test == 0], "Attack (test)": scores[y_test == 1]}
    if heldout_scores is not None and len(heldout_scores):
        groups["PortScan (held-out)"] = heldout_scores
    allv = np.clip(np.concatenate(list(groups.values())), 1e-9, None)
    bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 80)
    plt.figure(figsize=(8, 4.5))
    for name, v in groups.items():
        plt.hist(np.clip(v, 1e-9, None), bins=bins, alpha=0.5, density=True, label=name)
    plt.axvline(threshold, color="k", linestyle="--", label=f"Threshold = {threshold:.2e}")
    plt.xscale("log"); plt.xlabel("Reconstruction error (MSE, log scale)"); plt.ylabel("Density")
    plt.title(f"Reconstruction Error Distributions ({tag})"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(PLOTS_DIR / f"error_distributions_{variant}.png", dpi=200); plt.close()


# --------------------------------------------------------------------------- #
# Per-variant evaluation
# --------------------------------------------------------------------------- #
def load_ae(path_stem: str):
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_DIR / f"best_autoencoder_{path_stem}.keras")
    with open(MODEL_DIR / f"autoencoder_metadata_{path_stem}.json") as f:
        meta = json.load(f)
    return model, meta


def evaluate_variant(variant: str, data_path: Path):
    main_path = MODEL_DIR / f"best_autoencoder_{variant}.keras"
    if not main_path.exists() or not (MODEL_DIR / f"autoencoder_metadata_{variant}.json").exists():
        print(f"[!] Artifacts for '{variant}' not found. Train first.")
        return None

    model, meta = load_ae(variant)
    threshold = float(meta["calibrated_threshold"])

    X_test = pd.read_parquet(data_path / "X_test_anomaly.parquet").to_numpy(dtype=np.float32)
    y_test = pd.read_parquet(data_path / "y_test_anomaly_binary.parquet").squeeze().to_numpy(dtype=np.int8)

    held_file = data_path / "X_heldout_portscan.parquet"
    X_held = pd.read_parquet(held_file).to_numpy(dtype=np.float32) if held_file.exists() else None

    test_scores, pred_time, latency_us = timed_scoring(model, X_test)
    held_scores = reconstruction_error(model, X_held) if X_held is not None else None

    m, y_pred, cm = evaluate_scores(y_test, test_scores, threshold, held_scores)

    # ---- Save raw scores for the hybrid NIDS / comparison ------------------
    pd.DataFrame({
        "row_idx": np.arange(len(y_test)),
        "y_true": y_test,
        "ae_score": test_scores,
        "ae_pred": y_pred,
    }).to_parquet(SCORES_DIR / f"ae_test_scores_{variant}.parquet", index=False)
    if held_scores is not None:
        np.save(SCORES_DIR / f"ae_heldout_scores_{variant}.npy", held_scores)

    # ---- Plots, threshold sensitivity, per-attack recall -------------------
    make_plots(variant, y_test, test_scores, held_scores, threshold, cm, m["test_ap"])

    sens = threshold_sensitivity(
        y_test, test_scores,
        {"F1-optimal (validation)": threshold,
         "p95 of benign train error": meta.get("benign_train_error_p95"),
         "p99 of benign train error": meta.get("benign_train_error_p99")},
        held_scores,
    )
    sens.insert(0, "variant", variant)

    per_attack, note = per_attack_recall(y_test, y_pred, data_path)
    print(f"[i] [{variant}] per-attack recall: {note}")
    if per_attack is not None:
        per_attack.to_csv(RESULTS_DIR / f"ae_per_attack_recall_{variant}.csv", index=False)

    # ---- Seed variance (main model + any *_seedN models) -------------------
    seed_rows = [{"seed": meta.get("seed", 42), **{k: m[k] for k in
                  ("test_f1", "test_recall", "test_fpr", "test_roc_auc", "test_ap",
                   "heldout_recall", "heldout_roc_auc")}}]
    for p in sorted(MODEL_DIR.glob(f"best_autoencoder_{variant}_seed*.keras")):
        stem = p.stem.replace("best_autoencoder_", "")
        sm, smeta = load_ae(stem)
        s_scores = reconstruction_error(sm, X_test)
        s_held = reconstruction_error(sm, X_held) if X_held is not None else None
        s, _, _ = evaluate_scores(y_test, s_scores, float(smeta["calibrated_threshold"]), s_held)
        seed_rows.append({"seed": smeta.get("seed"), **{k: s[k] for k in
                          ("test_f1", "test_recall", "test_fpr", "test_roc_auc", "test_ap",
                           "heldout_recall", "heldout_roc_auc")}})
    seed_df = pd.DataFrame(seed_rows)
    seed_df.insert(0, "variant", variant)

    # ---- Detailed + academic rows -----------------------------------------
    r4 = lambda x: round(float(x), 4) if x is not None else "N/A"
    pct = lambda x: round(float(x) * 100, 2) if x is not None else "N/A"

    detailed = {
        "variant": variant,
        "input_dim": meta.get("input_dim"),
        "best_latent_dim": meta.get("best_latent_dim"),
        "calibrated_threshold": threshold,
        "test_roc_auc": r4(m["test_roc_auc"]), "test_ap": r4(m["test_ap"]),
        "test_f1": r4(m["test_f1"]), "test_accuracy": r4(m["test_accuracy"]),
        "test_precision": r4(m["test_precision"]), "test_recall": r4(m["test_recall"]),
        "test_fpr": r4(m["test_fpr"]),
        "false_positives": m["false_positives"], "false_negatives": m["false_negatives"],
        "heldout_portscan_recall": r4(m["heldout_recall"]),
        "heldout_portscan_roc_auc": r4(m["heldout_roc_auc"]),
        "heldout_flagged_counts": m["heldout_flagged"],
        "train_time_sec": round(meta.get("train_time_sec", float("nan")), 1),
        "epochs_run": meta.get("epochs_run"),
        "param_count": meta.get("param_count", int(model.count_params())),
        "amortized_us_per_sample": round(latency_us, 2),
        "pred_time_sec": round(pred_time, 2),
        "n_test": int(len(y_test)),
    }

    academic = {
        "Variant": "AE (Base)" if variant == "base" else "AE (Log1p)",
        "Accuracy (%)": pct(m["test_accuracy"]),
        "Recall (%)": pct(m["test_recall"]),
        "FAR (%)": pct(m["test_fpr"]),
        "F1-Score (%)": pct(m["test_f1"]),
        "AP (%)": pct(m["test_ap"]),
        "ROC-AUC (%)": pct(m["test_roc_auc"]),
        "PortScan Catch (%)": pct(m["heldout_recall"]),
        "PortScan AUC (%)": pct(m["heldout_roc_auc"]),
        "Train time (s)": round(meta.get("train_time_sec", float("nan")), 1),
        "Params": detailed["param_count"],
        "Pred. time (s)": round(pred_time, 2),
        "us/sample (amortized)": round(latency_us, 2),
    }
    return detailed, academic, sens, seed_df


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_evaluation():
    print("=" * 80)
    print("AUTOENCODER METRIC EVALUATION (BASE VS. LOG1P)")
    print("=" * 80)

    variants = [("base", DATA_PROCESSED_DIR), ("log1p", DATA_PROCESSED_DIR / "log1p")]
    detailed, academic, sens_all, seed_all = [], [], [], []

    for name, path in variants:
        if not path.exists():
            print(f"[!] {path} not found. Skipping {name}.")
            continue
        res = evaluate_variant(name, path)
        if res:
            d, a, s, sd = res
            detailed.append(d); academic.append(a); sens_all.append(s); seed_all.append(sd)

    if not detailed:
        print("[!] Nothing evaluated.")
        return

    df_detail, df_table = pd.DataFrame(detailed), pd.DataFrame(academic)
    df_sens, df_seeds = pd.concat(sens_all, ignore_index=True), pd.concat(seed_all, ignore_index=True)

    summary = (df_seeds.groupby("variant")
               .agg(n_seeds=("seed", "count"),
                    **{f"{c}_mean": (c, "mean") for c in
                       ("test_f1", "test_recall", "test_fpr", "test_roc_auc", "test_ap",
                        "heldout_recall", "heldout_roc_auc")},
                    **{f"{c}_std": (c, "std") for c in
                       ("test_f1", "test_recall", "test_fpr", "test_roc_auc", "test_ap",
                        "heldout_recall", "heldout_roc_auc")})
               .round(4).reset_index())

    print("\n" + "=" * 80)
    print("ANOMALY DETECTION COMPARATIVE BENCHMARK TABLE (AUTOENCODER)")
    print("=" * 80)
    print(df_table.to_string(index=False))
    print("\nTHRESHOLD SENSITIVITY\n" + df_sens.round(4).to_string(index=False))
    print("\nSEED VARIANCE (std is NaN when only one seed was trained)\n" + summary.to_string(index=False))

    df_table.to_csv(RESULTS_DIR / "ae_academic_table_metrics.csv", index=False)
    df_detail.to_csv(RESULTS_DIR / "autoencoder_test_metrics.csv", index=False)
    df_sens.to_csv(RESULTS_DIR / "ae_threshold_sensitivity.csv", index=False)
    df_seeds.to_csv(RESULTS_DIR / "ae_seed_runs.csv", index=False)
    summary.to_csv(RESULTS_DIR / "ae_seed_variance_summary.csv", index=False)
    with open(RESULTS_DIR / "ae_all_combined_metrics.json", "w") as f:
        json.dump(detailed, f, indent=4, default=str)

    print(f"\n[✓] Results saved under: {RESULTS_DIR}")
    print(f"    Scores for hybrid/comparison: {SCORES_DIR}")
    print(f"    Plots: {PLOTS_DIR}")


if __name__ == "__main__":
    run_evaluation()