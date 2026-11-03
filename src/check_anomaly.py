#!/usr/bin/env python
"""
Pre-training checks: ANOMALY track (Isolation Forest & Autoencoder).

Run from the project root:
    python scripts/check_anomaly.py                            # data/processed
    python scripts/check_anomaly.py --dir data/processed/log1p
    python scripts/check_anomaly.py --smoke                    # + quick Isolation Forest baseline

Exit code 1 if any FAIL. WARNs are decisions to make/document, not blockers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_tally = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(name, ok, detail="", warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    _tally[status] += 1
    print(f"[{status}] {name}" + (f"  ->  {detail}" if (detail and not ok) else ""))
    return ok


def section(title):
    print(f"\n=== {title} ===")


def row_hash(X):
    return pd.util.hash_pandas_object(X, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed")
    ap.add_argument("--heldout", default="portscan")
    ap.add_argument("--smoke", action="store_true", help="fit a quick Isolation Forest baseline")
    args = ap.parse_args()
    D, ho = Path(args.dir), args.heldout.lower()

    # ------------------------------------------------------------------ 1
    section("1. Files")
    needed = ["X_train_anomaly_benign.parquet",
              "X_val_anomaly.parquet", "y_val_anomaly_binary.parquet",
              "X_test_anomaly.parquet", "y_test_anomaly_binary.parquet",
              f"X_heldout_{ho}.parquet", f"y_heldout_{ho}.parquet",
              "y_train_supervised.parquet", "y_val_supervised.parquet", "y_test_supervised.parquet",
              "label_mapping.json", "feature_names.json"]
    missing = [f for f in needed if not (D / f).exists()]
    if not report("All anomaly artifacts present", not missing, f"missing: {missing}"):
        sys.exit(1)

    Xb = pd.read_parquet(D / "X_train_anomaly_benign.parquet")
    Xv = pd.read_parquet(D / "X_val_anomaly.parquet")
    Xt = pd.read_parquet(D / "X_test_anomaly.parquet")
    Xh = pd.read_parquet(D / f"X_heldout_{ho}.parquet")
    yv = pd.read_parquet(D / "y_val_anomaly_binary.parquet")["Anomaly"]
    yt = pd.read_parquet(D / "y_test_anomaly_binary.parquet")["Anomaly"]
    yh = pd.read_parquet(D / f"y_heldout_{ho}.parquet")["Anomaly"]
    ys = {s: pd.read_parquet(D / f"y_{s}_supervised.parquet")["Label"] for s in ("train", "val", "test")}
    mapping = json.loads((D / "label_mapping.json").read_text(encoding="utf-8"))
    feats = json.loads((D / "feature_names.json").read_text(encoding="utf-8"))
    benign_id = mapping["BENIGN"]
    for name, A in [("benign train", Xb), ("val", Xv), ("test", Xt), ("held-out", Xh)]:
        print(f"    {name:12s}: {len(A):>10,} rows x {A.shape[1]} features")

    # ------------------------------------------------------------------ 2
    section("2. Structure & alignment")
    report("Columns match feature_names.json in all sets",
           all(list(A.columns) == feats for A in (Xb, Xv, Xt, Xh)))
    for name, A, b in [("val", Xv, yv), ("test", Xt, yt), ("held-out", Xh, yh)]:
        report(f"{name}: X and y aligned", len(A) == len(b) and A.index.equals(b.index))
    same_rows = Xv.index.equals(ys["val"].index) and Xt.index.equals(ys["test"].index)
    report("Val/test anomaly rows == supervised val/test rows (needed for the hybrid)", same_rows)
    if same_rows:
        for name, b, s in [("val", yv, "val"), ("test", yt, "test")]:
            expected = (ys[s] != benign_id).astype("int8")
            report(f"{name}: binary label == (multiclass label != BENIGN)",
                   bool((b.to_numpy() == expected.to_numpy()).all()))
    for name, A in [("benign train", Xb), ("val", Xv), ("test", Xt), ("held-out", Xh)]:
        bad = int((~np.isfinite(A.to_numpy())).sum())
        report(f"{name}: no NaN / inf", bad == 0, f"{bad:,} non-finite cells")

    # ------------------------------------------------------------------ 3
    section("3. Benign-only training set")
    tr_benign = ys["train"].index[ys["train"] == benign_id]
    report("Benign train holds only BENIGN rows from the train split", bool(Xb.index.isin(tr_benign).all()))
    report("Benign train holds all BENIGN train rows", len(Xb) == len(tr_benign),
           f"{len(Xb):,} vs {len(tr_benign):,}", warn_only=True)
    dup = float(Xb.duplicated().mean())
    print(f"    info: {dup:.2%} of benign train rows are duplicate feature vectors")

    # ------------------------------------------------------------------ 4
    section("4. Held-out (unseen) attack set")
    report("Held-out labels non-empty and all 1", len(yh) > 0 and bool((yh == 1).all()))
    used = np.concatenate([ys[s].index.to_numpy() for s in ys])
    n = int(Xh.index.isin(used).sum())
    report("Held-out rows never appear in train/val/test", n == 0, f"{n:,} overlapping indices")
    same = int(row_hash(Xh).isin(row_hash(Xb).to_numpy()).sum())
    report("Held-out flows are not feature-identical to benign train flows", same == 0,
           f"{same:,} held-out flows ({same / len(Xh):.1%}) look exactly like benign -> "
           "undetectable by any model; report as a ceiling", warn_only=True)

    # ------------------------------------------------------------------ 5
    section("5. Scaling / clipping effects")
    caps, any_outside = {}, False
    for name, A in [("benign train", Xb), ("val", Xv), ("test", Xt), ("held-out", Xh)]:
        a = A.to_numpy()
        outside = float(((a < 0) | (a > 1)).mean())
        any_outside |= outside > 0
        caps[name] = float((a >= 1.0).any(axis=1).mean())
        print(f"    {name:12s}: rows with any feature >= 1.0: {caps[name]:7.3%} | "
              f"cells outside [0,1]: {outside:.4%}")
    clipped = not any_outside
    print(f"    info: data looks {'CLIPPED' if clipped else 'unclipped'}")
    report("Clipping is not hiding held-out extremes",
           not (clipped and caps["held-out"] > 5 * max(caps["benign train"], 1e-4)),
           "held-out flows hit the 1.0 cap far more often than benign -> "
           "regenerate anomaly splits with clip=False", warn_only=True)

    sample = Xb.sample(min(200_000, len(Xb)), random_state=42)
    p99 = sample.quantile(0.99)
    squashed = int((p99 < 0.01).sum())
    report("Benign features not squashed near 0", squashed < 0.25 * len(feats),
           f"{squashed}/{len(feats)} features have 99% of benign values < 0.01 -> "
           "prefer the log1p variant for the Autoencoder", warn_only=True)

    # ------------------------------------------------------------------ 6
    section("6. Evaluation-set composition")
    print(f"    val  anomaly rate: {yv.mean():.2%}  ({int(yv.sum()):,} attacks / {int((yv == 0).sum()):,} benign)")
    print(f"    test anomaly rate: {yt.mean():.2%}  ({int(yt.sum()):,} attacks / {int((yt == 0).sum()):,} benign)")
    print(f"    held-out: {len(Xh):,} attacks only -> pair with test BENIGN rows for precision/FPR")

    # ------------------------------------------------------------------ 7
    section("7. log1p variant consistency")
    base = D.parent if D.name == "log1p" else D
    L = base / "log1p"
    if not L.exists():
        report("log1p variant generated", False, f"{L} not found", warn_only=True)
    else:
        def idx(p, f):
            return pd.read_parquet(p / f, columns=[feats[0]]).index if (p / f).exists() else None
        for f in ("X_train_anomaly_benign.parquet", "X_val_anomaly.parquet",
                  "X_test_anomaly.parquet", f"X_heldout_{ho}.parquet"):
            a, b = idx(base, f), idx(L, f)
            report(f"Same rows in base and log1p: {f}", a is not None and b is not None and a.equals(b))

    # ------------------------------------------------------------------ 8
    section("8. Environment & memory")
    gb = Xb.memory_usage(index=False).sum() / 1e9
    print(f"    benign train in memory: {gb:.2f} GB (float32 tensor for the AE is about the same)")
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        print(f"    torch {torch.__version__}, device: {dev}")
        report("GPU available for the Autoencoder", dev != "cpu",
               "CPU only: train the AE on a benign subsample or use fewer epochs", warn_only=True)
    except ImportError:
        report("torch installed", False)

    # ------------------------------------------------------------------ 9
    if args.smoke:
        section("9. Smoke test: Isolation Forest baseline")
        from sklearn.ensemble import IsolationForest
        from sklearn.metrics import roc_auc_score

        t = time.time()
        fit = Xb.sample(min(200_000, len(Xb)), random_state=42)
        iso = IsolationForest(n_estimators=200, max_samples=256, n_jobs=-1, random_state=42).fit(fit)
        sv, st, sh = (-iso.score_samples(A) for A in (Xv, Xt, Xh))   # higher = more anomalous
        yv_np, yt_np = yv.to_numpy(), yt.to_numpy()
        auc = roc_auc_score(yv_np, sv)
        thr = np.quantile(sv[yv_np == 0], 0.99)                       # benign-only threshold, ~1% FPR
        print(f"    val ROC-AUC              : {auc:.3f}")
        print(f"    test FPR (benign)        : {(st[yt_np == 0] > thr).mean():.2%}")
        print(f"    test detection (known)   : {(st[yt_np == 1] > thr).mean():.2%}")
        print(f"    held-out detection ({ho}) : {(sh > thr).mean():.2%}")
        print(f"    ({time.time() - t:.0f}s)")
        report("Isolation Forest separates benign from attacks (val AUC > 0.6)", auc > 0.6,
               f"AUC = {auc:.3f}", warn_only=True)

    section("Summary")
    print(f"PASS {_tally['PASS']}   WARN {_tally['WARN']}   FAIL {_tally['FAIL']}")
    sys.exit(1 if _tally["FAIL"] else 0)


if __name__ == "__main__":
    main()
