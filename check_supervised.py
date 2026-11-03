#!/usr/bin/env python
"""
Pre-training checks: SUPERVISED track (Random Forest & XGBoost).

Run from the project root:
    python scripts/check_supervised.py                         # data/processed
    python scripts/check_supervised.py --dir data/processed/log1p
    python scripts/check_supervised.py --smoke                 # + quick RF/XGB fit on a sample

Exit code 1 if any FAIL. WARNs are decisions to make/document, not blockers.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LEAKY = {"flow id", "source ip", "src ip", "destination ip", "dst ip",
         "timestamp", "source port", "src port", "destination port", "dst port"}
SPLITS = ("train", "val", "test")
_tally = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(name, ok, detail="", warn_only=False):
    status = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    _tally[status] += 1
    print(f"[{status}] {name}" + (f"  ->  {detail}" if (detail and not ok) else ""))
    return ok


def section(title):
    print(f"\n=== {title} ===")


def row_hash(X, y=None):
    frame = X if y is None else X.assign(_y=y.to_numpy())
    return pd.util.hash_pandas_object(frame, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed")
    ap.add_argument("--heldout", default="portscan")
    ap.add_argument("--smoke", action="store_true", help="train small RF/XGB on a sample")
    args = ap.parse_args()
    D = Path(args.dir)

    # ------------------------------------------------------------------ 1
    section("1. Files")
    needed = [f"{a}_{s}_supervised.parquet" for s in SPLITS for a in ("X", "y")]
    needed += ["label_mapping.json", "feature_names.json", "scaler.joblib"]
    missing = [f for f in needed if not (D / f).exists()]
    if not report("All supervised artifacts present", not missing, f"missing: {missing}"):
        sys.exit(1)

    X = {s: pd.read_parquet(D / f"X_{s}_supervised.parquet") for s in SPLITS}
    y = {s: pd.read_parquet(D / f"y_{s}_supervised.parquet")["Label"] for s in SPLITS}
    mapping = json.loads((D / "label_mapping.json").read_text(encoding="utf-8"))
    feats = json.loads((D / "feature_names.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in mapping.items()}
    for s in SPLITS:
        print(f"    {s:5s}: {X[s].shape[0]:>10,} rows x {X[s].shape[1]} features")

    # ------------------------------------------------------------------ 2
    section("2. Structure & alignment")
    for s in SPLITS:
        report(f"{s}: X and y aligned (same length & index)",
               len(X[s]) == len(y[s]) and X[s].index.equals(y[s].index))
    report("Columns identical across splits and match feature_names.json",
           all(list(X[s].columns) == feats for s in SPLITS))
    leaky = [c for c in feats if c.strip().casefold() in LEAKY]
    report("No identifier / port / timestamp columns", not leaky, f"found: {leaky}")
    non32 = [c for c in feats if X["train"][c].dtype != np.float32]
    report("All features float32", not non32, f"{len(non32)} non-float32 columns", warn_only=True)

    # ------------------------------------------------------------------ 3
    section("3. Values")
    for s in SPLITS:
        bad = int((~np.isfinite(X[s].to_numpy())).sum())
        report(f"{s}: no NaN / inf", bad == 0, f"{bad:,} non-finite cells")
    mins, maxs = X["train"].min(), X["train"].max()
    const = maxs.index[(maxs - mins) <= 0].tolist()
    report("No constant features in train", not const, f"{const}", warn_only=True)
    report("Train scaled to exactly [0, 1] per feature",
           np.allclose(mins, 0, atol=1e-6) and np.allclose(maxs[~maxs.index.isin(const)], 1, atol=1e-6),
           f"min in [{mins.min():.3g}, {mins.max():.3g}], max in [{maxs.min():.3g}, {maxs.max():.3g}]")
    for s in ("val", "test"):
        a = X[s].to_numpy()
        frac = float(((a < 0) | (a > 1)).mean())
        report(f"{s}: values within [0, 1]", frac == 0,
               f"{frac:.4%} cells outside (unclipped; harmless for trees)", warn_only=True)

    # ------------------------------------------------------------------ 4
    section("4. Leakage between splits")
    idx = {s: X[s].index for s in SPLITS}
    ho_path = D / f"X_heldout_{args.heldout}.parquet"
    if ho_path.exists():
        idx["heldout"] = pd.read_parquet(ho_path, columns=[feats[0]]).index
    else:
        report("Held-out file present", False, f"{ho_path.name} not found -> re-run the pipeline",
               warn_only=True)
    names = list(idx)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            n = int(idx[a].isin(idx[b]).sum())
            report(f"No index overlap {a} / {b}", n == 0, f"{n:,} shared indices")

    full_tr = row_hash(X["train"], y["train"]).to_numpy()
    feat_tr = row_hash(X["train"]).to_numpy()
    for s in ("val", "test"):
        fh, full = row_hash(X[s]), row_hash(X[s], y[s])
        exact = int(full.isin(full_tr).sum())
        conflict = int((fh.isin(feat_tr) & ~full.isin(full_tr)).sum())
        report(f"{s}: no exact (features + label) copies of train rows", exact == 0, f"{exact:,} rows")
        report(f"{s}: no rows with train-identical features but a different label", conflict == 0,
               f"{conflict:,} rows (label noise; caps achievable accuracy, document it)", warn_only=True)
    dup = int(X["train"].duplicated().sum())
    print(f"    info: {dup:,} duplicate feature rows inside train ({dup / len(X['train']):.2%})")

    # ------------------------------------------------------------------ 5
    section("5. Labels & class balance")
    report("BENIGN encoded as 0", mapping.get("BENIGN") == 0, f"BENIGN -> {mapping.get('BENIGN')}")
    report("Class ids contiguous 0..K-1 (XGBoost requires this)",
           sorted(mapping.values()) == list(range(len(mapping))))
    leaked = [k for k in mapping if k.replace(" ", "").lower() == args.heldout.lower()]
    report("Held-out attack absent from label mapping", not leaked, f"found {leaked}")
    for s in SPLITS:
        miss = sorted(set(mapping.values()) - set(y[s].unique()))
        report(f"{s}: every class present", not miss, f"missing: {[inv[c] for c in miss]}")

    counts = pd.DataFrame({s: y[s].value_counts() for s in SPLITS}).fillna(0).astype(int)
    counts.index = [inv[i] for i in counts.index]
    counts = counts.sort_values("train", ascending=False)
    counts["train_%"] = (100 * counts["train"] / counts["train"].sum()).round(3)
    print(counts.to_string())

    small = counts.index[(counts["val"] < 10) | (counts["test"] < 10)].tolist()
    report("Every class has >= 10 rows in val and test", not small,
           f"too few to evaluate reliably: {small} -> drop / merge / report separately", warn_only=True)
    prop = counts[list(SPLITS)] / counts[list(SPLITS)].sum()
    big = counts["train"] >= 100
    ratio = prop.loc[big, "test"] / prop.loc[big, "train"]
    ok = ratio.between(0.8, 1.25)
    report("Stratification preserved (test/train class share within 0.8-1.25)", ok.all(),
           f"off: {ratio[~ok].round(2).to_dict()}", warn_only=True)
    imb = counts["train"].max() / max(counts["train"].min(), 1)
    print(f"    info: imbalance ratio largest/smallest train class = {imb:,.0f}:1 -> use class weights")

    # ------------------------------------------------------------------ 6
    if args.smoke:
        section("6. Smoke test (stratified sample, max 20k rows per class)")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, f1_score
        from sklearn.utils.class_weight import compute_sample_weight

        sidx = (y["train"].groupby(y["train"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), 20_000), random_state=42)).index)
        Xs, ys = X["train"].loc[sidx], y["train"].loc[sidx]
        target_names = [inv[i] for i in range(len(mapping))]

        t = time.time()
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                    class_weight="balanced_subsample", random_state=42).fit(Xs, ys)
        pred = rf.predict(X["val"])
        f1 = f1_score(y["val"], pred, average="macro")
        print(f"    RF : macro-F1(val) = {f1:.4f}  ({time.time() - t:.0f}s on {len(Xs):,} rows)")
        print(classification_report(y["val"], pred, labels=list(range(len(mapping))),
                                    target_names=target_names, digits=3, zero_division=0))
        report("RF trains and scores plausibly (0.5 < macro-F1 < 1.0)", 0.5 < f1 < 0.9999,
               "F1 == 1.0 -> suspect leakage; < 0.5 -> suspect broken labels", warn_only=True)
        top = pd.Series(rf.feature_importances_, index=feats).nlargest(10)
        print("    top-10 RF importances:\n" + top.round(4).to_string())
        report("No single feature dominates (top importance < 0.5)", top.iloc[0] < 0.5,
               f"{top.index[0]} = {top.iloc[0]:.2f} -> check it is not a shortcut", warn_only=True)

        try:
            from xgboost import XGBClassifier
            t = time.time()
            xgb = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                tree_method="hist", n_jobs=-1, random_state=42)
            xgb.fit(Xs, ys, sample_weight=compute_sample_weight("balanced", ys))
            f1x = f1_score(y["val"], xgb.predict(X["val"]), average="macro")
            print(f"    XGB: macro-F1(val) = {f1x:.4f}  ({time.time() - t:.0f}s)")
            report("XGBoost trains on the encoded labels", True)
        except ImportError:
            report("xgboost installed", False)

    section("Summary")
    print(f"PASS {_tally['PASS']}   WARN {_tally['WARN']}   FAIL {_tally['FAIL']}")
    sys.exit(1 if _tally["FAIL"] else 0)


if __name__ == "__main__":
    main()
