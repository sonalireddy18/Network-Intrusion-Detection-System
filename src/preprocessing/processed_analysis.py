"""
NIDS report analysis (tailored to data/processed/ and data/processed/log1p/)
----------------------------------------------------------------------------
Produces tables, graphs and a summary.md for the preprocessing section of the report:
  1. Split sizes: supervised (train/val/test), anomaly (benign-only train, val, test), PortScan holdout
  2. Class distribution per split (uses label_mapping.json) + stratification check
  3. Pipeline artifacts: feature count, log1p columns, twin_drop_index, scaler
  4. Data quality of both versions (no-log1p vs log1p)
  5. log1p effect: skewness, kurtosis, tail extremeness, histograms before/after
  6. Leakage checks: duplicate rows across splits, scaler fitted on train only, holdout absent from train
  7. Parquet vs CSV benchmark

Run from your project root:   python nids_report_analysis.py
Needs: pandas numpy matplotlib pyarrow joblib   (scikit-learn only if the scaler needs it to unpickle)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

# ============================ CONFIG: EDIT IF NEEDED ========================
ROOT = Path("data/processed")        # version WITHOUT log1p (scaled)
LOG1P = ROOT / "log1p"               # version WITH log1p (then scaled)
OUT = Path("report_analysis")
SAMPLE_N = 300_000                   # rows sampled for skew/plots (speed + memory)
TOP_K = 6                            # features in the before/after histogram figure
SEED = 42
# ============================================================================

SUP = ["train_supervised", "val_supervised", "test_supervised"]
ANO = ["train_anomaly_benign", "val_anomaly", "test_anomaly"]
HOLD = ["heldout_portscan"]
ALL_X = SUP + ANO + HOLD
Y_FILES = {
    "train_supervised": "y_train_supervised", "val_supervised": "y_val_supervised",
    "test_supervised": "y_test_supervised", "val_anomaly": "y_val_anomaly_binary",
    "test_anomaly": "y_test_anomaly_binary", "heldout_portscan": "y_heldout_portscan",
}

OUT.mkdir(exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
summary = []


def log(title, df=None, text=None):
    print(f"\n=== {title} ===")
    summary.append(f"\n## {title}\n")
    if df is not None:
        s = df.to_string()
        print(s)
        summary.append("```\n" + s + "\n```")
    if text:
        print(text)
        summary.append(text)


def px(base, name):
    return base / f"X_{name}.parquet"


def read_y(name):
    p = LOG1P / f"{Y_FILES[name]}.parquet"
    if not p.exists():
        p = ROOT / f"{Y_FILES[name]}.parquet"
    d = pd.read_parquet(p)
    return d.iloc[:, 0] if isinstance(d, pd.DataFrame) else d


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def fix_columns(X, feature_names):
    if feature_names and len(feature_names) == X.shape[1] and list(X.columns) != list(feature_names):
        if not set(feature_names) <= set(map(str, X.columns)):
            X.columns = feature_names
    return X


# --------------------------- label mapping ---------------------------------
lm = load_json(LOG1P / "label_mapping.json") or load_json(ROOT / "label_mapping.json") or {}
try:
    id2name = {v: k for k, v in lm.items()} if all(isinstance(v, int) for v in lm.values()) \
        else {int(k): v for k, v in lm.items()}
except Exception:
    id2name = {}


def named(y):
    return y.map(id2name).fillna(y) if id2name else y


feat_log1p = load_json(LOG1P / "feature_names.json")
feat_base = load_json(ROOT / "feature_names.json")
log1p_cols = load_json(LOG1P / "log1p_columns.json") or []
if isinstance(log1p_cols, dict):
    log1p_cols = list(log1p_cols.get("columns", log1p_cols.keys()))
twin = load_json(ROOT / "twin_drop_index.json")

# ----------------------- 1. split sizes (metadata only) --------------------
rows = []
for n in ALL_X:
    p = px(LOG1P, n)
    if not p.exists():
        continue
    md = pq.ParquetFile(p).metadata
    group = "supervised" if n in SUP else "anomaly" if n in ANO else "holdout"
    rows.append({"set": n, "group": group, "rows": md.num_rows, "features": md.num_columns,
                 "file_MB": round(p.stat().st_size / 1e6, 1)})
sizes = pd.DataFrame(rows).set_index("set")
sizes["percent_of_group"] = (sizes["rows"] / sizes.groupby("group")["rows"].transform("sum") * 100).round(2)
sizes.to_csv(OUT / "split_sizes.csv")
sup_total = sizes.loc[sizes.group == "supervised", "rows"].sum()
log("1. Split sizes", sizes,
    f"\nSupervised total: {sup_total:,} rows. percent_of_group is the share inside its own group "
    f"(supervised / anomaly / holdout).")

fig, ax = plt.subplots(figsize=(10, 4.5))
colors = sizes["group"].map({"supervised": "#4c72b0", "anomaly": "#55a868", "holdout": "#c44e52"})
ax.bar(sizes.index, sizes["rows"], color=colors)
ax.set_ylabel("rows"); ax.set_title("Rows per dataset (blue: supervised, green: anomaly, red: PortScan holdout)")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig(OUT / "figures/01_split_sizes.png", dpi=200); plt.close()

# ------------------------ 2. class distribution ----------------------------
sup_counts = pd.DataFrame({n: named(read_y(n)).value_counts() for n in SUP})
sup_counts = sup_counts.fillna(0).astype(int).sort_values("train_supervised", ascending=False)
sup_pct = (sup_counts / sup_counts.sum() * 100).round(3)
sup_counts.to_csv(OUT / "class_counts_supervised.csv")
sup_pct.to_csv(OUT / "class_percent_supervised.csv")
gap = (sup_pct.max(axis=1) - sup_pct.min(axis=1)).max()
log("2a. Supervised class counts", sup_counts,
    f"\nLargest gap in a class's share between train/val/test: {gap:.3f} percentage points "
    f"(near 0 means stratified).")

bin_counts = pd.DataFrame({n: read_y(n).value_counts() for n in ["val_anomaly", "test_anomaly"]}).fillna(0).astype(int)
bin_counts.to_csv(OUT / "class_counts_anomaly_binary.csv")
log("2b. Anomaly val/test binary labels (assumed 0 = benign, 1 = attack; confirm)", bin_counts)

hold_counts = named(read_y("heldout_portscan")).value_counts()
log("2c. Held-out set labels", hold_counts.to_frame("rows"))

fig, ax = plt.subplots(figsize=(12, 5))
sup_counts.plot.bar(ax=ax, logy=True)
ax.set_title("Class distribution per supervised split (log scale)"); ax.set_ylabel("samples")
plt.tight_layout(); plt.savefig(OUT / "figures/02_class_distribution.png", dpi=200); plt.close()

# -------------------- 3. pipeline artifacts --------------------------------
art = {
    "features (no-log1p version)": len(feat_base) if feat_base else "n/a",
    "features (log1p version)": len(feat_log1p) if feat_log1p else "n/a",
    "columns transformed with log1p": len(log1p_cols),
    "twin_drop_index entries": len(twin) if hasattr(twin, "__len__") else "n/a",
    "classes in label_mapping": len(id2name),
}
try:
    import joblib
    sc = joblib.load(LOG1P / "scaler.joblib")
    art["scaler type"] = type(sc).__name__
    art["scaler n_features_in_"] = getattr(sc, "n_features_in_", "n/a")
except Exception as e:
    art["scaler"] = f"could not load ({type(e).__name__})"
art_df = pd.DataFrame(art.items(), columns=["item", "value"]).set_index("item")
art_df.to_csv(OUT / "pipeline_artifacts.csv")
extra = ""
if log1p_cols:
    extra += f"\nlog1p columns (first 15): {list(log1p_cols)[:15]}"
if isinstance(twin, list):
    extra += f"\ntwin_drop_index (first 15): {twin[:15]}"
log("3. Pipeline artifacts", art_df, extra)

# ------------- load supervised X for both versions (+ quality) -------------
Xs = {n: fix_columns(pd.read_parquet(px(LOG1P, n)), feat_log1p) for n in SUP}

q = []
for ver, base in [("no_log1p", ROOT), ("log1p", LOG1P)]:
    for n in ALL_X:
        p = px(base, n)
        if not p.exists():
            continue
        X = Xs[n] if (ver == "log1p" and n in Xs) else pd.read_parquet(p)
        q.append({"version": ver, "set": n, "NaN": int(X.isna().sum().sum()),
                  "inf": int(np.isinf(X).sum().sum()), "constant_cols": int((X.nunique() <= 1).sum()),
                  "min": round(float(X.min().min()), 3), "max": round(float(X.max().max()), 3)})
        if not (ver == "log1p" and n in Xs):
            del X
q_df = pd.DataFrame(q).set_index(["version", "set"])
q_df.to_csv(OUT / "data_quality.csv")
log("4. Data quality (both versions)", q_df,
    "\nNaN and inf must be 0. min/max show the value range after scaling.")

# --------------------------- 5. log1p effect -------------------------------
A_full = fix_columns(pd.read_parquet(px(ROOT, "train_supervised")), feat_base)
B_full = Xs["train_supervised"]
common = [c for c in A_full.columns if c in B_full.columns]
rng = np.random.default_rng(SEED)
if len(A_full) == len(B_full):
    idx = np.sort(rng.choice(len(A_full), min(SAMPLE_N, len(A_full)), replace=False))
    A, B = A_full.iloc[idx][common], B_full.iloc[idx][common]
else:
    A = A_full[common].sample(min(SAMPLE_N, len(A_full)), random_state=SEED)
    B = B_full[common].sample(min(SAMPLE_N, len(B_full)), random_state=SEED)

lc = [c for c in log1p_cols if c in common]
non_lc = [c for c in common if c not in lc]
same_rest = None
if len(A) == len(B) and non_lc:
    same_rest = bool(np.allclose(A[non_lc].to_numpy(dtype="float64"), B[non_lc].to_numpy(dtype="float64"), atol=1e-4))


def tail_stats(d):
    return pd.DataFrame({"skew": d.skew(), "kurtosis": d.kurt(), "max_abs_z": d.abs().max(),
                         "pct_beyond_5": (d.abs() > 5).mean() * 100})


tb, ta = tail_stats(A), tail_stats(B)
cmp_df = pd.DataFrame({
    "skew_before": tb["skew"], "skew_after": ta["skew"],
    "kurt_before": tb["kurtosis"], "kurt_after": ta["kurtosis"],
    "max_abs_z_before": tb["max_abs_z"], "max_abs_z_after": ta["max_abs_z"],
    "pct_beyond_5_before": tb["pct_beyond_5"], "pct_beyond_5_after": ta["pct_beyond_5"],
})
cmp_df["log1p_applied"] = cmp_df.index.isin(lc)
cmp_df = cmp_df.sort_values("skew_before", ascending=False)
cmp_df.to_csv(OUT / "log1p_before_after_stats.csv")

L = cmp_df[cmp_df.log1p_applied]
msg = (f"\nSkewness is unchanged by linear scaling, so comparing the two scaled versions is valid.\n"
       f"Features: {len(common)}; log1p applied to {len(lc)}.\n"
       f"On the log1p columns: median |skew| {L['skew_before'].abs().median():.2f} -> {L['skew_after'].abs().median():.2f}; "
       f"median kurtosis {L['kurt_before'].median():.1f} -> {L['kurt_after'].median():.1f}; "
       f"median max |z| {L['max_abs_z_before'].median():.1f} -> {L['max_abs_z_after'].median():.1f}; "
       f"median % of values beyond 5 std {L['pct_beyond_5_before'].median():.3f} -> {L['pct_beyond_5_after'].median():.3f}.")
if same_rest is not None:
    msg += f"\nColumns NOT in log1p_columns identical between versions (sanity check): {same_rest}."
log("5. log1p effect (no_log1p vs log1p, train_supervised sample)", L.head(15).round(3), msg)

top = L.head(15)
fig, ax = plt.subplots(figsize=(11, 6))
i = np.arange(len(top)); w = 0.4
ax.barh(i - w / 2, top["skew_before"], w, label="before log1p", color="#c44e52")
ax.barh(i + w / 2, top["skew_after"], w, label="after log1p", color="#4c72b0")
ax.set_yticks(i); ax.set_yticklabels(top.index); ax.invert_yaxis()
ax.set_xlabel("skewness"); ax.set_title("Skewness of the 15 most skewed log1p features"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "figures/03_skewness_before_after.png", dpi=200); plt.close()

feats = list(L.index[:TOP_K])
if feats:
    fig, axes = plt.subplots(len(feats), 2, figsize=(11, 2.6 * len(feats)))
    axes = np.atleast_2d(axes)
    for r, f in enumerate(feats):
        axes[r, 0].hist(A[f].clip(-5, 60), bins=80, color="#c44e52")
        axes[r, 0].set_title(f"{f} - before log1p (clipped at 60 std)", fontsize=9)
        axes[r, 1].hist(B[f], bins=80, color="#4c72b0")
        axes[r, 1].set_title(f"{f} - after log1p", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "figures/04_histograms_before_after.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(L["max_abs_z_before"], L["max_abs_z_after"], alpha=0.7)
lim = [1, max(L["max_abs_z_before"].max(), 10)]
ax.plot(lim, lim, "k--", lw=1, label="no change")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("max |z-score| before log1p"); ax.set_ylabel("max |z-score| after log1p")
ax.set_title("Outlier extremeness per feature"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "figures/05_outlier_extremeness.png", dpi=200); plt.close()

# ---------------------------- 6. leakage checks ----------------------------
def hashes(X):
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


H = {n: hashes(Xs[n]) for n in SUP}
leak = []
for n in SUP:
    leak.append({"check": f"duplicate rows inside {n}", "value": len(H[n]) - len(np.unique(H[n]))})
for a, b in [("train_supervised", "val_supervised"), ("train_supervised", "test_supervised"),
             ("val_supervised", "test_supervised")]:
    inter = np.isin(H[b], H[a])
    leak.append({"check": f"rows of {b} also present in {a}", "value": int(inter.sum())})
hold_names = set(named(read_y("heldout_portscan")).unique())
train_names = set(named(read_y("train_supervised")).unique())
leak.append({"check": f"holdout labels {sorted(map(str, hold_names))} present in train", "value": len(hold_names & train_names)})
for n in SUP:
    X = Xs[n]
    leak.append({"check": f"{n}: mean |column mean| / mean column std",
                 "value": f"{X.mean().abs().mean():.4f} / {X.std().mean():.4f}"})
leak_df = pd.DataFrame(leak).set_index("check")
leak_df.to_csv(OUT / "leakage_checks.csv")
log("6. Leakage checks", leak_df,
    "\nA scaler fitted on train only gives train mean ~0 and std ~1, with val/test close but not exact. "
    "Identical rows across splits are normal in CICIDS2017 (repeated flows); report the count and, "
    "if large, mention it as a limitation. The holdout label must NOT appear in train (value 0).")

# --------------------------- 7. parquet vs csv -----------------------------
bench_df = Xs["train_supervised"].head(500_000)
csv_p, pq_p = OUT / "_b.csv", OUT / "_b.parquet"
cols10 = list(bench_df.columns[:10])


def timed(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


_, w_c = timed(lambda: bench_df.to_csv(csv_p, index=False))
_, w_p = timed(lambda: bench_df.to_parquet(pq_p, compression="snappy"))
cf, r_c = timed(lambda: pd.read_csv(csv_p))
pf, r_p = timed(lambda: pd.read_parquet(pq_p))
_, rc10 = timed(lambda: pd.read_csv(csv_p, usecols=cols10))
_, rp10 = timed(lambda: pd.read_parquet(pq_p, columns=cols10))
bench = pd.DataFrame({
    "csv": [csv_p.stat().st_size / 1e6, w_c, r_c, rc10, cf.memory_usage(deep=True).sum() / 1e6],
    "parquet": [pq_p.stat().st_size / 1e6, w_p, r_p, rp10, pf.memory_usage(deep=True).sum() / 1e6],
}, index=["file_size_MB", "write_time_s", "read_all_columns_s", "read_10_columns_s", "memory_after_load_MB"]).round(3)
bench["csv_over_parquet"] = (bench["csv"] / bench["parquet"]).round(2)
bench.to_csv(OUT / "parquet_vs_csv.csv")
dtype_ok = bool((pf.dtypes.values == bench_df.dtypes.values).all())
csv_p.unlink(missing_ok=True); pq_p.unlink(missing_ok=True)
log("7. Parquet vs CSV (first 500k rows of X_train_supervised)", bench,
    f"\nParquet round-trips dtypes exactly: {dtype_ok} (CSV reloads as float64 and loses float32).")

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, m in zip(ax, ["file_size_MB", "read_all_columns_s", "read_10_columns_s"]):
    bench.loc[m, ["csv", "parquet"]].plot.bar(ax=a, color=["#c44e52", "#4c72b0"])
    a.set_title(m); a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(OUT / "figures/06_parquet_vs_csv.png", dpi=200); plt.close()

# ------------------------------ summary.md ---------------------------------
summary.append("""
## Reasons to cite in the report (edit to match what you did)

**Why separate supervised and anomaly datasets:** Random Forest and XGBoost need labelled
data from all classes (supervised sets). The Autoencoder and Isolation Forest learn what
normal traffic looks like, so their training set (X_train_anomaly_benign) contains benign
flows only, while their val/test sets contain benign and attack flows with binary labels.

**Why train/val/test:** train fits the models, validation is used for tuning and threshold
selection, test is used once for the final score. The split is stratified (see
class_percent_supervised.csv), so rare classes appear in every split.

**Why a PortScan holdout:** PortScan is excluded from training and evaluated separately
to measure detection of an attack type the models have never seen.

**Why log1p:** flow features (durations, byte and packet counts, inter-arrival times) are
heavily right-skewed with a few huge values. log(1+x) compresses the tail, preserves
ordering, and is defined at 0. After log1p the skewness, kurtosis and extreme z-scores
drop (see figures 03-05), which helps the Autoencoder and Isolation Forest. Tree models
(Random Forest, XGBoost) are invariant to monotonic transforms, so it does not change them.

**Why scaling fitted on train only:** prevents information from val/test leaking into
training; the scaler is saved (scaler.joblib) so the same transform is applied at inference.

**Why Parquet:** compressed columnar format, so files are smaller and load faster than CSV,
dtypes are preserved with no re-parsing, and single columns can be read without loading the
whole file (see figure 06 and parquet_vs_csv.csv).
""")
(OUT / "summary.md").write_text("# NIDS preprocessing analysis\n" + "\n".join(summary), encoding="utf-8")
print(f"\nDone. Everything saved in: {OUT.resolve()}")