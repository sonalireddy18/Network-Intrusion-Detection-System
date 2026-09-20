"""
Week 2 — CICIDS2017 data-quality + initial EDA pipeline.

Run from the repository root:
    python -m src.preprocessing.cleaning

Expected input:
    data/raw/*.csv

The script does not train any model. It produces the evidence needed
before Week 3 preprocessing/model development.
"""

from pathlib import Path
import rep
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CICIDS2017's target column is normally "Label", sometimes with whitespace.
LABEL_CANDIDATES = {"label", " Label", "Label"}


#1
#---------------------------------------------------------------
def clean_column_names(columns):
    """Strip whitespace and remove duplicate column names safely."""
    cleaned = [str(c).strip() for c in columns]
    seen = {}
    result = []
    for c in cleaned:
        n = seen.get(c, 0)
        result.append(c if n == 0 else f"{c}_{n}")
        seen[c] = n + 1
    return result

#2
#---------------------------------------------------------------
def find_label_column(columns):
    for c in columns:
        if str(c).strip().lower() == "label":
            return c
    raise ValueError(
        "Could not find the CICIDS2017 target column 'Label'. "
        f"Columns found: {list(columns)[:20]}"
    )

#3
#---------------------------------------------------------------
def numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

#4
#---------------------------------------------------------------
def safe_numeric(df):
    """Convert numeric-looking columns to numeric without touching Label."""
    out = df.copy()
    label = find_label_column(out.columns)
    for c in out.columns:
        if c != label:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

#5
#---------------------------------------------------------------
def load_cicids():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}. "
            "Extract the CICIDS2017 CSV files into data/raw/ first."
        )

    frames = []
    print(f"Found {len(files)} CSV file(s).")
    for f in files:
        print(f"Reading: {f.name}")
        df = pd.read_csv(f, low_memory=False)
        df.columns = clean_column_names(df.columns)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data = safe_numeric(data)
    return data

#6
#---------------------------------------------------------------
def basic_summary(df):
    label = find_label_column(df.columns)
    n_rows, n_cols = df.shape
    num_cols = numeric_columns(df.drop(columns=[label]))
    cat_cols = [c for c in df.columns if c not in num_cols and c != label]

    summary = pd.DataFrame({
        "item": [
            "rows", "columns", "numeric_features",
            "categorical_non_target_features", "target_column",
            "unique_target_labels"
        ],
        "value": [
            n_rows, n_cols, len(num_cols), len(cat_cols),
            label, df[label].nunique(dropna=True)
        ]
    })
    summary.to_csv(OUT_DIR / "dataset_summary.csv", index=False)

    dtype_df = pd.DataFrame({
        "feature": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_values": [int(df[c].isna().sum()) for c in df.columns],
        "missing_percent": [
            round(float(df[c].isna().mean() * 100), 4) for c in df.columns
        ]
    })
    dtype_df.to_csv(OUT_DIR / "feature_summary.csv", index=False)

    print("\nDATASET SUMMARY")
    print(f"Rows: {n_rows:,}")
    print(f"Columns: {n_cols}")
    print(f"Numeric features: {len(num_cols)}")
    print(f"Categorical non-target features: {len(cat_cols)}")
    print(f"Target: {label}")
    return label

#7
#---------------------------------------------------------------
def label_distribution(df, label):
    counts = (
        df[label]
        .fillna("MISSING_LABEL")
        .astype(str)
        .str.strip()
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
    )
    counts["percentage"] = (counts["count"] / len(df) * 100).round(4)
    counts.to_csv(OUT_DIR / "attack_distribution.csv", index=False)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=counts, x="count", y="label")
    plt.title("CICIDS2017 Label Distribution")
    plt.xlabel("Number of flows")
    plt.ylabel("Label")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "attack_distribution.png", dpi=200)
    plt.close()

    print("\nLABEL DISTRIBUTION")
    print(counts.to_string(index=False))
    return counts

#8
#---------------------------------------------------------------
def data_quality(df, label):
    # Infinite values are only meaningful for numeric columns.
    nums = numeric_columns(df.drop(columns=[label]))
    inf_counts = pd.Series(
        np.isinf(df[nums]).sum(axis=0), index=nums, name="infinite_values"
    )
    missing_counts = df.isna().sum().rename("missing_values")

    quality = pd.concat([missing_counts, inf_counts], axis=1).fillna(0)
    quality["missing_values"] = quality["missing_values"].astype(int)
    quality["infinite_values"] = quality["infinite_values"].astype(int)
    quality["total_problem_values"] = (
        quality["missing_values"] + quality["infinite_values"]
    )
    quality.to_csv(OUT_DIR / "data_quality.csv")

    # Duplicate count across the complete concatenated dataframe.
    duplicate_count = int(df.duplicated().sum())
    invalid_label_count = int(df[label].isna().sum())

    with open(OUT_DIR / "cleaning_report.txt", "w", encoding="utf-8") as f:
        f.write("CICIDS2017 Week 2 Data Quality Report\n")
        f.write("=" * 45 + "\n")
        f.write(f"Rows: {len(df):,}\n")
        f.write(f"Columns: {df.shape[1]}\n")
        f.write(f"Duplicate rows: {duplicate_count:,}\n")
        f.write(f"Rows with missing target label: {invalid_label_count:,}\n")
        f.write(f"Total missing cells: {int(df.isna().sum().sum()):,}\n")
        f.write(f"Total infinite numeric cells: {int(np.isinf(df[nums]).sum().sum()):,}\n")

    print("\nDATA QUALITY")
    print(f"Duplicate rows: {duplicate_count:,}")
    print(f"Missing target labels: {invalid_label_count:,}")
    print(f"Missing cells: {int(df.isna().sum().sum()):,}")
    print(f"Infinite numeric cells: {int(np.isinf(df[nums]).sum().sum()):,}")
    return quality

#9
#---------------------------------------------------------------
def feature_eda(df, label):
    nums = numeric_columns(df.drop(columns=[label]))

    # Replace inf with NaN for descriptive statistics/correlation.
    work = df[nums].replace([np.inf, -np.inf], np.nan)

    desc = work.describe().T
    desc["missing"] = work.isna().sum()
    desc.to_csv(OUT_DIR / "numeric_feature_statistics.csv")

    # Correlation matrix — limit displayed features to the first 25 numeric
    # features to keep the figure readable. The full matrix is saved separately.
    corr = work.corr(numeric_only=True)
    corr.to_csv(OUT_DIR / "correlation_matrix.csv")

    display_cols = nums[:25]
    if len(display_cols) >= 2:
        plt.figure(figsize=(16, 12))
        sns.heatmap(
            corr.loc[display_cols, display_cols],
            cmap="coolwarm",
            center=0,
            linewidths=0.1
        )
        plt.title("CICIDS2017 Feature Correlation — First 25 Numeric Features")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "correlation_heatmap.png", dpi=200)
        plt.close()

    # Distributions for up to 12 features.
    plot_cols = nums[:12]
    for col in plot_cols:
        values = work[col].dropna()
        if values.empty:
            continue
        # Sampling makes plots fast while retaining the distribution shape.
        if len(values) > 100_000:
            values = values.sample(100_000, random_state=42)
        plt.figure(figsize=(9, 5))
        sns.histplot(values, bins=50, kde=False)
        plt.title(f"Distribution: {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", col)
        plt.savefig(OUT_DIR / f"distribution_{safe}.png", dpi=160)
        plt.close()

#10
#---------------------------------------------------------------
def main():
    start = time.time()
    df = load_cicids()

    # Normalize labels once so later experiments do not inherit whitespace.
    label = find_label_column(df.columns)
    df[label] = df[label].astype("string").str.strip()

    basic_summary(df)
    label_distribution(df, label)
    data_quality(df, label)
    feature_eda(df, label)

    print(f"\nWeek 2 EDA completed in {time.time() - start:.1f} seconds.")
    print(f"Outputs: {OUT_DIR}")

if __name__ == "__main__":
    main()
