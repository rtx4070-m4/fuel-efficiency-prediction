"""
data_preprocessing.py - Data cleaning, transformation, and outlier removal.

Pipeline:
  1. Fix '?' placeholders in horsepower → numeric
  2. Drop rows with remaining NAs
  3. Drop the 'car name' column
  4. One-hot encode 'origin' (1/2/3 → origin_1, origin_2, origin_3)
  5. Remove outliers using Median ± 3 × MAD
"""

import pandas as pd
import numpy as np
from src.utils import print_section, print_dataframe_info


# ── Step 1-4: Core cleaning ──────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform all basic cleaning steps and return a fully numeric DataFrame.

    Steps:
      • Convert 'horsepower' from object → float (replace '?' with NaN)
      • Drop rows that still contain NaN
      • Drop the 'car name' column
      • One-hot encode 'origin'
    """
    print_section("DATA PREPROCESSING")
    df = df.copy()

    # ── Horsepower fix ────────────────────────────────────────────────────
    if "horsepower" in df.columns:
        df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
        n_missing = df["horsepower"].isnull().sum()
        print(f"[INFO] Converted 'horsepower' to numeric – {n_missing} missing values introduced.")

    # ── Drop NAs ──────────────────────────────────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Dropped {before - len(df)} rows with missing values. Remaining: {len(df)}")

    # ── Drop 'car name' ──────────────────────────────────────────────────
    if "car name" in df.columns:
        df.drop(columns=["car name"], inplace=True)
        print("[INFO] Dropped 'car name' column.")

    # ── One-hot encode 'origin' ──────────────────────────────────────────
    if "origin" in df.columns:
        df["origin"] = df["origin"].astype(int)
        origin_dummies = pd.get_dummies(df["origin"], prefix="origin", dtype=int)
        df = pd.concat([df.drop(columns=["origin"]), origin_dummies], axis=1)
        print(f"[INFO] One-hot encoded 'origin' → {list(origin_dummies.columns)}")

    print_dataframe_info(df, "Cleaned Data")
    return df


# ── Step 5: Outlier removal via Median ± 3 × MAD ────────────────────────────

def remove_outliers_mad(df: pd.DataFrame, target_col: str = "mpg",
                        threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove rows where ANY numeric predictor falls outside
    Median ± threshold × MAD (Median Absolute Deviation).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (fully numeric).
    target_col : str
        Name of the target variable (excluded from outlier detection).
    threshold : float
        Number of MADs to allow around the median (default 3).

    Returns
    -------
    pd.DataFrame with outlier rows removed.
    """
    print_section("OUTLIER REMOVAL (Median ± 3 × MAD)")
    df = df.copy()

    # Identify numeric predictor columns (exclude target & binary dummies)
    predictor_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != target_col and df[c].nunique() > 2
    ]

    mask = pd.Series(True, index=df.index)

    for col in predictor_cols:
        median = df[col].median()
        mad = np.median(np.abs(df[col] - median))
        # Avoid zero MAD (constant column)
        if mad == 0:
            continue
        lower = median - threshold * mad
        upper = median + threshold * mad
        col_mask = df[col].between(lower, upper)
        outliers = (~col_mask).sum()
        if outliers > 0:
            print(f"  {col}: median={median:.2f}, MAD={mad:.2f}, "
                  f"range=[{lower:.2f}, {upper:.2f}] → {outliers} outliers")
        mask &= col_mask

    before = len(df)
    df = df.loc[mask].reset_index(drop=True)
    print(f"\n[INFO] Removed {before - len(df)} outlier rows. Remaining: {len(df)}")
    return df


# ── Convenience pipeline ─────────────────────────────────────────────────────

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline: clean → remove outliers."""
    df = clean_data(df)
    df = remove_outliers_mad(df)
    return df
