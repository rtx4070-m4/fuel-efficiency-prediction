"""
feature_selection.py - Feature selection techniques for MPG prediction.

Implements:
  1. Correlation-based selection  (|corr with target| > 0.5)
  2. Forward Stepwise Selection   (greedy addition based on R² improvement)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from src.utils import print_section


# ── 1. Correlation-based selection ────────────────────────────────────────────

def correlation_based_selection(df: pd.DataFrame,
                                target_col: str = "mpg",
                                threshold: float = 0.5) -> list:
    """
    Select features whose absolute Pearson correlation with the target
    exceeds `threshold`.

    Returns
    -------
    list of selected feature names (sorted by descending |correlation|).
    """
    print_section("CORRELATION-BASED FEATURE SELECTION")

    # Compute correlations with the target
    correlations = df.corr(numeric_only=True)[target_col].drop(target_col)
    abs_corr = correlations.abs().sort_values(ascending=False)

    print("\nCorrelation with MPG:")
    for feat, val in abs_corr.items():
        marker = " ✓" if val >= threshold else ""
        print(f"  {feat:20s}  {correlations[feat]:+.4f}  (|r| = {val:.4f}){marker}")

    selected = abs_corr[abs_corr >= threshold].index.tolist()
    print(f"\n[INFO] Selected {len(selected)} features (|corr| ≥ {threshold}): {selected}")
    return selected


# ── 2. Forward Stepwise Selection ─────────────────────────────────────────────

def forward_stepwise_selection(X: pd.DataFrame,
                                y: pd.Series,
                                max_features: int | None = None) -> list:
    """
    Greedy forward selection: at each step add the feature that yields the
    largest R² improvement on the training set.

    Parameters
    ----------
    X : pd.DataFrame  – predictor matrix
    y : pd.Series     – target vector
    max_features : int or None – stop after this many features (default: all)

    Returns
    -------
    list of feature names in the order they were selected.
    """
    print_section("FORWARD STEPWISE SELECTION")

    remaining = list(X.columns)
    selected: list[str] = []
    best_r2 = -np.inf

    if max_features is None:
        max_features = len(remaining)

    for step in range(1, max_features + 1):
        best_candidate = None
        for candidate in remaining:
            trial = selected + [candidate]
            model = LinearRegression().fit(X[trial], y)
            score = r2_score(y, model.predict(X[trial]))
            if score > best_r2:
                best_r2 = score
                best_candidate = candidate

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)
        print(f"  Step {step}: + {best_candidate:20s}  →  R² = {best_r2:.6f}")

    print(f"\n[INFO] Forward selection chose {len(selected)} features: {selected}")
    return selected


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_feature_selection(df: pd.DataFrame, target_col: str = "mpg"):
    """
    Run both selection methods and return a dict of results.

    Returns
    -------
    dict with keys 'correlation' and 'forward', each mapping to a list of
    selected feature names.
    """
    corr_features = correlation_based_selection(df, target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col]
    fwd_features = forward_stepwise_selection(X, y)

    # Summary comparison
    print_section("FEATURE SELECTION COMPARISON")
    print(f"  Correlation-based : {corr_features}")
    print(f"  Forward Stepwise  : {fwd_features}")
    overlap = set(corr_features) & set(fwd_features)
    print(f"  Overlap           : {sorted(overlap)}")

    return {"correlation": corr_features, "forward": fwd_features}
