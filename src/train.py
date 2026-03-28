"""
train.py - Train all regression models for MPG prediction.

Models:
  1. Linear Regression (baseline)
  2. Forward Stepwise Regression (uses forward-selected features)
  3. Lasso Regression  (with LassoCV for alpha tuning)
  4. Ridge Regression  (with RidgeCV for alpha tuning)
  5. Principal Component Regression (PCR)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from src.utils import print_section


def train_linear_regression(X_train, y_train):
    """Baseline OLS Linear Regression."""
    print_section("TRAINING: Linear Regression (Baseline)")
    model = LinearRegression()
    model.fit(X_train, y_train)
    _print_cv(model, X_train, y_train)
    return model


def train_forward_stepwise_regression(X_train, y_train, selected_features: list):
    """
    Linear Regression using only the features chosen by forward selection.
    Returns (model, selected_features) so the evaluator knows which columns to use.
    """
    print_section("TRAINING: Forward Stepwise Regression")
    print(f"  Using features: {selected_features}")
    model = LinearRegression()
    model.fit(X_train[selected_features], y_train)
    _print_cv(model, X_train[selected_features], y_train)
    return model, selected_features


def train_lasso_regression(X_train, y_train, cv: int = 5):
    """Lasso Regression with built-in cross-validated alpha selection."""
    print_section("TRAINING: Lasso Regression (LassoCV)")
    alphas = np.logspace(-4, 2, 100)
    model = LassoCV(alphas=alphas, cv=cv, max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    print(f"  Best alpha: {model.alpha_:.6f}")
    non_zero = np.sum(model.coef_ != 0)
    print(f"  Non-zero coefficients: {non_zero}/{len(model.coef_)}")
    _print_cv(model, X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train, cv: int = 5):
    """Ridge Regression (Bayesian interpretation) with cross-validated alpha."""
    print_section("TRAINING: Ridge Regression (RidgeCV)")
    alphas = np.logspace(-4, 4, 100)
    model = RidgeCV(alphas=alphas, cv=cv, scoring="r2")
    model.fit(X_train, y_train)
    print(f"  Best alpha: {model.alpha_:.6f}")
    _print_cv(model, X_train, y_train)
    return model


def train_pcr(X_train, y_train, n_components: int | None = None):
    """
    Principal Component Regression: StandardScaler → PCA → LinearRegression.

    If n_components is None, choose the number of components that explains
    ≥ 95 % of variance.
    """
    print_section("TRAINING: Principal Component Regression (PCR)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Determine optimal n_components
    if n_components is None:
        pca_full = PCA().fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.argmax(cumvar >= 0.95)) + 1
        print(f"  Auto-selected n_components = {n_components} (≥ 95% variance)")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("regressor", LinearRegression()),
    ])
    pipeline.fit(X_train, y_train)

    # Print explained variance
    pca_step = pipeline.named_steps["pca"]
    total_var = sum(pca_step.explained_variance_ratio_) * 100
    print(f"  Explained variance ({n_components} PCs): {total_var:.2f}%")
    _print_cv(pipeline, X_train, y_train)
    return pipeline


# ── Helper ────────────────────────────────────────────────────────────────────

def _print_cv(model, X, y, cv: int = 5):
    """Print 5-fold cross-validation R² for quick sanity check."""
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    print(f"  5-Fold CV R²: {scores.mean():.4f} ± {scores.std():.4f}")


# ── Master training function ─────────────────────────────────────────────────

def train_all_models(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     forward_features: list) -> dict:
    """
    Train every model and return a dict of {name: model_or_tuple}.

    For the forward stepwise model the value is (model, selected_features).
    """
    models = {}

    models["Linear Regression"] = train_linear_regression(X_train, y_train)

    fwd_model, fwd_feats = train_forward_stepwise_regression(
        X_train, y_train, forward_features
    )
    models["Forward Stepwise"] = (fwd_model, fwd_feats)

    models["Lasso Regression"] = train_lasso_regression(X_train, y_train)
    models["Ridge Regression"] = train_ridge_regression(X_train, y_train)
    models["PCR"] = train_pcr(X_train, y_train)

    return models
