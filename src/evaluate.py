"""
evaluate.py - Evaluate all trained models and produce a comparison table.

Metrics computed for each model:
  • RMSE  (Root Mean Squared Error)
  • R²    (Coefficient of Determination)
  • MAE   (Mean Absolute Error)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.utils import print_section


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   feature_subset: list | None = None) -> dict:
    """
    Evaluate a single model on the test set.

    Parameters
    ----------
    model : fitted estimator
    X_test : test features
    y_test : test target
    feature_subset : if not None, restrict X_test to these columns
                     (used for Forward Stepwise model)

    Returns
    -------
    dict with keys 'RMSE', 'R2', 'MAE'.
    """
    X = X_test[feature_subset] if feature_subset else X_test
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return {"RMSE": round(rmse, 4), "R2": round(r2, 4), "MAE": round(mae, 4)}


def evaluate_all_models(models: dict,
                        X_test: pd.DataFrame,
                        y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate every model in `models` and return a comparison DataFrame.

    `models` values can be:
      • a fitted estimator, or
      • a tuple (estimator, feature_list)  (for Forward Stepwise).
    """
    print_section("MODEL EVALUATION ON TEST SET")

    results = []

    for name, obj in models.items():
        if isinstance(obj, tuple):
            model, feats = obj
            metrics = evaluate_model(model, X_test, y_test, feature_subset=feats)
        else:
            metrics = evaluate_model(obj, X_test, y_test)

        metrics["Model"] = name
        results.append(metrics)

    df = pd.DataFrame(results)[["Model", "RMSE", "R2", "MAE"]]
    df.sort_values("R2", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\n" + df.to_string(index=False))

    best = df.iloc[0]
    print(f"\n★  Best model: {best['Model']}  "
          f"(R² = {best['R2']}, RMSE = {best['RMSE']}, MAE = {best['MAE']})")

    return df


def get_best_model(models: dict, results_df: pd.DataFrame):
    """
    Return the best model object (and its name) based on the results table.
    """
    best_name = results_df.iloc[0]["Model"]
    best_obj = models[best_name]

    # Unwrap tuple if forward stepwise
    if isinstance(best_obj, tuple):
        return best_obj[0], best_name
    return best_obj, best_name
