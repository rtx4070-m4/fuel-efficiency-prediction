"""
main.py - End-to-end pipeline for the Automobile MPG Prediction project.

Usage:
    python main.py

Steps:
  1. Load data
  2. Preprocess (clean + outlier removal)
  3. Feature selection
  4. Train/test split
  5. Train all models
  6. Evaluate & compare
  7. Save best model
"""

import sys
import os
import warnings

# Ensure project root is on sys.path so `src.*` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    load_csv, save_model, get_data_path, get_model_path, print_section
)
from src.data_preprocessing import preprocess_pipeline
from src.feature_selection import run_feature_selection
from src.train import train_all_models
from src.evaluate import evaluate_all_models, get_best_model


def main():
    print_section("AUTOMOBILE FUEL EFFICIENCY (MPG) PREDICTION")
    print("  Authors : Rishiraj Karn (2025DSS1020), Ritam Rabha (2025DSS1021)")
    print("  Guide   : Prof. Jayram Vallaru")
    print("  Institute: IIT Ropar")

    # ── 1. Load data ──────────────────────────────────────────────────────
    raw_df = load_csv(get_data_path("auto-mpg.csv"))

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    df = preprocess_pipeline(raw_df)

    # ── 3. Feature selection ──────────────────────────────────────────────
    selection = run_feature_selection(df, target_col="mpg")

    # ── 4. Train / test split ─────────────────────────────────────────────
    print_section("TRAIN / TEST SPLIT")
    X = df.drop(columns=["mpg"])
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Test set     : {X_test.shape[0]} samples")

    # ── 5. Train all models ───────────────────────────────────────────────
    models = train_all_models(X_train, y_train, selection["forward"])

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    results_df = evaluate_all_models(models, X_test, y_test)

    # ── 7. Save best model ────────────────────────────────────────────────
    best_model, best_name = get_best_model(models, results_df)
    print_section("SAVING BEST MODEL")
    print(f"  Best model: {best_name}")

    # Save a dict with model + metadata so the Streamlit app has everything
    artifact = {
        "model": best_model,
        "name": best_name,
        "features": list(X.columns),
        "results": results_df.to_dict(orient="records"),
    }
    save_model(artifact, get_model_path("saved_model.pkl"))

    # Also save the results table as CSV
    results_path = os.path.join(os.path.dirname(get_model_path()), "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Results table saved → {results_path}")

    print_section("PIPELINE COMPLETE")
    print("  Next steps:")
    print("    • Review notebooks/eda.ipynb for exploratory analysis")
    print("    • Run the web app:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
