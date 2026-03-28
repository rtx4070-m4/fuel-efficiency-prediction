"""
utils.py - Utility functions for the MPG Prediction Project.
Contains helpers for logging, path management, and common operations.
"""

import os
import pickle
import pandas as pd
import numpy as np


# ── Path helpers ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_data_path(filename: str = "auto-mpg.csv") -> str:
    """Return absolute path to a file inside the data/ directory."""
    return os.path.join(DATA_DIR, filename)


def get_model_path(filename: str = "saved_model.pkl") -> str:
    """Return absolute path to a file inside the models/ directory."""
    return os.path.join(MODELS_DIR, filename)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with basic validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def save_model(model, filepath: str) -> None:
    """Persist a model (or any Python object) to disk with pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] Model saved → {filepath}")


def load_model(filepath: str):
    """Load a pickled model from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    print(f"[INFO] Model loaded ← {filepath}")
    return model


# ── Display helpers ───────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    """Print a formatted section header for console output."""
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_dataframe_info(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """Print concise info about a DataFrame."""
    print(f"\n[{label}]  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    missing = df.isnull().sum()
    if missing.any():
        print(f"  Missing values:\n{missing[missing > 0].to_string()}")
    else:
        print("  Missing values: None")
