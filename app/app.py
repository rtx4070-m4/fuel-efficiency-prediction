"""
app.py - Streamlit web application for MPG Prediction.

Usage:
    streamlit run app/app.py

Provides input fields for vehicle characteristics and returns a predicted MPG.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Resolve paths ─────────────────────────────────────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "saved_model.pkl")

sys.path.insert(0, PROJECT_ROOT)


# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifact():
    """Load the saved model artifact from disk."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "Model file not found. Please run `python main.py` first to train "
            "and save the model."
        )
        st.stop()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPG Prediction",
    page_icon="🚗",
    layout="centered",
)

st.title("🚗 Automobile Fuel Efficiency Predictor")
st.markdown(
    "Enter vehicle characteristics below and click **Predict** to estimate "
    "the fuel efficiency (Miles Per Gallon)."
)

artifact = load_artifact()
model = artifact["model"]
feature_names = artifact["features"]

# ── Sidebar: model info ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write(f"**Best model:** {artifact['name']}")
    st.write("**Features used:**")
    for f in feature_names:
        st.write(f"  • {f}")

    st.divider()
    st.subheader("📊 Model Comparison")
    results_df = pd.DataFrame(artifact["results"])
    st.dataframe(results_df, hide_index=True)

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Enter Vehicle Details")

col1, col2 = st.columns(2)

with col1:
    cylinders = st.selectbox("Cylinders", options=[3, 4, 5, 6, 8], index=1)
    displacement = st.slider("Displacement (cu. inches)", 50.0, 500.0, 150.0, 1.0)
    horsepower = st.slider("Horsepower", 40.0, 250.0, 100.0, 1.0)

with col2:
    weight = st.slider("Weight (lbs)", 1500, 5500, 3000, 10)
    acceleration = st.slider("Acceleration (sec 0-60)", 8.0, 25.0, 15.0, 0.1)
    model_year = st.slider("Model Year (70-82)", 70, 82, 76)

origin = st.radio("Origin", options=["USA (1)", "Europe (2)", "Japan (3)"], horizontal=True)

# ── Build feature vector ──────────────────────────────────────────────────────

def build_input():
    """Construct a single-row DataFrame matching the training feature set."""
    origin_map = {"USA (1)": 1, "Europe (2)": 2, "Japan (3)": 3}
    origin_val = origin_map[origin]

    data = {
        "cylinders": cylinders,
        "displacement": displacement,
        "horsepower": horsepower,
        "weight": weight,
        "acceleration": acceleration,
        "model year": model_year,
        "origin_1": 1 if origin_val == 1 else 0,
        "origin_2": 1 if origin_val == 2 else 0,
        "origin_3": 1 if origin_val == 3 else 0,
    }

    row = pd.DataFrame([data])

    # Ensure column order matches training
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]
    return row


# ── Predict ───────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔮 Predict MPG", use_container_width=True):
    input_df = build_input()
    prediction = model.predict(input_df)[0]

    st.success(f"### Predicted Fuel Efficiency: **{prediction:.2f} MPG**")

    # Quick interpretation
    if prediction >= 30:
        st.info("This vehicle would be considered **fuel-efficient** for its era.")
    elif prediction >= 20:
        st.info("This vehicle has **average** fuel efficiency.")
    else:
        st.info("This vehicle has **below-average** fuel efficiency.")

st.markdown("---")
st.caption(
    "Project: Automobile Fuel Efficiency Prediction · "
    "Rishiraj Karn & Ritam Rabha · IIT Ropar · Prof. Jayram Vallaru"
)
