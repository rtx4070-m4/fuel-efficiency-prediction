# 🚗 Automobile Fuel Efficiency (MPG) Prediction

**Authors:** Rishiraj Karn (2025DSS1020), Ritam Rabha (2025DSS1021)  
**Submitted to:** Prof. Jayram Vallaru  
**Institute:** Indian Institute of Technology, Ropar

---

## Project Overview

This project builds a complete machine-learning pipeline to **predict automobile fuel efficiency (Miles Per Gallon)** from vehicle characteristics such as engine displacement, horsepower, weight, number of cylinders, model year, and country of origin.

Five regression models are trained and compared:

| # | Model | Description |
|---|-------|-------------|
| 1 | **Linear Regression** | OLS baseline |
| 2 | **Forward Stepwise Regression** | Greedy feature addition based on R² |
| 3 | **Lasso Regression** | L1-regularised with cross-validated α |
| 4 | **Ridge Regression** | L2-regularised (Bayesian) with cross-validated α |
| 5 | **Principal Component Regression** | PCA dimensionality reduction + OLS |

The best model is persisted and served via a **Streamlit web application** for interactive predictions.

---

## Project Structure

```
project/
├── data/
│   └── auto-mpg.csv            # Raw dataset
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Cleaning, encoding, outlier removal
│   ├── feature_selection.py     # Correlation & forward stepwise selection
│   ├── train.py                 # Train all 5 models
│   ├── evaluate.py              # RMSE / R² / MAE comparison
│   └── utils.py                 # Helpers (I/O, paths, display)
├── models/
│   ├── saved_model.pkl          # Best model artifact (after training)
│   └── results.csv              # Model comparison table
├── app/
│   └── app.py                   # Streamlit web app
├── main.py                      # End-to-end pipeline entry point
├── generate_notebook.py         # Script to create the EDA notebook
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Setup Instructions

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Train all models

```bash
python main.py
```

This runs the full pipeline: data loading → preprocessing → outlier removal → feature selection → model training → evaluation → saves the best model to `models/saved_model.pkl`.

### Launch the web app

```bash
streamlit run app/app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`). Enter vehicle parameters and click **Predict** to get an MPG estimate.

### Generate the EDA notebook

```bash
python generate_notebook.py
```

Then open `notebooks/eda.ipynb` in Jupyter.

---

## Dataset

**Source:** UCI Auto-MPG dataset  
**Records:** 398 (6 rows with missing horsepower are dropped)  
**Target:** `mpg` (Miles Per Gallon)

| Feature | Description |
|---------|-------------|
| cylinders | Number of engine cylinders |
| displacement | Engine displacement (cubic inches) |
| horsepower | Engine horsepower |
| weight | Vehicle weight (lbs) |
| acceleration | Time to accelerate 0-60 mph (seconds) |
| model year | Model year (70–82) |
| origin | 1 = USA, 2 = Europe, 3 = Japan |

---

## Methodology

1. **Data Preprocessing** — Handle missing values (`?` in horsepower), drop `car name`, one-hot encode `origin`.
2. **Outlier Removal** — Robust detection using **Median ± 3 × MAD** on all continuous predictors.
3. **Feature Selection** — Correlation-based (|r| > 0.5) and Forward Stepwise Selection.
4. **Model Training** — Linear, Forward Stepwise, Lasso (CV), Ridge (CV), PCR.
5. **Evaluation** — RMSE, R², MAE on a held-out 20% test set.
6. **Deployment** — Best model served through a Streamlit UI.

---

## Results

After running `python main.py`, a comparison table is printed and saved to `models/results.csv`. The best model is automatically selected based on the highest R² score.

---

## Technologies

- Python 3.10+
- pandas, NumPy, scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Jupyter

---

## License

This project is submitted as academic coursework for IIT Ropar and is intended for educational purposes.
