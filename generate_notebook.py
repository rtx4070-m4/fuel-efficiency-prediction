"""
generate_notebook.py - Programmatically create the EDA Jupyter notebook.
Run this script to produce notebooks/eda.ipynb.
"""

import json
import os

cells = []

def add_md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": [],
        "execution_count": None
    })

# ── Title ─────────────────────────────────────────────────────────────────────
add_md("# 🚗 Exploratory Data Analysis – Auto MPG Dataset\n\n"
       "**Authors:** Rishiraj Karn (2025DSS1020), Ritam Rabha (2025DSS1021)  \n"
       "**Submitted to:** Prof. Jayram Vallaru, IIT Ropar")

# ── Imports ───────────────────────────────────────────────────────────────────
add_code(
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n"
    "%matplotlib inline"
)

# ── Load data ─────────────────────────────────────────────────────────────────
add_md("## 1. Load the Dataset")
add_code(
    "df = pd.read_csv('../data/auto-mpg.csv')\n"
    "print(f'Shape: {df.shape}')\n"
    "df.head(10)"
)

# ── Basic info ────────────────────────────────────────────────────────────────
add_md("## 2. Basic Information")
add_code("df.info()")
add_code("df.describe()")

# ── Missing values ────────────────────────────────────────────────────────────
add_md("## 3. Missing Values & Data Cleaning\n\n"
       "The `horsepower` column contains `?` as placeholders for missing values.")
add_code(
    "# Count '?' in horsepower\n"
    "print('Missing horsepower entries:', (df['horsepower'] == '?').sum())\n\n"
    "# Convert and drop NAs\n"
    "df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')\n"
    "df.dropna(inplace=True)\n"
    "df.reset_index(drop=True, inplace=True)\n"
    "print(f'Shape after cleaning: {df.shape}')"
)

# ── Target distribution ──────────────────────────────────────────────────────
add_md("## 4. Distribution of Target Variable (MPG)")
add_code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
    "axes[0].hist(df['mpg'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')\n"
    "axes[0].set_xlabel('MPG')\n"
    "axes[0].set_ylabel('Frequency')\n"
    "axes[0].set_title('Distribution of MPG')\n\n"
    "axes[1].boxplot(df['mpg'], vert=True)\n"
    "axes[1].set_ylabel('MPG')\n"
    "axes[1].set_title('Box Plot of MPG')\n\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print(f'Mean MPG: {df[\"mpg\"].mean():.2f}')\n"
    "print(f'Median MPG: {df[\"mpg\"].median():.2f}')\n"
    "print(f'Std MPG: {df[\"mpg\"].std():.2f}')"
)

# ── Feature distributions ────────────────────────────────────────────────────
add_md("## 5. Distribution of Predictor Variables")
add_code(
    "numeric_cols = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year']\n\n"
    "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
    "for ax, col in zip(axes.flatten(), numeric_cols):\n"
    "    ax.hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='teal')\n"
    "    ax.set_title(col.title())\n"
    "    ax.set_xlabel(col)\n"
    "    ax.set_ylabel('Frequency')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ── Correlation heatmap ──────────────────────────────────────────────────────
add_md("## 6. Correlation Heatmap")
add_code(
    "corr = df[['mpg'] + numeric_cols].corr()\n\n"
    "plt.figure(figsize=(10, 7))\n"
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,\n"
    "            linewidths=0.5, square=True)\n"
    "plt.title('Pearson Correlation Matrix')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ── Scatter plots ─────────────────────────────────────────────────────────────
add_md("## 7. Scatter Plots: Features vs MPG")
add_code(
    "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n"
    "for ax, col in zip(axes.flatten(), numeric_cols):\n"
    "    ax.scatter(df[col], df['mpg'], alpha=0.5, s=20, color='darkorange')\n"
    "    ax.set_xlabel(col.title())\n"
    "    ax.set_ylabel('MPG')\n"
    "    ax.set_title(f'{col.title()} vs MPG')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ── Origin analysis ──────────────────────────────────────────────────────────
add_md("## 8. MPG by Origin\n\n"
       "Origin: 1 = USA, 2 = Europe, 3 = Japan")
add_code(
    "origin_labels = {1: 'USA', 2: 'Europe', 3: 'Japan'}\n"
    "df['origin_label'] = df['origin'].map(origin_labels)\n\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.boxplot(x='origin_label', y='mpg', data=df, palette='Set2')\n"
    "plt.title('MPG Distribution by Origin')\n"
    "plt.xlabel('Origin')\n"
    "plt.ylabel('MPG')\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print(df.groupby('origin_label')['mpg'].describe().round(2))"
)

# ── Model year trend ─────────────────────────────────────────────────────────
add_md("## 9. MPG Trend Over Model Years")
add_code(
    "yearly = df.groupby('model year')['mpg'].mean()\n\n"
    "plt.figure(figsize=(10, 5))\n"
    "plt.plot(yearly.index, yearly.values, marker='o', linewidth=2, color='seagreen')\n"
    "plt.xlabel('Model Year')\n"
    "plt.ylabel('Average MPG')\n"
    "plt.title('Average MPG by Model Year')\n"
    "plt.grid(True, alpha=0.3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ── Pair plot ─────────────────────────────────────────────────────────────────
add_md("## 10. Pair Plot (Key Features)")
add_code(
    "key_features = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']\n"
    "sns.pairplot(df[key_features], diag_kind='kde', plot_kws={'alpha': 0.4, 's': 15})\n"
    "plt.suptitle('Pair Plot of Key Features', y=1.02)\n"
    "plt.show()"
)

# ── Insights ──────────────────────────────────────────────────────────────────
add_md(
    "## 11. Key Insights\n\n"
    "1. **Weight is the strongest negative predictor** of MPG — heavier cars are less fuel-efficient.\n"
    "2. **Displacement and horsepower** also show strong negative correlations with MPG.\n"
    "3. **Model year has a positive relationship** with MPG — newer cars (in this dataset era) tend to be more efficient.\n"
    "4. **Japanese cars** (origin 3) achieve the highest average MPG, followed by European cars.\n"
    "5. **Cylinders** is strongly correlated with displacement and weight (multicollinearity), "
    "which motivates techniques like PCA and regularisation.\n"
    "6. **Acceleration** has a weak positive correlation with MPG and may not be a strong standalone predictor.\n"
    "7. The target variable (MPG) is roughly **right-skewed**, with a mean around 23 MPG."
)

# ── Write notebook ────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells,
}

outpath = os.path.join(os.path.dirname(__file__), "notebooks", "eda.ipynb")
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook saved → {outpath}")
