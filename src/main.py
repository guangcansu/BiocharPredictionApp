"""
src/main.py
===========
Trains a Gradient Boosting Machine (GBM) model on the biochar dataset
and exposes the fitted objects for use by the GUI.

Can also be run standalone to print performance metrics:
    python src/main.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "DATAT.xlsx")

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------
INPUT_FEATURES = ['C', 'H', 'O', 'N', 'FC', 'VM', 'Ash', 'HHV', 'Tem', 'RT', 'HR']
OUTPUT_FEATURES = ['CP', 'HP', 'OP', 'NP', 'FCP', 'VMP', 'AshP', 'Yield', 'HHVP']
OUTPUT_LABELS = {
    'Output_1': 'C (wt%)',
    'Output_2': 'H (wt%)',
    'Output_3': 'O (wt%)',
    'Output_4': 'N (wt%)',
    'Output_5': 'FC (wt%)',
    'Output_6': 'VM (wt%)',
    'Output_7': 'Ash (wt%)',
    'Output_8': 'Yield (wt%)',
    'Output_9': 'HHV (MJ/kg)',
}


def train():
    """
    Load data, preprocess, train GBM models, and return fitted objects.

    Returns
    -------
    models : dict[str, GradientBoostingRegressor]
    metrics : dict[str, dict]
    imputer_X : SimpleImputer
    scaler : StandardScaler
    """
    data = pd.read_excel(DATA_PATH)

    X = data[INPUT_FEATURES].values
    y = data[OUTPUT_FEATURES].values

    # Imputation
    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)

    imputer_y = SimpleImputer(strategy='mean')
    y = imputer_y.fit_transform(y)

    # Standardisation
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=150
    )

    models = {}
    metrics = {}

    for i in range(y.shape[1]):
        model = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.5, max_depth=5, random_state=85
        )
        model.fit(X_train, y_train[:, i])
        y_pred = model.predict(X_test)

        mse  = mean_squared_error(y_test[:, i], y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test[:, i], y_pred)
        mape = np.mean(np.abs((y_test[:, i] - y_pred) / (y_test[:, i] + 1e-8))) * 100
        r2   = r2_score(y_test[:, i], y_pred)

        col = f'Output_{i + 1}'
        models[col]  = model
        metrics[col] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R²': r2}

    return models, metrics, imputer_X, scaler


def print_metrics(metrics: dict):
    print("\n=== GBM Model Performance ===")
    for col, m in metrics.items():
        print(f"\n  [{OUTPUT_LABELS.get(col, col)}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    avg = {k: np.mean([metrics[c][k] for c in metrics]) for k in metrics['Output_1']}
    print("\n  --- Average across all outputs ---")
    for k, v in avg.items():
        print(f"    {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    models, metrics, imputer_X, scaler = train()
    print_metrics(metrics)
