"""
src/utils.py
============
Shared constants, feature definitions, and helper functions
used across all model scripts.
"""

import os
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "DATAT.xlsx")

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# Standard 11 inputs (used by GBM, RF, SVR, ANN, DNN, XGB, MARS)
INPUT_11 = ['C', 'H', 'O', 'N', 'FC', 'VM', 'Ash', 'HHV', 'Tem', 'RT', 'HR']

# Extended 16 inputs (used by MLP-NN, DT, KNN models)
INPUT_16 = ['C', 'H', 'O', 'N', 'S', 'FC', 'VM', 'Ash', 'CL', 'HC', 'LG', 'PS', 'SM', 'HHV', 'Tem', 'RT', 'HR']

# Standard 9 outputs
OUTPUT_9 = ['CP', 'HP', 'OP', 'NP', 'FCP', 'VMP', 'AshP', 'Yield', 'HHVP']

# Extended 13 outputs
OUTPUT_13 = ['BY', 'HHVP', 'EY', 'CP', 'HP', 'OP', 'NP', 'FCP', 'VMP', 'AshP', 'CLP', 'HCP', 'LGP']

# Human-readable labels for 9 outputs
OUTPUT_LABELS_9 = {
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

# Human-readable labels for 13 outputs
OUTPUT_LABELS_13 = {
    'Output_1':  'BY (wt%)',
    'Output_2':  'HHV_P (MJ/kg)',
    'Output_3':  'EY (%)',
    'Output_4':  'CP (wt%)',
    'Output_5':  'HP (wt%)',
    'Output_6':  'OP (wt%)',
    'Output_7':  'NP (wt%)',
    'Output_8':  'FCP (wt%)',
    'Output_9':  'VMP (wt%)',
    'Output_10': 'AshP (wt%)',
    'Output_11': 'CLP (wt%)',
    'Output_12': 'HCP (wt%)',
    'Output_13': 'LGP (wt%)',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_metrics(metrics: dict, labels: dict = None):
    """Print per-output metrics and averages."""
    if labels is None:
        labels = {}
    for col, m in metrics.items():
        name = labels.get(col, col)
        print(f"\n  [{name}]")
        for k, v in m.items():
            print(f"    {k}: {v:.4f}")

    avg = {k: np.mean([metrics[c][k] for c in metrics]) for k in next(iter(metrics.values()))}
    print("\n  --- Average across all outputs ---")
    for k, v in avg.items():
        print(f"    {k}: {v:.4f}")


def compute_metrics(y_true, y_pred) -> dict:
    """Return MSE, RMSE, MAE, MAPE(%), R² for a single output."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R²': r2}
