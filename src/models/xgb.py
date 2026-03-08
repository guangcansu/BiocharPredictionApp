"""
src/models/xgb.py
=================
XGBoost — multi-output (11 inputs → 9 outputs).
Requires: pip install xgboost
Run: python src/models/xgb.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import DATA_PATH, INPUT_11, OUTPUT_9, OUTPUT_LABELS_9, compute_metrics, print_metrics


def train():
    data = pd.read_excel(DATA_PATH)
    X = data[INPUT_11].values
    y = data[OUTPUT_9].values

    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)
    imputer_y = SimpleImputer(strategy='mean')
    y = imputer_y.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)

    models, metrics = {}, {}
    for i in range(y.shape[1]):
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=85)
        model.fit(X_train, y_train[:, i])
        y_pred = model.predict(X_test)
        col = f'Output_{i+1}'
        models[col] = model
        metrics[col] = compute_metrics(y_test[:, i], y_pred)

    return models, metrics, imputer_X, scaler


if __name__ == '__main__':
    print("\n=== XGBoost Model (Multi-Output) ===")
    models, metrics, _, _ = train()
    print_metrics(metrics, OUTPUT_LABELS_9)
