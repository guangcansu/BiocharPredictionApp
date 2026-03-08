"""
src/models/knn.py
=================
K-Nearest Neighbors Regressor — multi-output (16 inputs → 13 outputs).
Run: python src/models/knn.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.utils import DATA_PATH, INPUT_16, OUTPUT_13, OUTPUT_LABELS_13, compute_metrics, print_metrics


def train():
    data = pd.read_excel(DATA_PATH)
    X = data[INPUT_16].values
    y = data[OUTPUT_13].values

    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)
    imputer_y = SimpleImputer(strategy='mean')
    y = imputer_y.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

    models, metrics = {}, {}
    for i in range(y.shape[1]):
        model = KNeighborsRegressor(n_neighbors=5)
        model.fit(X_train, y_train[:, i])
        y_pred = model.predict(X_test)
        col = f'Output_{i+1}'
        models[col] = model
        metrics[col] = compute_metrics(y_test[:, i], y_pred)

    return models, metrics, imputer_X, scaler


if __name__ == '__main__':
    print("\n=== KNN Model (Multi-Output) ===")
    models, metrics, _, _ = train()
    print_metrics(metrics, OUTPUT_LABELS_13)
