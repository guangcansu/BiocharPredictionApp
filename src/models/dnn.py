"""
src/models/dnn.py
=================
Deep Neural Network (TensorFlow/Keras) — multi-output (11 inputs → 9 outputs).
Requires: pip install tensorflow
Run: python src/models/dnn.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

    models, metrics = {}, {}
    for i in range(y.shape[1]):
        model = Sequential([
            Dense(100, input_dim=X.shape[1], activation='relu'),
            Dense(50, activation='relu'),
            Dense(1),
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train[:, i], epochs=100, batch_size=32, verbose=0)

        y_pred = model.predict(X_test).flatten()
        col = f'Output_{i+1}'
        models[col] = model
        metrics[col] = compute_metrics(y_test[:, i], y_pred)

    return models, metrics, imputer_X, scaler


if __name__ == '__main__':
    print("\n=== DNN Model (Multi-Output) ===")
    models, metrics, _, _ = train()
    print_metrics(metrics, OUTPUT_LABELS_9)
