"""
src/analysis/tune_ann.py
=========================
ANN (MLPRegressor) GridSearchCV — finds best hidden_layer_sizes and random_state.
Run: python src/analysis/tune_ann.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from src.utils import DATA_PATH, INPUT_16, OUTPUT_13

data = pd.read_excel(DATA_PATH)
X = data[INPUT_16].values
y = data[OUTPUT_13[3]].values  # CP as representative output

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.reshape(-1, 1)).ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=85)

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    'random_state':       [42, 85, 123],
}
grid = GridSearchCV(MLPRegressor(max_iter=2000), param_grid, cv=5, scoring='r2')
grid.fit(X_train, y_train)

best = grid.best_params_
print(f"Best params: {best}")

model = MLPRegressor(max_iter=2000, **best)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Optimized R²: {r2_score(y_test, y_pred):.4f}")
