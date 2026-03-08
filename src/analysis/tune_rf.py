"""
src/analysis/tune_rf.py
========================
RF parameter tuning — 3D surface plot of RMSE vs n_estimators vs random_state.
Run: python src/analysis/tune_rf.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import DATA_PATH, INPUT_11, OUTPUT_9

data = pd.read_excel(DATA_PATH)
X = data[INPUT_11].values
y = data[OUTPUT_9[7]].values  # Yield as representative output

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y.reshape(-1, 1)).ravel()

scaler = StandardScaler()
X = scaler.fit_transform(X)

random_states    = [42, 75, 100, 125, 150]
n_estimators_vals = [50, 100, 150, 200, 250]
results = np.zeros((len(random_states), len(n_estimators_vals)))

for i, rs in enumerate(random_states):
    for j, ne in enumerate(n_estimators_vals):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)
        model = RandomForestRegressor(n_estimators=ne, random_state=rs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[i, j] = np.sqrt(mean_squared_error(y_test, y_pred))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X_grid, Y_grid = np.meshgrid(n_estimators_vals, random_states)
surf = ax.plot_surface(X_grid, Y_grid, results, cmap='viridis', edgecolor='k')
ax.set_title('Random Forest RMSE — Parameter Tuning')
ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Random State')
ax.set_zlabel('RMSE')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'results', 'tune_rf_3d.png'), dpi=150)
plt.show()
print(f"\nBest: n_estimators={n_estimators_vals[np.argmin(results) % len(n_estimators_vals)]}, "
      f"random_state={random_states[np.argmin(results) // len(n_estimators_vals)]}, "
      f"RMSE={results.min():.4f}")
