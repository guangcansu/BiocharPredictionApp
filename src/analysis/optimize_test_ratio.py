"""
src/analysis/optimize_test_ratio.py
=====================================
Sweeps test-set ratios (5%–30%) and plots average R² across all 9 GBM outputs.
Run: python src/analysis/optimize_test_ratio.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import DATA_PATH, INPUT_11, OUTPUT_9

data = pd.read_excel(DATA_PATH)
X = data[INPUT_11].values
y = data[OUTPUT_9].values

imputer_X = SimpleImputer(strategy='mean')
X = imputer_X.fit_transform(X)
imputer_y = SimpleImputer(strategy='mean')
y = imputer_y.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

ratios, avg_r2s = np.arange(0.05, 0.31, 0.01), []

for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=150)
    r2s = []
    for i in range(y.shape[1]):
        m = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=5, random_state=85)
        m.fit(X_train, y_train[:, i])
        r2s.append(r2_score(y_test[:, i], m.predict(X_test)))
    avg_r2s.append(np.mean(r2s))

best_ratio = ratios[np.argmax(avg_r2s)]
print(f"Best test ratio: {best_ratio:.2f}  (avg R²={max(avg_r2s):.4f})")

plt.figure(figsize=(8, 5))
plt.plot(ratios, avg_r2s, marker='o', linewidth=2)
plt.axvline(best_ratio, color='red', linestyle='--', label=f'Best ratio={best_ratio:.2f}')
plt.title('GBM — Average R² vs Test Set Ratio')
plt.xlabel('Test Set Ratio')
plt.ylabel('Average R²')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'results', 'optimize_test_ratio.png'), dpi=150)
plt.show()

print("\nRatio\t\tAvg R²")
for r, v in zip(ratios, avg_r2s):
    print(f"{r:.2f}\t\t{v:.4f}")
