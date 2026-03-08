"""
BiocharPredictionApp - main.py
Predicts biochar properties from lignocellulosic biomass dry torrefaction
using a Gradient Boosting Machine (GBM) model with a Tkinter GUI.

Inputs  (11): C, H, O, N, FC, VM, Ash, HHV, Temperature, Residence Time, Heating Rate
Outputs  (9): C%, H%, O%, N%, FC%, VM%, Ash%, Yield, HHV
"""

import os
import tkinter as tk

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data loading — path is relative to this script's location
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "original_data.xlsx")

data = pd.read_excel(DATA_PATH)

# ---------------------------------------------------------------------------
# Features & targets
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

X = data[INPUT_FEATURES].values
y = data[OUTPUT_FEATURES].values

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
imputer_X = SimpleImputer(strategy='mean')
X = imputer_X.fit_transform(X)

imputer_y = SimpleImputer(strategy='mean')
y = imputer_y.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=150
)

# ---------------------------------------------------------------------------
# Model training (one GBM per output)
# ---------------------------------------------------------------------------
models = {}
metrics = {}

for i in range(y.shape[1]):
    model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.5, max_depth=5, random_state=85
    )
    model.fit(X_train, y_train[:, i])
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test[:, i], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test[:, i], y_pred)
    mape = np.mean(np.abs((y_test[:, i] - y_pred) / (y_test[:, i] + 1e-8))) * 100
    r2 = r2_score(y_test[:, i], y_pred)

    col = f'Output_{i + 1}'
    models[col] = model
    metrics[col] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2}

# ---------------------------------------------------------------------------
# Print metrics summary
# ---------------------------------------------------------------------------
print("\n=== Model Performance Metrics ===")
for col, m in metrics.items():
    label = OUTPUT_LABELS.get(col, col)
    print(f"\n[{label}]")
    for k, v in m.items():
        unit = "%" if k == "MAPE" else ""
        print(f"  {k}: {v:.4f}{unit}")

avg = {k: np.mean([metrics[c][k] for c in metrics]) for k in metrics['Output_1']}
print("\n=== Average Metrics (all outputs) ===")
for k, v in avg.items():
    print(f"  {k}: {v:.4f}")


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------
class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biochar Property Prediction")
        self.root.resizable(False, False)

        self.features = [
            'C (wt%)', 'H (wt%)', 'O (wt%)', 'N (wt%)',
            'FC (wt%)', 'VM (wt%)', 'Ash (wt%)', 'HHV (MJ/kg)',
            'Tem (℃)', 'RT (min)', 'HR (℃/min)',
        ]
        self.input_labels = {f: tk.Label(root, text=f) for f in self.features}
        self.input_entries = {f: tk.Entry(root, width=10) for f in self.features}

        # Layout: 4 inputs per row
        for i, (label, entry) in enumerate(
            zip(self.input_labels.values(), self.input_entries.values())
        ):
            label.grid(row=i // 4, column=i % 4 * 2, padx=(5, 20), pady=5, sticky='e')
            entry.grid(row=i // 4, column=i % 4 * 2 + 1, padx=5, pady=5)

        btn_row = (len(self.features) - 1) // 4 + 1
        tk.Button(root, text="Predict", command=self.predict_outputs).grid(
            row=btn_row, column=0, columnspan=8, pady=10
        )

        # Output labels
        self.out_labels = [tk.Label(root, text="") for _ in range(4)]
        for i, lbl in enumerate(self.out_labels):
            lbl.grid(row=btn_row + 1 + i, column=0, columnspan=8, pady=3)
        self.out_labels[0].config(text="Predicted Outputs:")

    def preprocess_input(self):
        vals = np.array(
            [float(e.get()) for e in self.input_entries.values()]
        ).reshape(1, -1)
        vals = imputer_X.transform(vals)
        vals = scaler.transform(vals)
        return vals

    def predict_outputs(self):
        try:
            inp = self.preprocess_input()
        except ValueError:
            self.out_labels[1].config(text="⚠ Please fill in all fields with numeric values.")
            self.out_labels[2].config(text="")
            self.out_labels[3].config(text="")
            return

        preds = {col: model.predict(inp)[0] for col, model in models.items()}

        items = [
            f"{OUTPUT_LABELS[c].split('(')[0].strip()}: {v:.4f}"
            for c, v in preds.items()
        ]
        # Split across three lines (4 / 3 / 2)
        self.out_labels[0].config(text="Predicted Outputs:")
        self.out_labels[1].config(text="  |  ".join(items[:4]))
        self.out_labels[2].config(text="  |  ".join(items[4:7]))
        self.out_labels[3].config(text="  |  ".join(items[7:]))


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
