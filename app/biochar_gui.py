"""
app/biochar_gui.py
==================
Tkinter GUI for the Biochar Prediction App.

Imports trained GBM models from src/main.py, then launches
an interactive window where researchers can enter biomass/process
parameters and get instant biochar property predictions.

Usage:
    python app/biochar_gui.py
"""

import os
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np

# Allow importing from src/ regardless of working directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.main import OUTPUT_LABELS, train  # noqa: E402

# ---------------------------------------------------------------------------
# Train models on startup
# ---------------------------------------------------------------------------
print("Training models, please wait…")
models, metrics, imputer_X, scaler = train()
print("Done. Launching GUI.\n")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class BiocharApp:
    FEATURES = [
        'C (wt%)', 'H (wt%)', 'O (wt%)', 'N (wt%)',
        'FC (wt%)', 'VM (wt%)', 'Ash (wt%)', 'HHV (MJ/kg)',
        'Tem (℃)', 'RT (min)', 'HR (℃/min)',
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Biochar Property Prediction")
        self.root.resizable(False, False)

        self._build_input_section()
        self._build_button_section()
        self._build_output_section()

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------
    def _build_input_section(self):
        frame = tk.LabelFrame(self.root, text="Biomass / Process Inputs", padx=10, pady=8)
        frame.grid(row=0, column=0, padx=15, pady=10, sticky='ew')

        self.entries: dict[str, tk.Entry] = {}
        for i, feature in enumerate(self.FEATURES):
            row, col = divmod(i, 4)
            tk.Label(frame, text=feature, anchor='e').grid(
                row=row, column=col * 2, padx=(5, 4), pady=4, sticky='e'
            )
            entry = tk.Entry(frame, width=9)
            entry.grid(row=row, column=col * 2 + 1, padx=(0, 12), pady=4)
            self.entries[feature] = entry

    def _build_button_section(self):
        frame = tk.Frame(self.root)
        frame.grid(row=1, column=0, pady=4)
        tk.Button(frame, text="  Predict  ", command=self.predict, width=14).pack()

    def _build_output_section(self):
        frame = tk.LabelFrame(self.root, text="Predicted Biochar Properties", padx=10, pady=8)
        frame.grid(row=2, column=0, padx=15, pady=10, sticky='ew')

        self.result_vars: list[tk.StringVar] = []
        # 9 outputs → 3 rows × 3 outputs
        for row in range(3):
            for col_idx in range(3):
                out_idx = row * 3 + col_idx
                if out_idx >= len(OUTPUT_LABELS):
                    break
                label_text = OUTPUT_LABELS.get(f'Output_{out_idx + 1}', f'Output {out_idx + 1}')
                tk.Label(frame, text=label_text + ":", anchor='e', width=14).grid(
                    row=row, column=col_idx * 2, padx=(8, 4), pady=4, sticky='e'
                )
                var = tk.StringVar(value="—")
                tk.Label(frame, textvariable=var, width=10, anchor='w',
                         relief='sunken', bg='white').grid(
                    row=row, column=col_idx * 2 + 1, padx=(0, 12), pady=4
                )
                self.result_vars.append(var)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _get_inputs(self) -> np.ndarray:
        """Parse entry fields → (1, 11) numpy array, raises ValueError on bad input."""
        vals = []
        for feature, entry in self.entries.items():
            raw = entry.get().strip()
            if raw == "":
                raise ValueError(f"Missing value for '{feature}'")
            vals.append(float(raw))
        return np.array(vals, dtype=float).reshape(1, -1)

    def predict(self):
        try:
            x = self._get_inputs()
        except ValueError as e:
            messagebox.showwarning("Input Error", str(e))
            return

        x = imputer_X.transform(x)
        x = scaler.transform(x)

        for i, (col, model) in enumerate(models.items()):
            pred = model.predict(x)[0]
            self.result_vars[i].set(f"{pred:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    BiocharApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
