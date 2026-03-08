# BiocharPredictionApp

A machine-learning GUI for predicting biochar properties from lignocellulosic biomass **dry torrefaction**.  
Six ML models (GBM, RF, SVR, ANN, DNN, MARS) were benchmarked; **Gradient Boosting Machine (GBM)** proved most accurate and is used in the application.

---

## Overview

| Item | Detail |
|------|--------|
| **Task** | Multi-output regression |
| **Inputs** | 11 biomass/process features |
| **Outputs** | 9 biochar properties |
| **Best model** | GBM (`n_estimators=100`, `lr=0.5`, `max_depth=5`) |
| **GUI** | Python Tkinter |

### Input Features (11)

| Feature | Unit | Description |
|---------|------|-------------|
| C | wt% | Carbon content |
| H | wt% | Hydrogen content |
| O | wt% | Oxygen content |
| N | wt% | Nitrogen content |
| FC | wt% | Fixed carbon |
| VM | wt% | Volatile matter |
| Ash | wt% | Ash content |
| HHV | MJ/kg | Higher heating value |
| Tem | ℃ | Torrefaction temperature |
| RT | min | Residence time |
| HR | ℃/min | Heating rate |

### Output Predictions (9)

C%, H%, O%, N%, FC%, VM%, Ash%, **Yield** (wt%), **HHV** (MJ/kg) of the resulting biochar.

---

## Repository Structure

```
BiocharPredictionApp/
├── main.py             # Model training + Tkinter GUI
├── requirements.txt    # Python dependencies
├── data/
│   └── original_data.xlsx   # Experimental dataset
├── LICENSE
└── README.md
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/guangcansu/BiocharPredictionApp.git
cd BiocharPredictionApp

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Python 3.8+** required. Tkinter is included with most Python distributions.  
> On Linux you may need: `sudo apt-get install python3-tk`

---

## Usage

```bash
python main.py
```

1. The model trains automatically on startup (a few seconds).
2. Enter your biomass/process parameters in the GUI fields.
3. Click **Predict** — the 9 biochar properties appear instantly.

---

## Model Comparison

Six models were evaluated on the same dataset:

| Model | Description |
|-------|-------------|
| **GBM** ✅ | Gradient Boosting Machine — best overall R² |
| RF | Random Forest |
| SVR | Support Vector Regression |
| ANN | Artificial Neural Network (shallow) |
| DNN | Deep Neural Network |
| MARS | Multivariate Adaptive Regression Splines |

GBM was selected for the final application based on lowest RMSE and highest R² across all 9 outputs.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
