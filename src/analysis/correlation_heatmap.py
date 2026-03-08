"""
src/analysis/correlation_heatmap.py
=====================================
Pearson correlation heatmap for all variables.
Run: python src/analysis/correlation_heatmap.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils import DATA_PATH

data = pd.read_excel(DATA_PATH)

variables = ['C','H','O','N','FC','VM','Ash','CL','HC','LG','PS','SM','HHV',
             'Tem','RT','HR','BY','HHVP','EY','CP','HP','OP','NP','FCP','VMP',
             'AshP','CLP','HCP','LGP']

data[variables] = data[variables].replace({',': '.'}, regex=True).astype(float)
corr = data[variables].corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr, annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Pearson Correlation Coefficient Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         'results', 'correlation_heatmap.png'), dpi=150)
plt.show()
