import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load data from Excel file
data = pd.read_excel(r"E:\DATA.xlsx")

# Extract features and targets
X = data[['C', 'H', 'O', 'N', 'FC', 'VM', 'Ash', 'HHV', 'Tem', 'RT', 'HR']].values
y = data[['CP', 'HP', 'OP', 'NP', 'FCP', 'VMP', 'AshP', 'Yield', 'HHVP']].values

# Handle missing values with SimpleImputer using mean strategy
imputer_X = SimpleImputer(strategy='mean')
X = imputer_X.fit_transform(X)

imputer_y = SimpleImputer(strategy='mean')
y = imputer_y.fit_transform(y)  # Filling NaN in the target variables, if any

# Data standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=150)

# Create a dictionary to store models and metrics
models = {}
metrics = {}

# Train individual models for each output
for i in range(y.shape[1]):
    # Create the Gradient Boosting Regressor model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5, max_depth=5, random_state=85)

    # Train the model
    model.fit(X_train, y_train[:, i])

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test[:, i], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test[:, i], y_pred)
    mape = np.mean(np.abs((y_test[:, i] - y_pred) / y_test[:, i])) * 100
    r2 = r2_score(y_test[:, i], y_pred)

    # Store the model and metrics in the dictionary
    column_name = f'Output_{i+1}'  # You can customize the column name as needed
    models[column_name] = model
    metrics[column_name] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R-squared': r2
    }

# Print metrics for each output
for column, metric in metrics.items():
    print(f'Metrics for {column}:')
    print(f'Mean Squared Error (MSE): {metric["MSE"]:.4f}')
    print(f'Root Mean Squared Error (RMSE): {metric["RMSE"]:.4f}')
    print(f'Mean Absolute Error (MAE): {metric["MAE"]:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {metric["MAPE"]:.4f}%')
    print(f'R-squared (R²): {metric["R-squared"]:.4f}')
    print('\n')


# Calculate average metrics
average_metrics = {metric: np.mean([metrics[column][metric] for column in metrics]) for metric in metrics['Output_1']}

# Calculate weighted metrics (adjust weights as needed)
weights = {'Output_1': 1, 'Output_2': 1, 'Output_3': 1, 'Output_4': 1, 'Output_5': 1, 'Output_6': 1, 'Output_7': 1, 'Output_8': 1, 'Output_9': 1}
weighted_metrics = {metric: sum(weights[column] * metrics[column][metric] for column in metrics) / sum(weights.values()) for metric in metrics['Output_1']}

# Print average metrics
print("Average Metrics:")
for metric, value in average_metrics.items():
    print(f'{metric}: {value:.4f}')

# Print weighted metrics
print("\nWeighted Metrics:")
for metric, value in weighted_metrics.items():
    print(f'{metric}: {value:.4f}')

import tkinter as tk
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biochar prediction")

        # Create input fields
        self.features = ['C (wt%)', 'H (wt%)', 'O (wt%)', 'N (wt%)', 'FC (wt%)', 'VM (wt%)',  'Ash (wt%)', 'HHV (MJ/kg)', 'Tem (℃)', 'RT (min)', 'HR (℃/min)']
        self.input_labels = {feature: tk.Label(self.root, text=feature) for feature in self.features}
        self.input_entries = {feature: tk.Entry(self.root, width=10) for feature in self.features}

        # Pack labels and input fields in rows of four with increased horizontal distance
        for i, (label, entry) in enumerate(zip(self.input_labels.values(), self.input_entries.values())):
            label.grid(row=i // 4, column=i % 4 * 2, padx=(5, 20), pady=5, sticky='e')
            entry.grid(row=i // 4, column=i % 4 * 2 + 1, padx=5, pady=5)

        # Create submit button below input fields
        submit_button = tk.Button(self.root, text="Predict", command=self.predict_outputs)
        submit_button.grid(row=(len(self.features) - 1) // 4 + 1, column=0, columnspan=9, pady=10)

        # Create result labels in three rows
        self.result_label_1 = tk.Label(self.root, text="Predicted Outputs: ")
        self.result_label_1.grid(row=(len(self.features) - 1) // 4 + 2, column=0, columnspan=9, pady=5)

        self.result_label_2 = tk.Label(self.root, text="")
        self.result_label_2.grid(row=(len(self.features) - 1) // 4 + 3, column=0, columnspan=9, pady=5)

        self.result_label_3 = tk.Label(self.root, text="")
        self.result_label_3.grid(row=(len(self.features) - 1) // 4 + 4, column=0, columnspan=9, pady=5)

        self.result_label_4 = tk.Label(self.root, text="")
        self.result_label_4.grid(row=(len(self.features) - 1) // 4 + 5, column=0, columnspan=9, pady=5)

    def preprocess_input(self):
        # Convert user input to a NumPy array and scale it
        input_array = np.array([float(entry.get()) for entry in self.input_entries.values()]).reshape(1, -1)
        input_array = imputer_X.transform(input_array)
        input_array = scaler.transform(input_array)
        return input_array

    def predict_outputs(self):
        # Preprocess user input
        input_array = self.preprocess_input()

        # Make predictions
        predictions = {}
        for i, (column, model) in enumerate(models.items()):
            predictions[column] = model.predict(input_array)[0]

        # Display predictions in three rows with decreasing number of predictions
        mapped_output_labels = {'Output_1': 'C', 'Output_2': 'H', 'Output_3': 'O', 'Output_4': 'N', 'Output_5': 'FC', 'Output_6': 'VM', 'Output_7': 'Ash', 'Output_8': 'Yield', 'Output_9': 'HHV'}
        result_text_1 = ", ".join([f'{mapped_output_labels[column]}: {value:.4f}' for column, value in predictions.items() if column in mapped_output_labels][:4])
        result_text_2 = ", ".join([f'{mapped_output_labels[column]}: {value:.4f}' for column, value in predictions.items() if column in mapped_output_labels][4:7])
        result_text_3 = ", ".join([f'{mapped_output_labels[column]}: {value:.4f}' for column, value in predictions.items() if column in mapped_output_labels][7:])

        # Update result labels
        self.result_label_1.config(text="Predicted Outputs: " )
        self.result_label_2.config(text=result_text_1)
        self.result_label_3.config(text=result_text_2)
        self.result_label_4.config(text=result_text_3)  # Set it to an empty string initially

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()

