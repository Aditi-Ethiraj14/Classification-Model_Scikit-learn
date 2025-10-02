"""
Prediction Script
Loads trained model and predicts sample cases.
"""

import pandas as pd
import joblib
from data_preprocessing import le_dict, scaler, categorical_cols, le_target

# Load model
best_xgb_model = joblib.load("xgb_model.pkl")

# Example test cases
test_cases = pd.DataFrame({
    "age": [25, 40, 50, 35],
    "workclass": ["Private", "Self-emp-not-inc", "Federal-gov", "Private"],
    "fnlwgt": [226802, 123011, 300000, 180000],
    "education": ["11th", "Bachelors", "Doctorate", "Masters"],
    "education-num": [7, 13, 16, 14],
    "marital-status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Divorced"],
    "occupation": ["Machine-op-inspct", "Exec-managerial", "Prof-specialty", "Sales"],
    "relationship": ["Own-child", "Husband", "Husband", "Not-in-family"],
    "race": ["White", "White", "Asian-Pac-Islander", "Black"],
    "sex": ["Male", "Male", "Male", "Female"],
    "capital-gain": [0, 0, 5000, 0],
    "capital-loss": [0, 0, 0, 0],
    "hours-per-week": [40, 50, 60, 20],
    "native-country": ["United-States", "United-States", "United-States", "Mexico"]
})

# Preprocess test cases
for col in categorical_cols:
    test_cases[col] = le_dict[col].transform(test_cases[col])

test_cases_scaled = scaler.transform(test_cases)

predictions = best_xgb_model.predict(test_cases_scaled)
test_cases["Prediction"] = le_target.inverse_transform(predictions)

print("\n🔍 Example Predictions:")
print(test_cases)
