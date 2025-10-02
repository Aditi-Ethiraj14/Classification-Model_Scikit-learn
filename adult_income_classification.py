"""
Adult Income Classification Project
Predicts whether an individual's income exceeds $50K/year.
Uses Scikit-learn for preprocessing and XGBoost for classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

#Load Dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)

print("Dataset Loaded.")
print(df.head())


#Preprocess Dataset
df.dropna(inplace=True)

categorical_cols = df.select_dtypes(include=["object"]).columns.drop("income")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

le_target = LabelEncoder()
df["income"] = le_target.fit_transform(df["income"])

X = df.drop("income", axis=1)
y = df["income"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_test_original = pd.DataFrame(X_test, columns=X.columns)

# Train XGBoost Model

xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)

param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=10, scoring="accuracy", cv=3, verbose=1, random_state=42)
random_search.fit(X_train, y_train)

best_xgb_model = random_search.best_estimator_
print("\nBest XGBoost Parameters:", random_search.best_params_)

#Model Evaluation

y_pred = best_xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_target.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Tuned XGBoost")
plt.show()

#Feature Importance

xgb.plot_importance(best_xgb_model, max_num_features=10, importance_type='gain', height=0.5)
plt.title("Top 10 Feature Importances — Tuned XGBoost")
plt.show()

#Example Predictions

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

#Preprocess test cases
for col in categorical_cols:
    test_cases[col] = le_dict[col].transform(test_cases[col])
test_cases_scaled = scaler.transform(test_cases)

predictions = best_xgb_model.predict(test_cases_scaled)
test_cases["Prediction"] = le_target.inverse_transform(predictions)

print("\n🔍 Example Predictions:")
print(test_cases)
