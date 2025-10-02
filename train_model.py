"""
XGBoost Training Script
Trains XGBoost model on preprocessed data and saves the model.
"""

import xgboost as xgb
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from data_preprocessing import X_train, X_test, y_train, y_test, le_target

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)

# Hyperparameter grid
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 150],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# Randomized Search
random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=10, scoring="accuracy", cv=3, verbose=1, random_state=42)
random_search.fit(X_train, y_train)

best_xgb_model = random_search.best_estimator_
joblib.dump(best_xgb_model, "xgb_model.pkl")

print("\n✅ Best XGBoost Parameters:", random_search.best_params_)

# Model evaluation
y_pred = best_xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Performance:")
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

# Feature importance
xgb.plot_importance(best_xgb_model, max_num_features=10, importance_type='gain', height=0.5)
plt.title("Top 10 Feature Importances — Tuned XGBoost")
plt.show()
