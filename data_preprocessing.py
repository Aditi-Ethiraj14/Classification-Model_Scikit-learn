"""
Adult Income Dataset Preprocessing
Loads dataset and preprocesses it for model training.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load Dataset
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
df = pd.read_csv(url, header=None, names=column_names, na_values=" ?", skipinitialspace=True)

print("✅ Dataset Loaded.")
print(df.head())

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.drop("income")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
df["income"] = le_target.fit_transform(df["income"])

# Scale features
X = df.drop("income", axis=1)
y = df["income"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("✅ Preprocessing complete.")
