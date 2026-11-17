# Import required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

print("Libraries imported successfully!")
# Load the customer churn dataset

data_path = "../data/Telco-Customer-Churn.csv"

df = pd.read_csv(data_path)

# Display dataset information

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())
# Convert TotalCharges column into numeric values

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values

print("\nMissing Values:")
print(df.isnull().sum())

# Drop customerID column

df = df.drop("customerID", axis=1)

print("\nDataset Shape After Dropping CustomerID:", df.shape)
# Separate input features and target variable

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Convert target values into numerical format

y = y.map({"No": 0, "Yes": 1})

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())
# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)

print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)
# Identify numerical and categorical columns

numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

print("\nNumber of Numerical Features:", len(numerical_features))
print("Number of Categorical Features:", len(categorical_features))
# Numerical preprocessing pipeline

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing pipeline

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

print("\nPreprocessing Transformers Created Successfully!")
# Combine numerical and categorical transformers

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

print("\nColumn Transformer Created Successfully!")
# Create Logistic Regression pipeline

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

print("\nLogistic Regression Pipeline Created Successfully!")
# Train Logistic Regression model

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")