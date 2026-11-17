import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "customer_churn_xgb.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)

# Business threshold selected during training
BEST_THRESHOLD = 0.59


def create_features(data):
    data = data.copy()

    # Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )

    # Create tenure group
    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"]
    )

    # Average monthly spend
    data["avg_monthly_spend"] = (
        data["TotalCharges"] /
        data["tenure"].replace(0, np.nan)
    )

    # Count subscribed services
    service_columns = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]

    data["service_count"] = (
        data[service_columns].eq("Yes").sum(axis=1)
    )

    # Long-term contract
    data["long_term_contract"] = (
        data["Contract"]
        .isin(["One year", "Two year"])
        .astype(int)
    )

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Predict customer churn from a CSV file"
    )

    parser.add_argument("input_csv", help="Input customer CSV file")
    args = parser.parse_args()

    input_path = Path(args.input_csv)

    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    # Read new customer data
    customers = pd.read_csv(input_path)

    # Keep customer IDs for the output
    if "customerID" in customers.columns:
        customer_ids = customers["customerID"]
    else:
        customer_ids = pd.Series(
            range(1, len(customers) + 1),
            name="customerID"
        )

    # Remove ID and target if present
    X_new = customers.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    # Apply feature engineering
    X_new = create_features(X_new)

    # Predict probabilities and classes
    probabilities = model.predict_proba(X_new)[:, 1]

    predictions = (
        probabilities >= BEST_THRESHOLD
    ).astype(int)

    # Create output DataFrame
    results = pd.DataFrame({
        "customerID": customer_ids,
        "predicted_churn": [
            "Yes" if value == 1 else "No"
            for value in predictions
        ],
        "churn_probability": probabilities
    })

    # Save predictions
    output_path = BASE_DIR / "predictions.csv"
    results.to_csv(output_path, index=False)

    print("Predictions generated successfully!")
    print(results.head())
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()