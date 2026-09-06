"""
Load the trained model and score new transactions.

Usage:
    python src/predict.py
"""
import os

import joblib
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH = os.path.join(ROOT, "models", "logistic_regression_fraud_model.joblib")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.joblib")
THRESHOLD = 0.172  # tuned threshold from train.py; see outputs/metrics.json


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    df must contain the original columns: Time, V1..V28, Amount
    (Class is not required/used).
    """
    df = df.copy()
    df[["Time_scaled", "Amount_scaled"]] = scaler.transform(df[["Time", "Amount"]])
    X = df.drop(columns=["Time", "Amount"], errors="ignore")
    if "Class" in X.columns:
        X = X.drop(columns=["Class"])

    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["is_fraud_predicted"] = (df["fraud_probability"] >= THRESHOLD).astype(int)
    return df


if __name__ == "__main__":
    model, scaler = load_artifacts()

    # Demo: score a few rows from the dataset (replace with real new transactions)
    sample = pd.read_csv(os.path.join(ROOT, "data", "creditcard.csv")).sample(
        5, random_state=1
    )
    scored = predict(sample, model, scaler)
    print(scored[["Time", "Amount", "Class", "fraud_probability", "is_fraud_predicted"]])
