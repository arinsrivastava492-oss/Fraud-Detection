"""
Live demo for the Credit Card Fraud Detection model.

Run locally:
    streamlit run app.py

Deploy for free at https://share.streamlit.io by connecting this repo.
"""
import os

import joblib
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(__file__)
MODEL_PATH = os.path.join(ROOT, "models", "logistic_regression_fraud_model.joblib")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.joblib")
SAMPLE_PATH = os.path.join(ROOT, "demo_data", "sample_transactions.csv")
THRESHOLD = 0.172  # tuned threshold from src/train.py, see outputs/metrics.json

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


@st.cache_data
def load_sample():
    return pd.read_csv(SAMPLE_PATH)


def score_transaction(row: pd.Series, model, scaler) -> float:
    df = pd.DataFrame([row])
    df[["Time_scaled", "Amount_scaled"]] = scaler.transform(df[["Time", "Amount"]])
    X = df.drop(columns=["Time", "Amount", "Class"], errors="ignore")
    return float(model.predict_proba(X)[:, 1][0])


st.title("💳 Credit Card Fraud Detector")
st.write(
    "A Logistic Regression model trained on the real-world "
    "[ULB Credit Card Fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) "
    "(284,807 transactions, 492 frauds). "
    "[View the full project on GitHub](https://github.com/arinsrivastava492-oss/Fraud-Detection)."
)

model, scaler = load_artifacts()
sample_df = load_sample()

st.subheader("Try a real transaction")
st.caption(
    "V1–V28 are PCA-anonymized features from the original dataset (not human-readable), "
    "so pick a sample transaction below to see the model score it."
)

idx = st.selectbox(
    "Pick a sample transaction",
    options=sample_df.index,
    format_func=lambda i: f"Transaction #{i}  —  Amount: ${sample_df.loc[i, 'Amount']:.2f}",
)
row = sample_df.loc[idx]

col1, col2 = st.columns(2)
col1.metric("Transaction amount", f"${row['Amount']:.2f}")
col2.metric("Actual label", "🚨 Fraud" if row["Class"] == 1 else "✅ Legitimate")

if st.button("Score this transaction", type="primary"):
    prob = score_transaction(row, model, scaler)
    pred = "🚨 FRAUD" if prob >= THRESHOLD else "✅ LEGITIMATE"

    st.metric("Fraud probability", f"{prob:.4f}")
    st.metric("Model prediction", pred, help=f"Decision threshold: {THRESHOLD}")

    if (prob >= THRESHOLD) == (row["Class"] == 1):
        st.success("Correct prediction ✅")
    else:
        st.warning("This one was misclassified — even a well-tuned model isn't perfect!")

with st.expander("Model performance on the full test set"):
    st.markdown(
        """
        | Metric | Score |
        |---|---|
        | ROC-AUC | 0.957 |
        | Accuracy | 99.91% |
        | Precision (fraud) | 74% |
        | Recall (fraud) | 77% |
        """
    )
    st.caption("See outputs/metrics.json in the repo for the full breakdown across all three model variants.")
