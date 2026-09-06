"""
Credit Card Fraud Detection — Logistic Regression pipeline.

Trains and compares three logistic regression variants on the ULB
Credit Card Fraud dataset (284,807 transactions, 492 frauds, 0.17%
positive class):

    1. Baseline        — plain LogisticRegression
    2. Class-weighted   — class_weight='balanced'
    3. SMOTE-resampled — SMOTE oversampling + LogisticRegression

For imbalanced fraud data, accuracy is a misleading metric (predicting
"never fraud" already scores ~99.8%). We report and select on
ROC-AUC / PR-AUC / recall / precision instead, and tune the decision
threshold using the precision-recall curve.

Usage:
    python src/train.py
"""
import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "creditcard.csv")
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")
RANDOM_STATE = 42

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(DATA_PATH)


def run_eda(df):
    """Save a handful of exploratory plots to outputs/."""
    # Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["Class"].value_counts()
    sns.barplot(x=["Legitimate", "Fraud"], y=counts.values, ax=ax)
    ax.set_ylabel("Number of transactions")
    ax.set_title(
        f"Class balance — fraud is {100 * counts[1] / counts.sum():.3f}% of data"
    )
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "class_balance.png"), dpi=150)
    plt.close(fig)

    # Transaction amount by class (log scale, capped for readability)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cls, label, color in [(0, "Legitimate", "tab:blue"), (1, "Fraud", "tab:red")]:
        sns.kdeplot(
            np.log1p(df.loc[df["Class"] == cls, "Amount"]),
            label=label,
            ax=ax,
            color=color,
        )
    ax.set_xlabel("log(1 + Amount)")
    ax.set_title("Transaction amount distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "amount_distribution.png"), dpi=150)
    plt.close(fig)


def make_features(df):
    df = df.copy()
    scaler = StandardScaler()
    df[["Time_scaled", "Amount_scaled"]] = scaler.fit_transform(df[["Time", "Amount"]])
    df = df.drop(columns=["Time", "Amount"])
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y, scaler


def evaluate(name, model, X_test, y_test, results):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], output_dict=True)

    print(f"\n=== {name} (threshold = 0.50) ===")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC (avg precision): {pr_auc:.4f}")

    results[name] = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_fraud": report["Fraud"]["precision"],
        "recall_fraud": report["Fraud"]["recall"],
        "f1_fraud": report["Fraud"]["f1-score"],
        "accuracy": report["accuracy"],
    }
    return y_prob, y_pred


def tune_threshold(y_test, y_prob):
    """Pick the threshold that maximizes F1 on the fraud class."""
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1s[:-1])  # last point has no matching threshold
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, precision[:-1], label="Precision")
    ax.plot(thresholds, recall[:-1], label="Recall")
    ax.plot(thresholds, f1s[:-1], label="F1", linestyle="--")
    ax.axvline(best_threshold, color="black", linestyle=":", label=f"Best F1 @ {best_threshold:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. threshold (fraud class)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "threshold_tuning.png"), dpi=150)
    plt.close(fig)

    return best_threshold, f1s[best_idx]


def main():
    print("Loading data...")
    df = load_data()
    print(f"Shape: {df.shape}, fraud cases: {df['Class'].sum()} "
          f"({100 * df['Class'].mean():.3f}%)")

    run_eda(df)

    X, y, scaler = make_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results = {}

    # 1. Baseline logistic regression
    baseline = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    baseline.fit(X_train, y_train)
    evaluate("baseline", baseline, X_test, y_test, results)

    # 2. Class-weighted logistic regression
    weighted = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    )
    weighted.fit(X_train, y_train)
    y_prob_weighted, _ = evaluate("class_weighted", weighted, X_test, y_test, results)

    # 3. SMOTE-resampled logistic regression
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    smote_model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    smote_model.fit(X_train_sm, y_train_sm)
    y_prob_smote, _ = evaluate("smote", smote_model, X_test, y_test, results)

    # Pick the best model by PR-AUC (most informative metric for rare positive class)
    best_name = max(results, key=lambda k: results[k]["pr_auc"])
    best_model = {"baseline": baseline, "class_weighted": weighted, "smote": smote_model}[best_name]
    best_prob = {"baseline": None, "class_weighted": y_prob_weighted, "smote": y_prob_smote}[best_name]
    if best_prob is None:
        best_prob = best_model.predict_proba(X_test)[:, 1]

    print(f"\nBest model by PR-AUC: {best_name}")

    # Threshold tuning on the best model
    best_threshold, best_f1 = tune_threshold(y_test, best_prob)
    y_pred_tuned = (best_prob >= best_threshold).astype(int)
    tuned_report = classification_report(
        y_test, y_pred_tuned, target_names=["Legit", "Fraud"], output_dict=True
    )
    print(f"\n=== {best_name} (tuned threshold = {best_threshold:.3f}) ===")
    print(classification_report(y_test, y_pred_tuned, target_names=["Legit", "Fraud"]))

    results["best_model"] = best_name
    results["tuned_threshold"] = float(best_threshold)
    results["tuned_metrics"] = {
        "precision_fraud": tuned_report["Fraud"]["precision"],
        "recall_fraud": tuned_report["Fraud"]["recall"],
        "f1_fraud": tuned_report["Fraud"]["f1-score"],
        "accuracy": tuned_report["accuracy"],
    }

    # Confusion matrix plot for the tuned best model
    fig, ax = plt.subplots(figsize=(5, 5))
    cm = confusion_matrix(y_test, y_pred_tuned)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Confusion matrix — {best_name} @ threshold {best_threshold:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curves for all three models
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in [("baseline", baseline), ("class_weighted", weighted), ("smote", smote_model)]:
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC curves")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

    # Feature importance (coefficients) for the best model
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coefficient": best_model.coef_[0],
    }).sort_values("coefficient", key=np.abs, ascending=False)
    coefs.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 8))
    top = coefs.head(15).sort_values("coefficient")
    colors = ["tab:red" if c > 0 else "tab:blue" for c in top["coefficient"]]
    ax.barh(top["feature"], top["coefficient"], color=colors)
    ax.set_title("Top 15 features by |coefficient| (red = pushes toward fraud)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # Save model + scaler + metrics
    joblib.dump(best_model, os.path.join(MODEL_DIR, "logistic_regression_fraud_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR}/logistic_regression_fraud_model.joblib")
    print(f"Saved metrics to {OUT_DIR}/metrics.json")
    print(f"Saved plots to {OUT_DIR}/")


if __name__ == "__main__":
    main()
