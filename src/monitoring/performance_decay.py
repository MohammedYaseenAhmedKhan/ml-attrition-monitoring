import json
import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report

ENCODER_PATH = "artifacts/encoders.pkl"
BASELINE_METRICS_PATH = "artifacts/metrics.json"
MODEL_PATH = "artifacts/model.pkl"
REPORT_PATH = "artifacts/performance_decay_report.json"


def load_baseline_f1():
    with open(BASELINE_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    # F1-score for attrition class (class 1)
    return metrics["1"]["f1-score"]


def evaluate_on_new_data(model):
    df = pd.read_csv("data/raw/attrition.csv")
    df["MonthlyIncome"] *= 1.8

    columns_to_drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df = df.drop(columns=columns_to_drop)

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    # Load encoders
    encoders = joblib.load(ENCODER_PATH)

    # Apply encoding safely
    for col in encoders:
        if col in X.columns:
            X[col] = encoders[col].transform(X[col])

    # Ensure no string types remain
    X = X.astype(float)

    preds = model.predict(X)

    report = classification_report(y, preds, output_dict=True)

    return report["1"]["f1-score"]


def detect_performance_decay(baseline_f1, current_f1, threshold=0.1):
    drop = baseline_f1 - current_f1

    decay_detected = drop > threshold

    return {
        "baseline_f1": float(baseline_f1),
        "current_f1": float(current_f1),
        "f1_drop": float(drop),
        "performance_decay_detected": bool(decay_detected),
        "retraining_recommended": bool(decay_detected),
    }


def main():
    model = joblib.load(MODEL_PATH)

    baseline_f1 = load_baseline_f1()
    current_f1 = evaluate_on_new_data(model)

    report = detect_performance_decay(baseline_f1, current_f1)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("Performance decay report generated.")


if __name__ == "__main__":
    main()
