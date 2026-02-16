import json
import os
import numpy as np


LOG_PATH = "artifacts/prediction_logs.json"
REPORT_PATH = "artifacts/prediction_drift_report.json"


def load_predictions():
    if not os.path.exists(LOG_PATH):
        print("No prediction logs found.")
        return []

    with open(LOG_PATH, "r") as f:
        logs = json.load(f)

    probabilities = [entry["output"]["attrition_probability"] for entry in logs]

    return probabilities


def detect_prediction_drift(probabilities, baseline_mean=0.2, threshold=0.1):
    if len(probabilities) == 0:
        return None

    current_mean = np.mean(probabilities)

    drift_detected = abs(current_mean - baseline_mean) > threshold

    report = {
        "baseline_mean_probability": baseline_mean,
        "current_mean_probability": float(current_mean),
        "difference": float(abs(current_mean - baseline_mean)),
        "drift_detected": bool(drift_detected),
    }

    return report


def main():
    probabilities = load_predictions()

    report = detect_prediction_drift(probabilities)

    if report is None:
        return

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("Prediction drift report generated.")


if __name__ == "__main__":
    main()
