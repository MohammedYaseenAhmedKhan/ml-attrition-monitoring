import pandas as pd
import numpy as np
import json
import os


def calculate_psi(expected, actual, bins=10):
    expected_percents, bin_edges = np.histogram(expected, bins=bins)
    actual_percents, _ = np.histogram(actual, bins=bin_edges)

    expected_percents = expected_percents / len(expected)
    actual_percents = actual_percents / len(actual)

    psi = np.sum(
        (actual_percents - expected_percents) *
        np.log((actual_percents + 1e-6) / (expected_percents + 1e-6))
    )

    return psi

def detect_drift(baseline_df, new_df, threshold=0.25):
    drift_report = {}

    for column in baseline_df.columns:
        if baseline_df[column].dtype in ["int64", "float64"]:
            psi_value = calculate_psi(
                baseline_df[column],
                new_df[column]
            )

            drift_report[column] = {
                "psi": float(psi_value),
                "drift_detected": bool(psi_value > threshold)
            }


    return drift_report

def main():
    baseline = pd.read_csv("data/raw/attrition.csv")

    # Drop same columns as training
    columns_to_drop = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]
    baseline = baseline.drop(columns=columns_to_drop)

    # Simulate new data by slightly modifying salary
    new_data = baseline.copy()
    new_data["MonthlyIncome"] = new_data["MonthlyIncome"] * 1.5

    report = detect_drift(baseline, new_data)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/drift_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Drift report generated.")


if __name__ == "__main__":
    main()
