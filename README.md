# ML Attrition Monitoring System

A production-oriented machine learning system that predicts employee attrition and continuously monitors model health after deployment using statistical drift detection and performance evaluation techniques.

---

## Overview

Most machine learning projects stop at model training.  
This project extends beyond that by implementing a complete ML lifecycle:

- Model training
- API-based deployment
- Prediction logging
- Data drift detection
- Prediction drift monitoring
- Performance decay detection
- Retraining recommendation logic

The focus is not just predictive accuracy, but **model reliability in production environments**.

---

## Architecture



+------------------+
| Training Layer |
+------------------+
|
v
+-------------------------------+
| Saved Artifacts |
| model.pkl |
| encoders.pkl |
| metrics.json |
+-------------------------------+
|
v
+------------------+
| FastAPI Server |
| POST /predict |
+------------------+
|
v
+---------------------------+
| Prediction Logging |
| prediction_logs.json |
+---------------------------+
|
v
+--------------------------------------+
| Monitoring Layer |
| - Data Drift (PSI) |
| - Prediction Drift |
| - Performance Decay |
+--------------------------------------+
|
v
+-----------------------------+
| Retraining Recommendation |
+-----------------------------+


---

## Dataset

**IBM HR Attrition Dataset**

**File Location:**


data/raw/attrition.csv


### Target Variable
- `Attrition` (Yes / No → 1 / 0)

### Removed Columns
- EmployeeCount
- EmployeeNumber
- Over18
- StandardHours

These were removed because they were constant or non-informative.

### Encoded Categorical Features
- BusinessTravel
- Department
- EducationField
- Gender
- JobRole
- MaritalStatus
- OverTime

LabelEncoder was used to ensure schema consistency across training and inference.

---

## Model

**Algorithm:** XGBoost Classifier  

**Evaluation Metrics:**
- Precision
- Recall
- F1-score (primary metric due to class imbalance)

Baseline F1-score is stored in:



artifacts/metrics.json


This value is later used for performance decay detection.

---

## Inference API

Built using **FastAPI** and served via **Uvicorn**.

### Endpoint



POST /predict


Supports:
- Single employee prediction
- Batch employee prediction

Returns:
- Binary prediction (0 or 1)
- Attrition probability

All predictions are logged to:



artifacts/prediction_logs.json


---

## Monitoring Capabilities

### 1. Data Drift Detection

Uses **Population Stability Index (PSI)** to detect distribution shifts between baseline and new data.

PSI Thresholds:
- < 0.1 → Stable
- 0.1 – 0.25 → Moderate Drift
- > 0.25 → Significant Drift

---

### 2. Prediction Drift Detection

Monitors changes in mean predicted probability.

Drift is detected if:



abs(current_mean - baseline_mean) > threshold


---

### 3. Performance Decay Detection

Compares:

- Baseline F1-score
- Current F1-score

If degradation exceeds a defined threshold, retraining is recommended.

---

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- xgboost
- FastAPI
- uvicorn
- joblib

---

## How to Run

 1. Train the Model

``bash
python src/training/train_model.py

2. Start the API
uvicorn src.inference.app:app --reload


Open:

http://127.0.0.1:8000/docs

3. Run Monitoring Scripts
python src/monitoring/data_drift.py
python src/monitoring/prediction_drift.py
python src/monitoring/performance_decay.py

#Key Engineering Learnings

Maintaining schema consistency between training and inference

Saving preprocessing artifacts for production use

Monitoring deployed ML systems using statistical techniques

Detecting silent model degradation

Handling real-world ML deployment errors
