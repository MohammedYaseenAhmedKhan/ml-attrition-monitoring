 ML Attrition Monitoring System

A production-style machine learning system that not only predicts employee attrition but continuously monitors model health through data drift, prediction drift, and performance decay detection.

 Problem Statement

Machine learning models often degrade silently after deployment due to:

Changing data distributions

Shifts in model confidence

Real-world behavioral changes

This project demonstrates a full ML lifecycle system that detects such failures and recommends retraining when necessary.

System Architecture
Training Layer
    ↓
Model Artifacts (model.pkl, encoders.pkl, metrics.json)
    ↓
FastAPI Inference Service
    ↓
Prediction Logging
    ↓
Monitoring Layer
    ├── Data Drift Detection (PSI)
    ├── Prediction Drift Detection
    └── Performance Decay Detection
    ↓
Retraining Recommendation

 Project Structure
src/
├── training/
│   └── train_model.py
├── inference/
│   ├── app.py
│   └── schemas.py
└── monitoring/
    ├── data_drift.py
    ├── prediction_drift.py
    └── performance_decay.py

 Training Layer

Dataset: IBM HR Attrition Dataset

Model: XGBoost Classifier

Feature Engineering:

Dropped non-informative columns:

EmployeeCount

EmployeeNumber

Over18

StandardHours

Label encoding for categorical features

Evaluation:

Precision

Recall

F1-score

Artifacts saved:

model.pkl

encoders.pkl

metrics.json

Baseline F1-score is stored for future performance comparison.

 Inference Layer (FastAPI)

The trained model is deployed locally using FastAPI.

Endpoints:

GET / → Health check

POST /predict → Returns:

Prediction (0/1)

Attrition probability

Every prediction is logged to:

artifacts/prediction_logs.json


This enables real-time monitoring.

 Monitoring Layer
1️ Data Drift Detection

Uses Population Stability Index (PSI) to compare:

Baseline training distribution

New incoming data distribution

Threshold:

PSI < 0.1 → Stable

0.1–0.25 → Moderate drift

0.25 → Significant drift

Output:

artifacts/drift_report.json

2️ Prediction Drift Detection

Monitors shift in average predicted probability.

Compares:

Baseline mean probability

Current mean probability

Flags drift if deviation exceeds threshold.

Output:

artifacts/prediction_drift_report.json

3️ Performance Decay Detection

Compares:

Baseline F1-score

Current F1-score on new labeled data

If F1 drop exceeds threshold:

retraining_recommended = true


Output:

artifacts/performance_decay_report.json

 Retraining Logic

The system recommends retraining if:

Significant performance decay is detected

Monitoring thresholds are exceeded

This simulates enterprise ML lifecycle management.

 How to Run
1️ Train Model
python src/training/train_model.py

2️ Start API
uvicorn src.inference.app:app --reload


Open:

http://127.0.0.1:8000/docs

3️ Run Data Drift Detection
python src/monitoring/data_drift.py

4️ Run Prediction Drift Detection
python src/monitoring/prediction_drift.py

5️ Run Performance Decay Detection
python src/monitoring/performance_decay.py

 Key Learnings

Importance of feature consistency between training and inference

Handling schema mismatches in production ML systems

Statistical drift detection using PSI

Monitoring prediction behavior over time

Detecting performance degradation and triggering retraining decisions

 Why This Project Is Different

This is not just a classification model.

It demonstrates:

End-to-end ML system design

Deployment simulation

Monitoring and lifecycle management

Production-oriented engineering thinking

 Future Improvements

Dockerization

Scheduled monitoring jobs

Automated retraining pipeline

Cloud deployment

CI/CD integration

 Final Outcome

A complete machine learning lifecycle system capable of:

Training

Deployment

Monitoring

Drift detection

Performance evaluation

Retraining recommendation