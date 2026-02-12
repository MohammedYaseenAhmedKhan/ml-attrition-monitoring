# Production-Grade ML Monitoring System for Employee Attrition

An end-to-end machine learning system that focuses on model reliability after deployment, including data drift detection, prediction monitoring, and performance degradation tracking.

---

## Problem Statement

Machine learning models degrade in real-world production systems due to changes in input data distribution.

This project simulates a production ML lifecycle where an employee attrition prediction model is deployed and continuously monitored for:

- Data drift
- Prediction distribution shift
- Performance decay
- Retraining readiness

The objective is to detect silent failures before business impact occurs.

---

## Phase 1: Baseline Model

- Dataset: IBM HR Employee Attrition
- Model: XGBoost Classifier
- Stratified train-test split
- Model artifact persistence (model + encoders)

---

## Project Structure

ml-attrition-monitoring/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ └── training/
│ └── train_model.py
│
├── artifacts/
│ ├── model.pkl
│ └── encoders.pkl
│
├── requirements.txt
├── .gitignore
└── README.md


---

## How to Run

1. Create virtual environment
   python -m venv venv

2. Activate environment (Windows)
   .\venv\Scripts\Activate.ps1

3. Install dependencies
   pip install -r requirements.txt

4. Run training
   python src/training/train_model.py

---

## Upcoming Phases

- FastAPI inference service
- Statistical drift detection (PSI, KS Test)
- Prediction monitoring
- Alerting logic
- Automated retraining readiness
