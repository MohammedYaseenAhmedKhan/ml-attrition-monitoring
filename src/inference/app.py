import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI
from .schemas import EmployeeData

MODEL_PATH = "artifacts/model.pkl"
ENCODER_PATH = "artifacts/encoders.pkl"
LOG_PATH = "artifacts/prediction_logs.json"

app = FastAPI(title="Attrition Monitoring API")

# Load model at startup
model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


@app.get("/")
def health_check():
    return {"status": "API running"}


from typing import List


@app.post("/predict")
def predict(data: List[EmployeeData]):
    results = []

    for employee in data:
        input_dict = employee.dict()
        df = pd.DataFrame([input_dict])

        prediction = model.predict(df)[0]
        probability = float(model.predict_proba(df)[0][1])

        result = {"prediction": int(prediction), "attrition_probability": probability}

        log_prediction(input_dict, result)
        results.append(result)

    return results

    input_dict = data.dict()
    df = pd.DataFrame([input_dict])

    prediction = model.predict(df)[0]
    probability = float(model.predict_proba(df)[0][1])

    result = {"prediction": int(prediction), "attrition_probability": probability}

    log_prediction(input_dict, result)

    return result


def log_prediction(input_data, output_data):
    os.makedirs("artifacts", exist_ok=True)

    log_entry = {"input": input_data, "output": output_data}

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(LOG_PATH, "w") as f:
        json.dump(logs, f, indent=4)
