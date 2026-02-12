import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/raw/attrition.csv"
ARTIFACT_DIR = "artifacts"


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    df = df.copy()

    encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    return X, y, encoders


def train(X_train, y_train):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))


def save_artifacts(model, encoders):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.pkl"))
    joblib.dump(encoders, os.path.join(ARTIFACT_DIR, "encoders.pkl"))


def main():
    df = load_data(DATA_PATH)
    X, y, encoders = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_artifacts(model, encoders)


if __name__ == "__main__":
    main()
