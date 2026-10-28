import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "trained_model.joblib"
METRICS_PATH = Path(__file__).resolve().parents[1] / "models" / "model_metrics.json"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

def predict_one(model, values):
    row = pd.DataFrame([values])
    prediction = int(model.predict(row)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        if 1 in classes:
            probability = float(probabilities[classes.index(1)])
    return prediction, probability
