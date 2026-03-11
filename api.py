# api.py
import os, json, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

MODELS_DIR = "models"
pre = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
mdl = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))

class PredictIn(BaseModel):
    # accepts arbitrary feature dict
    features: dict
    patient_name: str = "Unknown"

app = FastAPI(title="Heart Risk API")

def risk_bucket(p):
    # 0: very low, 4: very high
    # (match UI scale in app.py)
    if p < 0.20: return 0, "Very Low"
    if p < 0.40: return 1, "Low"
    if p < 0.60: return 2, "Medium"
    if p < 0.80: return 3, "High"
    return 4, "Very High"

@app.post("/predict")
def predict(inp: PredictIn):
    try:
        X = pd.DataFrame([inp.features])
        Xt = pre.transform(X)
        prob = float(mdl.predict_proba(Xt)[0,1])
        level, label = risk_bucket(prob)
        return {"prob": prob, "risk_level": level, "risk_label": label}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
