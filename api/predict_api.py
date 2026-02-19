"""
House Price Prediction API
===========================
Loads the saved model and preprocessing; exposes POST /predict.
Run: uvicorn api.predict_api:app --reload
Or from project root: python -m uvicorn api.predict_api:app --reload
"""
import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Path to artifacts (run from project root)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_PATH = os.path.join(BASE, "models", "house_price_artifacts.joblib")

app = FastAPI(title="House Price Prediction API", version="1.0")
artifacts = None


def load_artifacts():
    global artifacts
    if not os.path.isfile(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f"Artifacts not found: {ARTIFACTS_PATH}. Run train_house_price_model.py first."
        )
    artifacts = joblib.load(ARTIFACTS_PATH)
    return artifacts


@app.on_event("startup")
def startup():
    load_artifacts()


class HouseFeatures(BaseModel):
    area_sqm: float
    bedrooms: int
    bathrooms: int
    age_years: int
    location: str
    has_garage: int
    has_garden: int
    near_school: int
    # rooms_total computed from bedrooms + bathrooms


@app.get("/")
def root():
    return {"message": "House Price Prediction API", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifacts is not None}


@app.post("/predict")
def predict(features: HouseFeatures):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_cols = artifacts["feature_cols"]

    # Build one row with same columns as training (include rooms_total)
    row = {
        "area_sqm": features.area_sqm,
        "bedrooms": features.bedrooms,
        "bathrooms": features.bathrooms,
        "age_years": features.age_years,
        "has_garage": features.has_garage,
        "has_garden": features.has_garden,
        "near_school": features.near_school,
        "rooms_total": features.bedrooms + features.bathrooms,
        "location": features.location,
    }
    import pandas as pd
    X = pd.DataFrame([row])[feature_cols]
    X_enc = preprocessor.transform(X)
    price = float(model.predict(X_enc)[0])
    return {"predicted_price": round(price, 2)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
