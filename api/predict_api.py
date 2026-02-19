"""
House Price Prediction API
===========================
Loads the saved model and preprocessing; exposes POST /predict and POST /predict_explain.
Run: uvicorn api.predict_api:app --reload
Or from project root: python -m uvicorn api.predict_api:app --reload
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Path to artifacts (run from project root)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_PATH = os.path.join(BASE, "models", "house_price_artifacts.joblib")

app = FastAPI(title="House Price Prediction API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
artifacts = None

# Human-readable feature labels and typical effect descriptions
FEATURE_LABELS = {
    "area_sqm": ("Area (m²)", "Larger area generally increases price."),
    "bedrooms": ("Bedrooms", "More bedrooms typically add value."),
    "bathrooms": ("Bathrooms", "Additional bathrooms boost price."),
    "age_years": ("Property age", "Newer properties often command higher prices."),
    "has_garage": ("Garage", "Having a garage adds value."),
    "has_garden": ("Garden", "A garden can increase price."),
    "near_school": ("Near school", "Proximity to schools can add premium."),
    "rooms_total": ("Total rooms", "More rooms usually increase price."),
    "location_Rural": ("Location: Rural", "Rural location affects price."),
    "location_Suburb": ("Location: Suburb", "Suburban location affects price."),
    "location_Midtown": ("Location: Midtown", "Midtown location affects price."),
    "location_Waterfront": ("Location: Waterfront", "Waterfront typically commands premium."),
}


def _get_label_and_reason(fname: str):
    """Resolve label/reason for a feature name (handles prefixes like num__, cat__)."""
    key = fname.split("__")[-1] if "__" in fname else fname
    return FEATURE_LABELS.get(key, (fname.replace("_", " ").title(), "Affects predicted price."))


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


def _build_row(features: HouseFeatures):
    return {
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


def _get_reasons(model, preprocessor, feature_cols, row_dict, X_enc):
    """Generate explanation reasons from model (feature_importances_ or coefficients)."""
    fnames = list(preprocessor.get_feature_names_out())
    reasons = []

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        pairs = sorted(zip(fnames, imp), key=lambda x: -x[1])[:6]
        for fname, importance in pairs:
            label, desc = _get_label_and_reason(fname)
            pct = round(importance * 100, 1)
            reasons.append({
                "feature": fname,
                "label": label,
                "reason": desc,
                "importance_pct": pct,
                "effect": "positive" if importance > 0 else "neutral",
            })
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        intercept = model.intercept_ if hasattr(model, "intercept_") else 0
        pairs = sorted(zip(fnames, coefs), key=lambda x: -abs(x[1]))[:6]
        for fname, coef in pairs:
            label, desc = _get_label_and_reason(fname)
            effect = "positive" if coef > 0 else "negative"
            reasons.append({
                "feature": fname,
                "label": label,
                "reason": desc,
                "effect": effect,
                "coefficient": round(float(coef), 2),
            })
    else:
        reasons.append({
            "feature": "model",
            "label": "Model prediction",
            "reason": "Prediction is based on the trained model; explanation not available for this model type.",
            "effect": "neutral",
        })
    return reasons


@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "docs": "/docs",
        "predict": "POST /predict",
        "predict_explain": "POST /predict_explain",
    }


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
    row = _build_row(features)
    X = pd.DataFrame([row])[feature_cols]
    X_enc = preprocessor.transform(X)
    price = float(model.predict(X_enc)[0])
    return {"predicted_price": round(price, 2)}


@app.post("/predict_explain")
def predict_explain(features: HouseFeatures):
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_cols = artifacts["feature_cols"]
    row = _build_row(features)
    X = pd.DataFrame([row])[feature_cols]
    X_enc = preprocessor.transform(X)
    price = float(model.predict(X_enc)[0])
    reasons = _get_reasons(model, preprocessor, feature_cols, row, X_enc)
    return {
        "predicted_price": round(price, 2),
        "reasons": reasons,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
