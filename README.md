# Python Basics for ML & House Price Prediction

A learning-focused codebase covering Python basics, NumPy, Pandas, and a full machine learning pipeline for **house price prediction**, including a trained model, REST API, and dashboard frontend.

---

## Overview

This project provides:

- **Learning materials** — Python basics for ML, NumPy, Pandas (guides, examples, Jupyter notebook)
- **Sample datasets** — Synthetic data with nulls and outliers for practice
- **House price model** — Train, tune, and save a regression model to predict house prices
- **Prediction API** — FastAPI service to get predictions and explanations
- **Dashboard UI** — Single-page app with a House Price Predictor that calls the API and shows results with reasons

---

## Project Structure

```
python-basics-aims/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── index.html                   # Dashboard + House Price Predictor (frontend)
│
├── python_basics_for_ml_guide.txt   # Text guide for Python/ML basics
├── python_basics_for_ml_examples.py # Runnable examples
├── numpy_basics.py              # NumPy intro (junior + senior)
├── pandas_basics.py             # Pandas intro (junior + senior)
├── numpy_pandas_guide.ipynb     # Jupyter notebook: NumPy & Pandas
│
├── prediction_models_demo.py    # Demo: 3 models, pick best, evaluate
│
├── data/
│   └── house_prices.csv         # ~1000 records, house features + price, nulls
├── models/
│   ├── house_price_artifacts.joblib  # Saved model + preprocessor
│   └── metrics.json                  # Best model metrics
│
├── api/
│   └── predict_api.py           # FastAPI: /predict, /predict_explain
│
├── generate_house_data.py       # Generate house_prices.csv
├── train_house_price_model.py   # Full ML pipeline: EDA → clean → 3 models → tune → save
│
├── HOUSE_PRICE_ML_PLAN.md       # ML pipeline plan
└── HOUSE_PRICE_GUIDE.md         # User guide: data, training, API
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate data (if needed)
python generate_house_data.py

# 3. Train the model
python train_house_price_model.py

# 4. Start the API
uvicorn api.predict_api:app --reload --port 8000

# serve the client 
python -m http.server 8080

# 5. Open the dashboard (in another terminal or browser)
# Serve index.html: e.g. python -m http.server 8080
# Then open http://localhost:8080
```

---

## Plan: House Price Predictor Frontend + Explanations API

### Goal

Convert the dashboard (`index.html`) into a frontend that:

1. Consumes the house price prediction API
2. Shows predicted price on screen
3. Displays **reasons** (explanations) for the prediction

---

### 1. API Changes (`api/predict_api.py`)

- **New endpoint: `POST /predict_explain`**
  - Accepts the same JSON body as `/predict` (area_sqm, bedrooms, bathrooms, age_years, location, has_garage, has_garden, near_school)
  - Returns:
    - `predicted_price` (float)
    - `reasons` (list of objects, e.g. `{"feature": "area_sqm", "effect": "positive", "reason": "Larger area tends to increase price"}`)

- **How reasons are generated**
  - Tree models (Random Forest, Gradient Boosting): use `feature_importances_` to rank features and assign human-readable reasons
  - Linear Regression: use coefficients (positive/negative) to describe direction of effect
  - Fallback: generic message if the model does not support explanations

- Keep existing `POST /predict` for backward compatibility

---

### 2. Training Pipeline (`train_house_price_model.py`)

- No changes needed: the preprocessor and model are already saved; feature names come from `preprocessor.get_feature_names_out()` at runtime
- Optional: save `feature_names_out` explicitly if useful for debugging

---

### 3. Frontend Changes (`index.html`)

- **New section: "House Price Predictor"**
  - Form inputs: area_sqm, bedrooms, bathrooms, age_years, location (dropdown: Downtown, Suburb, Rural, Midtown, Waterfront), has_garage, has_garden, near_school (checkboxes or 0/1)
  - "Predict" button

- **Results display**
  - Predicted price (formatted as currency)
  - List of reasons as bullet points or small cards
  - Loading state during the request
  - Error message if the API call fails

- **Configuration**
  - Configurable API base URL (e.g. `http://127.0.0.1:8000`)

- **Integration**
  - Add the predictor card into the existing dashboard layout and styling

---

### 4. Implementation Status

- [x] `POST /predict_explain` in the API (returns price + reasons from feature_importances_ or coefficients)
- [x] House Price Predictor card in `index.html`
- [x] JavaScript to call the API and render results and reasons
- [x] CORS enabled so the frontend can call the API from another port
- [x] `HOUSE_PRICE_GUIDE.md` updated

---

## Dependencies

- Python 3.8+
- numpy, pandas, scikit-learn, joblib
- fastapi, uvicorn (for the API)

---

## Further Reading

- `HOUSE_PRICE_GUIDE.md` — Detailed guide for data generation, training, and API usage
- `HOUSE_PRICE_ML_PLAN.md` — Full ML pipeline plan (EDA, cleaning, encoding, models, tuning)
- `python_basics_for_ml_guide.txt` — Python basics oriented toward machine learning
