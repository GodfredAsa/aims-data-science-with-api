# House Price Prediction: ML Pipeline & API — Plan

## Objective

Build an **estimator model** to predict house prices from features, compare **3 models**, select the **best** one, save it, and expose a **prediction API** for production use.

---

## 1. Problem Definition

- **Task:** Regression — predict continuous **price** from property and location features.
- **Input:** Features (e.g. area, bedrooms, location, age, amenities).
- **Output:** Predicted price.
- **Success:** Low error (e.g. RMSE/MAE) and a stable, deployable model + API.

---

## 2. Data Collection

- **Source:** Synthetic dataset (no external files).
- **Size:** ~1000 records.
- **Format:** CSV with headers.
- **Contents:**
  - **Target:** `price`
  - **Features:** e.g. `area_sqm`, `bedrooms`, `bathrooms`, `age_years`, `location` (categorical), `has_garage`, `has_garden`, etc.
- **Realism:** Include **null/missing values** in some columns (e.g. 3–8% per column) to practice cleaning.

Deliverable: `data/house_prices.csv` (or `house_prices.csv` in project root).

---

## 3. Exploratory Data Analysis (EDA)

- Load CSV, inspect shape, dtypes, missing counts.
- Summary statistics: mean, std, min, max, quartiles (e.g. `describe()`).
- Distributions: target (`price`) and key numeric features (histograms or similar).
- Correlations: correlation matrix with target; identify highly correlated features.
- Categorical: value counts for `location`, other categories.
- Outliers: simple checks (e.g. IQR or z-score) to inform cleaning step.

Deliverable: Script or notebook section that prints/plots the above; short comments on findings.

---

## 4. Data Cleaning (Including Outlier Treatment)

- **Missing values:**
  - Numeric: fill with median (or mean) per column, or drop if too many missing.
  - Categorical: fill with mode or "Unknown".
- **Outliers:**
  - Define strategy: e.g. IQR-based capping (clip at Q1 − k×IQR, Q3 + k×IQR) for numeric features, or cap extreme prices.
  - Apply to numeric columns (and optionally to target) and document choice.
- **Consistency:** Fix dtypes (e.g. integer columns, booleans), strip strings if needed.

Deliverable: Cleaned DataFrame/CSV or in-memory pipeline; no NaNs in columns used for modeling.

---

## 5. Encoding

- **Categorical features:** Map to numbers for tree and linear models.
  - Options: Label Encoding (e.g. `location` → 0,1,2…) or One-Hot Encoding (e.g. `get_dummies`).
  - Recommendation: One-Hot for linear models; Label or One-Hot for tree-based (sklearn accepts both with appropriate handling).
- **Target:** Keep continuous; no encoding.
- **Train/test:** Fit encoder (e.g. OneHotEncoder) on training set only, transform train and test to avoid leakage.

Deliverable: Feature matrix `X` (all numeric) and target `y`, with consistent encoding for train/test.

---

## 6. Feature Engineering and Selection

- **Engineering (optional but useful):**
  - Derived columns: e.g. `price_per_sqm`, `rooms_total` (bedrooms + bathrooms), `age_bucket`.
  - Polynomial or interaction terms if using linear model (optional).
- **Selection:**
  - Drop low-importance or redundant columns (e.g. from correlation or model-based importance).
  - Option: use sklearn `SelectKBest` or tree feature_importances_; keep a small set of interpretable features.
- **Final feature set:** Document which columns are used in the model.

Deliverable: Final list of feature names and transformed `X_train`, `X_test`, `y_train`, `y_test` (after train_test_split).

---

## 7. Model Selection (3 Models)

- **Train/test split:** e.g. 80/20 or 70/30, fixed `random_state` for reproducibility.
- **Models to implement:**
  1. **Linear Regression** — baseline, interpretable.
  2. **Random Forest Regressor** — robust, handles non-linearity.
  3. **Gradient Boosting Regressor** (e.g. sklearn `GradientBoostingRegressor` or lightgbm/xgboost if preferred) — often best accuracy.
- **Evaluation metrics:** RMSE, MAE, R² (and optionally MAPE) on **test set**.
- **Comparison:** Table of metrics; choose **best model** by lowest RMSE (or preferred metric).

Deliverable: Trained models, test metrics, and a clear “best model” (e.g. Gradient Boosting) saved to disk.

---

## 8. Training, Fitting, and Testing

- **Training:** Fit each model on `X_train`, `y_train`.
- **Validation:** Optionally use a validation set or cross-validation for tuning (see below).
- **Testing:** Predict on `X_test`; compute RMSE, MAE, R²; no refitting on test.
- **Saving:** Save the **best model** (e.g. `joblib` or `pickle`) plus fitted preprocessing (scaler, encoder) so the API can reproduce the same pipeline.

Deliverable: Script that trains all three, evaluates, picks best, saves best model + preprocessing artifacts.

---

## 9. Fine-Tuning (Hyperparameter Tuning)

- **Scope:** Apply to the best model (e.g. Random Forest or Gradient Boosting).
- **Method:** GridSearchCV or RandomizedSearchCV on training (or validation) data.
- **Parameters:** e.g. `n_estimators`, `max_depth`, `learning_rate` (for GB), `min_samples_leaf`.
- **Metric:** Minimize RMSE or MAE (via `scoring`).
- **Final model:** Retrain on full training set with best params, then evaluate once on test set and save this tuned model.

Deliverable: Tuned best model and preprocessing saved; optional summary of best hyperparameters.

---

## 10. API for Predictions

- **Purpose:** Load the saved best model (and preprocessing) and expose an HTTP endpoint to get predictions.
- **Stack:** FastAPI or Flask (e.g. FastAPI for async and automatic docs).
- **Endpoint:** e.g. `POST /predict` with JSON body: `{"area_sqm": 120, "bedrooms": 3, ...}`.
- **Flow:** Parse JSON → build feature vector in same order as training → apply same preprocessing (scaler, encoding) → model.predict → return `{"predicted_price": ...}`.
- **Artifacts:** Model file + scaler/encoder (or a single pipeline pickle) in a folder like `models/` or `artifacts/`.

Deliverable: API script (e.g. `api/predict_api.py` or `predict_api.py`), requirements for API (e.g. fastapi, uvicorn), and short instructions to run and call the API.

---

## 11. Documentation / Guide

- **README or GUIDE:**
  - How to generate/use the CSV.
  - How to run the full ML pipeline (steps 3–9) in order.
  - How to run the API and example `curl` or Python request for `/predict`.
  - Summary of the three models and why the chosen one is best (metrics + brief rationale).
- **Order of execution:**
  1. Generate data (if not already) → CSV.
  2. Run training script (EDA → clean → encode → feature eng → train 3 models → tune best → save).
  3. Run API server; call `/predict` with sample payload.

---

## File Layout (Proposed)

```
project/
├── HOUSE_PRICE_ML_PLAN.md          # This plan
├── data/
│   └── house_prices.csv            # ~1000 rows, nulls, features + price
├── train_house_price_model.py      # Full pipeline: load, EDA, clean, encode, 3 models, tune, save
├── models/                         # Saved model + preprocessing (created by script)
│   └── (best_model.joblib, preprocessing.joblib or pipeline.joblib)
├── api/
│   └── predict_api.py              # FastAPI app: load model, POST /predict
├── requirements.txt               # numpy, pandas, scikit-learn, fastapi, uvicorn, joblib
└── HOUSE_PRICE_GUIDE.md            # User-facing guide: run pipeline, run API, example requests
```

---

## Summary Checklist

- [ ] Problem definition
- [ ] Data collection (CSV, ~1000 rows, nulls)
- [ ] EDA (summary, distributions, correlations, outliers)
- [ ] Data cleaning (missing, outliers, dtypes)
- [ ] Encoding (categoricals)
- [ ] Feature engineering and selection
- [ ] Model selection (3 models: Linear, Random Forest, Gradient Boosting)
- [ ] Train, fit, test (metrics; choose best)
- [ ] Fine-tuning (best model only)
- [ ] Save best model + preprocessing
- [ ] API (load model, POST /predict, same preprocessing)
- [ ] Guide (run pipeline, run API, example requests)
