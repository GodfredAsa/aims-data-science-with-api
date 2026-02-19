# House Price Prediction — User Guide

This guide walks you through **generating the dataset**, **running the ML pipeline**, and **using the prediction API**, following the steps in `HOUSE_PRICE_ML_PLAN.md`.

---

## Prerequisites

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 1. Generate Sample Data

Create the CSV with ~1000 records and null values:

```bash
python generate_house_data.py
```

Output: `data/house_prices.csv`

**Columns:**

| Column       | Type    | Description                    |
|-------------|---------|--------------------------------|
| area_sqm    | float   | Living area (m²)              |
| bedrooms    | int     | Number of bedrooms            |
| bathrooms   | int     | Number of bathrooms           |
| age_years   | int     | Age of property (years)       |
| location    | str     | Downtown, Suburb, Rural, Midtown, Waterfront |
| has_garage   | 0/1     | Has garage                    |
| has_garden   | 0/1     | Has garden                    |
| near_school  | 0/1     | Near school                   |
| price       | int     | **Target** — price to predict |

The script injects missing values in several columns (about 1–6% per column) for practice.

---

## 2. Run the ML Pipeline

Train models, compare three estimators, tune the best one, and save artifacts:

```bash
python train_house_price_model.py
```

**What the script does (in order):**

1. **Load data** — reads `data/house_prices.csv`
2. **EDA** — summary stats, missing counts, correlations with `price`
3. **Data cleaning**
   - Fills missing numeric values with **median**, categorical with **mode**
   - **Outlier treatment**: IQR-based capping (1.5×IQR) on numeric columns
4. **Encoding** — numeric columns scaled (StandardScaler), `location` one-hot encoded (drop first)
5. **Feature engineering** — adds `rooms_total = bedrooms + bathrooms`
6. **Model selection** — trains three models on an 80/20 train–test split:
   - **Linear Regression**
   - **Random Forest**
   - **Gradient Boosting**
7. **Evaluation** — reports RMSE, MAE, R² on the test set and **selects the best** by lowest RMSE
8. **Fine-tuning** — runs `GridSearchCV` on the best model (Gradient Boosting or Random Forest)
9. **Save** — writes to `models/`:
   - `house_price_artifacts.joblib` (model + preprocessor + feature list)
   - `metrics.json` (best model name, tuned RMSE/MAE/R²)

**Why one model is preferred:** The script prints metrics for all three; the one with the **lowest test RMSE** is chosen. Usually Gradient Boosting or Random Forest wins because they capture non-linear relationships; Linear Regression is a simple baseline. The guide in the script output states which model was selected and that tuning was applied to that model.

---

## 3. Run the Prediction API

Start the API server (from the **project root**):

```bash
uvicorn api.predict_api:app --reload --host 0.0.0.0 --port 8000
```

Or:

```bash
python -m uvicorn api.predict_api:app --reload --port 8000
```

- **Docs:** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/health  

---

## 4. Call the API

**POST /predict** — send a JSON body with the same features as in the dataset (no `price`; no `rooms_total` — it is computed as `bedrooms + bathrooms`).

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "area_sqm": 120,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10,
    "location": "Suburb",
    "has_garage": 1,
    "has_garden": 1,
    "near_school": 1
  }'
```

Example response:

```json
{"predicted_price": 345678.12}
```

**Allowed values for `location`:** `"Downtown"`, `"Suburb"`, `"Rural"`, `"Midtown"`, `"Waterfront"` (same as in the training data).

**Python example:**

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/predict",
    json={
        "area_sqm": 120,
        "bedrooms": 3,
        "bathrooms": 2,
        "age_years": 10,
        "location": "Suburb",
        "has_garage": 1,
        "has_garden": 1,
        "near_school": 1,
    },
)
print(resp.json())  # {"predicted_price": 345678.12}
```

---

## 5. Order of Execution Summary

| Step | Command / Action |
|------|-------------------|
| 1 | `python generate_house_data.py` → creates `data/house_prices.csv` |
| 2 | `python train_house_price_model.py` → EDA, clean, train, tune, save to `models/` |
| 3 | `uvicorn api.predict_api:app --reload --port 8000` → start API |
| 4 | `POST /predict` with JSON body → get `predicted_price` |

---

## 6. Files Reference

| File | Purpose |
|------|---------|
| `HOUSE_PRICE_ML_PLAN.md` | High-level plan (problem, EDA, clean, encode, models, tune, API) |
| `generate_house_data.py` | Generate `data/house_prices.csv` with nulls |
| `data/house_prices.csv` | Sample dataset (~1000 rows) |
| `train_house_price_model.py` | Full pipeline: load → EDA → clean → encode → 3 models → tune best → save |
| `models/house_price_artifacts.joblib` | Saved best model + preprocessor + feature list |
| `models/metrics.json` | Best model name and tuned metrics |
| `api/predict_api.py` | FastAPI app: load artifacts, expose POST /predict |
| `HOUSE_PRICE_GUIDE.md` | This user guide |

For more detail on the ML steps (problem definition, data collection, EDA, cleaning, encoding, feature engineering, model selection, training, testing, fine-tuning, and API), see **HOUSE_PRICE_ML_PLAN.md**.
