"""
House Price Prediction — Full ML Pipeline
=========================================
Follows: Problem definition → Load → EDA → Clean (missing + outliers) →
Encoding → Feature engineering → 3 models → Select best → Tune → Save.

Run: python train_house_price_model.py
Requires: data/house_prices.csv (run generate_house_data.py first if missing)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "house_prices.csv")
MODELS_DIR = os.path.join(BASE, "models")
TARGET = "price"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    """Load CSV; ensure data/house_prices.csv exists."""
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Data not found: {DATA_PATH}. Run: python generate_house_data.py"
        )
    df = pd.read_csv(DATA_PATH)
    print("1. DATA LOADED")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    return df


def eda(df):
    """Exploratory data analysis: summary, missing, correlation."""
    print("\n2. EDA")
    print("   describe():")
    print(df.describe().round(2).to_string())
    print("\n   Missing per column:")
    print(df.isna().sum().to_string())
    print("\n   dtypes:", df.dtypes.to_dict())
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in numeric and len(numeric) > 1:
        corr = df[numeric].corr()[TARGET].sort_values(ascending=False)
        print(f"\n   Correlation with '{TARGET}':")
        print(corr.to_string())
    return df


def clean_data(df):
    """Handle missing values and outliers (IQR capping)."""
    print("\n3. DATA CLEANING")
    df = df.copy()
    # Missing: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            print(f"   Filled '{col}' with median")
        else:
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) else "Unknown"
            df[col] = df[col].fillna(fill_val)
            print(f"   Filled '{col}' with mode: {fill_val}")

    # Outliers: IQR capping for numeric columns (including target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = (df[col] < low) | (df[col] > high)
        df[col] = df[col].clip(lower=low, upper=high)
        if before.sum() > 0:
            print(f"   Capped outliers in '{col}' (IQR 1.5)")
    print("   No remaining NaNs:", df.isna().sum().sum() == 0)
    return df


def prepare_xy(df, feature_cols=None):
    """Split into X and y; return feature column names for encoding."""
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != TARGET]
    y = df[TARGET].values
    X = df[feature_cols]
    return X, y, feature_cols


def get_preprocessor(df, numeric_cols, categorical_cols):
    """Build ColumnTransformer: scale numeric, one-hot encode categorical."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def run_pipeline():
    """Full pipeline: load → EDA → clean → encode → train 3 models → tune best → save."""
    df = load_data()
    df = eda(df)
    df = clean_data(df)

    # Feature engineering (simple)
    df["rooms_total"] = df["bedrooms"] + df["bathrooms"]
    feature_cols = [
        "area_sqm", "bedrooms", "bathrooms", "age_years",
        "has_garage", "has_garden", "near_school", "rooms_total", "location"
    ]
    numeric_cols = ["area_sqm", "bedrooms", "bathrooms", "age_years",
                   "has_garage", "has_garden", "near_school", "rooms_total"]
    categorical_cols = ["location"]

    X, y, _ = prepare_xy(df, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = get_preprocessor(df, numeric_cols, categorical_cols)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)
    feature_names_out = list(preprocessor.get_feature_names_out())

    print("\n4. ENCODING & FEATURES")
    print(f"   Features after encoding: {feature_names_out}")

    # 5. Model selection: train 3 models
    print("\n5. MODEL SELECTION (3 models)")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, max_depth=12),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE, max_depth=5),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train_enc, y_train)
        y_pred = model.predict(X_test_enc)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2, "model": model})
        print(f"   {name}: RMSE={rmse:,.0f}, MAE={mae:,.0f}, R²={r2:.4f}")

    best = min(results, key=lambda r: r["rmse"])
    print(f"\n   Best model (by RMSE): {best['name']}")

    # 6. Fine-tune best model (Gradient Boosting or Random Forest)
    print("\n6. FINE-TUNING (best model)")
    param_grid = (
        {
            "n_estimators": [100, 150],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "min_samples_leaf": [2, 4],
        }
        if best["name"] == "Gradient Boosting"
        else {
            "n_estimators": [100, 150],
            "max_depth": [8, 12, 16],
            "min_samples_leaf": [2, 4],
        }
    )
    base_estimator = (
        GradientBoostingRegressor(random_state=RANDOM_STATE)
        if best["name"] == "Gradient Boosting"
        else RandomForestRegressor(random_state=RANDOM_STATE)
    )
    search = GridSearchCV(
        base_estimator, param_grid, scoring="neg_root_mean_squared_error",
        cv=3, n_jobs=-1, verbose=0
    )
    search.fit(X_train_enc, y_train)
    best_model = search.best_estimator_
    best_pred = best_model.predict(X_test_enc)
    best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    print(f"   Best params: {search.best_params_}")
    print(f"   Tuned test RMSE: {best_rmse:,.0f}")

    # 7. Save artifacts for API
    os.makedirs(MODELS_DIR, exist_ok=True)
    artifacts = {
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target": TARGET,
    }
    joblib.dump(artifacts, os.path.join(MODELS_DIR, "house_price_artifacts.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "best_model": best["name"],
            "tuned_rmse": float(best_rmse),
            "tuned_mae": float(mean_absolute_error(y_test, best_pred)),
            "tuned_r2": float(r2_score(y_test, best_pred)),
        }, f, indent=2)
    print(f"\n7. SAVED to {MODELS_DIR}")
    print("   house_price_artifacts.joblib, metrics.json")
    return artifacts


if __name__ == "__main__":
    run_pipeline()
