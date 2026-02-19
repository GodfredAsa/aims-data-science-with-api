"""
Prediction Models Demo — Sample Data with Preprocessing and Model Comparison
============================================================================

This module demonstrates a full pipeline for building prediction models on
rough real-world-like data: it includes missing values and outliers, shows
how to treat them, then trains and evaluates three different models.

Contents
--------
- generate_sample_data()   : Builds synthetic data with nulls and outliers.
- treat_missing_values()   : Fills NaNs with column median.
- treat_outliers_iqr()     : Caps outliers using IQR method.
- prepare_xy()             : Splits DataFrame into X and y.
- evaluate_model()         : Computes MSE, MAE, R².
- run_pipeline()           : End-to-end: data → preprocess → train → evaluate.

Usage
-----
  From command line:
    python prediction_models_demo.py

  From another script:
    from prediction_models_demo import run_pipeline
    df_clean, results = run_pipeline(seed=42)

Models compared
---------------
  1. Linear Regression  — baseline; interpretable, can underfit.
  2. Decision Tree      — captures non-linearity; can overfit.
  3. Random Forest      — usually best trade-off: accuracy and robustness.

The script prints test-set metrics and states which model is preferred and why.

Target audience: learners moving from Python basics toward machine learning.

Dependencies
------------
  numpy, pandas, scikit-learn
  Install: pip install numpy pandas scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# -----------------------------------------------------------------------------
# 1. SAMPLE DATA GENERATION (with intentional outliers and nulls)
# -----------------------------------------------------------------------------

def generate_sample_data(n_rows=200, seed=42):
    """
    Generate synthetic dataset for regression with outliers and missing values.

    Simulates a scenario like "predicting house price from area, age, rooms".
    Outliers and nulls are injected to mimic real messy data.

    Parameters
    ----------
    n_rows : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with columns: area_sqm, age_years, n_rooms, price (target).
    """
    rng = np.random.default_rng(seed)

    # Base features (roughly realistic ranges)
    area_sqm = rng.uniform(40, 200, n_rows)
    age_years = rng.integers(0, 50, n_rows)
    n_rooms = rng.integers(1, 6, n_rows)

    # Target: price (with some relationship to features + noise)
    price = (
        50 * area_sqm
        - 2 * age_years * area_sqm / 100
        + 20_000 * n_rooms
        + rng.normal(0, 15_000, n_rows)
    )
    price = np.clip(price, 30_000, 800_000)  # keep in plausible range before outliers

    df = pd.DataFrame({
        "area_sqm": area_sqm,
        "age_years": age_years,
        "n_rooms": n_rooms,
        "price": price,
    })

    # --- Inject missing values (e.g. 5–10% per column) ---
    for col in ["area_sqm", "age_years", "n_rooms"]:
        n_missing = rng.integers(int(0.05 * n_rows), int(0.12 * n_rows))
        idx = rng.choice(df.index, size=n_missing, replace=False)
        df.loc[idx, col] = np.nan

    # --- Inject outliers (a few extreme values) ---
    n_outliers = rng.integers(5, 15)
    out_idx = rng.choice(df.index, size=n_outliers, replace=False)
    df.loc[out_idx, "area_sqm"] = rng.uniform(350, 600, n_outliers)   # huge areas
    df.loc[out_idx, "price"] = rng.uniform(1_200_000, 2_500_000, n_outliers)  # very high prices

    return df


# -----------------------------------------------------------------------------
# 2. PREPROCESSING (treat missing values and outliers)
# -----------------------------------------------------------------------------

def treat_missing_values(df, strategy="median"):
    """
    Fill missing values with column median (numeric). Preserves DataFrame index.

    Parameters
    ----------
    df : pd.DataFrame
        Data with possible NaNs.
    strategy : str
        Currently only "median" is used (robust to outliers).

    Returns
    -------
    pd.DataFrame
        Copy of df with NaNs filled.
    """
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if out[col].isna().any():
            fill_val = out[col].median()
            out[col] = out[col].fillna(fill_val)
    return out


def treat_outliers_iqr(df, columns, factor=1.5):
    """
    Cap outliers using the IQR method: values beyond Q1 - factor*IQR or
    Q3 + factor*IQR are clipped to those bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Data with possible outliers.
    columns : list of str
        Column names to treat.
    factor : float
        IQR multiplier (default 1.5).

    Returns
    -------
    pd.DataFrame
        Copy of df with outliers capped.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def prepare_xy(df, target_col="price", feature_cols=None):
    """
    Split dataframe into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data.
    target_col : str
        Name of target column.
    feature_cols : list of str or None
        If None, use all columns except target_col.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y


# -----------------------------------------------------------------------------
# 3. MODEL TRAINING AND EVALUATION
# -----------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Compute MSE, MAE, and R² and return as a dict.

    Parameters
    ----------
    y_true : array-like
        Ground truth target.
    y_pred : array-like
        Predicted target.
    model_name : str
        Label for the results.

    Returns
    -------
    dict
        Keys: model_name, mse, mae, r2.
    """
    return {
        "model_name": model_name,
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def print_evaluation(results):
    """Print evaluation metrics in a readable format."""
    for r in results:
        print(f"  {r['model_name']}:")
        print(f"    MSE  = {r['mse']:,.2f}")
        print(f"    MAE  = {r['mae']:,.2f}")
        print(f"    R²   = {r['r2']:.4f}")
        print()


def run_pipeline(seed=42):
    """
    Run the full pipeline: generate data, preprocess, train three models,
    evaluate them, and state which is preferred and why.

    Returns
    -------
    tuple of (pd.DataFrame, list of dict)
        Preprocessed dataframe and list of evaluation result dicts.
    """
    # 1) Generate rough data
    df_raw = generate_sample_data(n_rows=200, seed=seed)
    print("Raw data shape:", df_raw.shape)
    print("Missing counts:\n", df_raw.isna().sum(), "\n")

    # 2) Preprocess: treat missing then outliers
    df_clean = treat_missing_values(df_raw)
    feature_cols = ["area_sqm", "age_years", "n_rooms"]
    df_clean = treat_outliers_iqr(df_clean, columns=feature_cols + ["price"], factor=1.5)
    print("After preprocessing: no missing values; outliers capped (IQR method).\n")

    # 3) Prepare X, y and split
    X, y = prepare_xy(df_clean, target_col="price", feature_cols=feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    # Scale features (important for Linear Regression; harmless for tree-based)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 4) Train three models
    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=seed, max_depth=6)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=seed, max_depth=8)),
    ]

    results = []
    for name, model in models:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        results.append(evaluate_model(y_test, y_pred, model_name=name))

    # 5) Report
    print("Evaluation on test set (lower MSE/MAE is better; higher R² is better):\n")
    print_evaluation(results)

    # 6) State preferred model and reason
    best = min(results, key=lambda r: r["mse"])
    print("Preferred model:", best["model_name"])
    print(
        "Reason: It achieves the best test MSE (and typically R²) on this dataset. "
        "Linear Regression is simple but can underfit when relationships are "
        "non-linear. Decision Trees can overfit. Random Forest usually gives a "
        "better trade-off: good accuracy and robustness to outliers/noise after "
        "preprocessing, without heavy overfitting when tuned (e.g. max_depth)."
    )
    if best["model_name"] != "Random Forest":
        print(
            "(On different data or seeds, Random Forest might still be preferred "
            "for robustness; here we pick the one with best test MSE.)"
        )

    return df_clean, results


# -----------------------------------------------------------------------------
# 4. MAIN (run when script is executed)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    df_clean, results = run_pipeline(seed=42)
