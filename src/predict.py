"""
src/predict.py
==============
Inference module for customer churn prediction.
Handles both single-customer scoring and batch prediction from a CSV.

Usage (CLI):
    # Score a single customer (JSON string)
    python src/predict.py --customer '{"tenure": 5, "MonthlyCharges": 70.5, ...}'

    # Score a CSV file
    python src/predict.py --input data/new_customers.csv --output data/scored.csv

Usage (Python):
    from src.predict import load_pipeline, predict_single, predict_batch
    model, scaler, feature_cols = load_pipeline()
    result = predict_single(model, scaler, feature_cols, customer_data_dict)
"""

import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Paths to saved artefacts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "models", "feature_cols.json")

SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
              "charge_per_tenure", "charges_ratio"]

RISK_THRESHOLDS = {
    "High":   0.60,
    "Medium": 0.35,
}


# ─── Load artefacts ─────────────────────────────────────────────────────────────
def load_pipeline(model_path: str = MODEL_PATH,
                  scaler_path: str = SCALER_PATH,
                  feature_cols_path: str = FEATURE_COLS_PATH):
    """
    Load the trained XGBoost model, StandardScaler, and feature column list.

    Returns
    -------
    model       : fitted XGBClassifier
    scaler      : fitted StandardScaler
    feature_cols: list[str] — column order expected by the model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run src/train.py first."
        )

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if os.path.exists(feature_cols_path):
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
    else:
        # Fallback: derive from model if XGBoost stores feature names
        feature_cols = getattr(model, "feature_names_in_", None)
        if feature_cols is not None:
            feature_cols = list(feature_cols)
        else:
            raise FileNotFoundError(
                f"Feature column list not found at '{feature_cols_path}'. "
                "Ensure train.py saves this file."
            )

    print(f"[load_pipeline] Model : {model_path}")
    print(f"[load_pipeline] Scaler: {scaler_path}")
    print(f"[load_pipeline] Features: {len(feature_cols)} columns")
    return model, scaler, feature_cols


# ─── Risk label helper ──────────────────────────────────────────────────────────
def risk_label(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["High"]:
        return "High"
    elif probability >= RISK_THRESHOLDS["Medium"]:
        return "Medium"
    else:
        return "Low"


# ─── Preprocess a raw customer dict ─────────────────────────────────────────────
def preprocess_customer(raw: dict, feature_cols: list,
                         scaler, scale_cols: list = SCALE_COLS) -> pd.DataFrame:
    """
    Convert a raw customer dictionary (as received by the API or from a CSV row)
    into a correctly ordered, scaled feature DataFrame.

    The raw dict is expected to already have engineered features; for production
    you would run the preprocess + features pipeline first. This function handles
    alignment and scaling.

    Parameters
    ----------
    raw          : dict  — customer feature values (key = column name)
    feature_cols : list  — expected column order from training
    scaler       : fitted StandardScaler
    scale_cols   : list  — columns to apply scaling to

    Returns
    -------
    pd.DataFrame with 1 row, ready for model.predict_proba()
    """
    df = pd.DataFrame([raw])

    # Add any missing columns as 0 (unseen categories, etc.)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Keep only training columns in correct order, and force numeric types
    df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale numeric columns
    scale_present = [c for c in scale_cols if c in df.columns]
    if scale_present:
        df[scale_present] = scaler.transform(df[scale_present])

    return df


# ─── Predict Single Customer ─────────────────────────────────────────────────────
def predict_single(model, scaler, feature_cols: list,
                   customer_data: dict) -> dict:
    """
    Score one customer and return a structured result dict.

    Parameters
    ----------
    model         : fitted XGBClassifier
    scaler        : fitted StandardScaler
    feature_cols  : list[str]
    customer_data : dict — raw feature values

    Returns
    -------
    dict:
        {
          "churn_probability": float,
          "predicted_churn":   int (0 or 1),
          "risk_level":        str ("Low" | "Medium" | "High"),
          "top_risk_factors":  list[str]  (requires SHAP)
        }
    """
    X = preprocess_customer(customer_data, feature_cols, scaler)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    result = {
        "churn_probability": round(prob, 4),
        "predicted_churn":   pred,
        "risk_level":        risk_label(prob),
        "top_risk_factors":  _get_shap_factors(model, X, feature_cols, top_n=5),
    }

    return result


# ─── SHAP factors (optional) ────────────────────────────────────────────────────
def _get_shap_factors(model, X: pd.DataFrame, feature_cols: list,
                       top_n: int = 5) -> list:
    """
    Return the top-N features driving a single prediction using SHAP values.
    Returns [] if shap is not installed.
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X)
        # shap_vals shape: (1, n_features)
        abs_shap = np.abs(shap_vals[0])
        top_idx  = abs_shap.argsort()[::-1][:top_n]
        factors  = [
            {
                "feature": feature_cols[i],
                "shap_value": round(float(shap_vals[0][i]), 4),
            }
            for i in top_idx
        ]
        return factors
    except ImportError:
        return []


# ─── Predict Batch ──────────────────────────────────────────────────────────────
def predict_batch(model, scaler, feature_cols: list,
                  df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Score a DataFrame of customers.

    Parameters
    ----------
    model        : fitted XGBClassifier
    scaler       : fitted StandardScaler
    feature_cols : list[str]
    df_raw       : pd.DataFrame — one row per customer, columns = feature names

    Returns
    -------
    pd.DataFrame — original columns + ChurnProbability, PredictedChurn, RiskLevel
    """
    X = df_raw.copy()

    # Align columns
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale
    scale_present = [c for c in SCALE_COLS if c in X.columns]
    if scale_present:
        X[scale_present] = scaler.transform(X[scale_present])

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_out = df_raw.copy()
    df_out["ChurnProbability"] = probs.round(4)
    df_out["PredictedChurn"]   = preds
    df_out["RiskLevel"]        = [risk_label(p) for p in probs]

    print(f"[predict_batch] Scored {len(df_out):,} customers")
    print(f"[predict_batch] Risk distribution:\n{df_out['RiskLevel'].value_counts().to_string()}")
    return df_out


# ─── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Churn Prediction — Inference Script"
    )
    parser.add_argument(
        "--customer", type=str, default=None,
        help="JSON string of a single customer's features"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input CSV for batch scoring"
    )
    parser.add_argument(
        "--output", type=str, default="data/churn_predictions.csv",
        help="Path to save batch predictions (default: data/churn_predictions.csv)"
    )

    args = parser.parse_args()

    model, scaler, feature_cols = load_pipeline()

    if args.customer:
        customer_data = json.loads(args.customer)
        result = predict_single(model, scaler, feature_cols, customer_data)
        print("\n── Single Customer Prediction ──")
        print(json.dumps(result, indent=2))

    elif args.input:
        df_raw = pd.read_csv(args.input)
        df_out = predict_batch(model, scaler, feature_cols, df_raw)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df_out.to_csv(args.output, index=False)
        print(f"\n[main] Predictions saved to: {args.output}")

    else:
        print("No input provided. Use --customer or --input. See --help.")


if __name__ == "__main__":
    main()