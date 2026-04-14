"""
src/preprocess.py
=================
Reusable data cleaning functions for the IBM Telco Customer Churn dataset.
All functions are designed to be called from notebooks OR from train.py.

Usage:
    from src.preprocess import load_data, clean_data
    df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_clean = clean_data(df)
"""

import pandas as pd
import numpy as np


# ─── Column groups ─────────────────────────────────────────────────────────────
CATEGORICAL_BINARY = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "PaperlessBilling", "Churn",
]

CATEGORICAL_MULTI = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

TARGET_COL = "Churn"
ID_COL = "customerID"


# ─── Load ───────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Path to WA_Fn-UseC_-Telco-Customer-Churn.csv

    Returns
    -------
    pd.DataFrame
        Raw dataframe with 7043 rows × 21 columns (approximately).
    """
    df = pd.read_csv(filepath)
    print(f"[load_data] Shape: {df.shape}")
    print(f"[load_data] Columns: {df.columns.tolist()}")
    return df


# ─── Inspect ────────────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print a quick data quality summary."""
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Duplicates     : {df.duplicated().sum()}")
    print(f"\nMissing values :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nData types     :\n{df.dtypes}")
    print("=" * 60)


# ─── Fix TotalCharges ───────────────────────────────────────────────────────────
def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is stored as object dtype.
    11 rows contain whitespace strings (' ') — these are new customers with
    tenure=0, so we impute TotalCharges = 0.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame with TotalCharges as float64
    """
    df = df.copy()

    # Replace blank strings with NaN, then cast
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Impute: tenure=0 → TotalCharges=0
    mask = df["TotalCharges"].isna()
    print(f"[fix_total_charges] {mask.sum()} blank TotalCharges rows imputed with 0")
    df.loc[mask, "TotalCharges"] = 0.0

    return df


# ─── Encode Target ──────────────────────────────────────────────────────────────
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Churn: 'Yes' → 1, 'No' → 0.
    """
    df = df.copy()
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    print(f"[encode_target] Churn distribution:\n{df[TARGET_COL].value_counts(normalize=True).round(3)}")
    return df


# ─── Drop Unnecessary Columns ───────────────────────────────────────────────────
def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Drop customerID — it's a unique identifier, not a feature."""
    df = df.copy()
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
        print(f"[drop_id_column] Dropped '{ID_COL}'")
    return df


# ─── Encode Binary Categoricals ────────────────────────────────────────────────
def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Yes/No binary columns as 1/0.
    SeniorCitizen is already 0/1 in the raw data.
    gender: Male → 1, Female → 0.
    """
    df = df.copy()

    yes_no_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling",
    ]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # gender
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    print(f"[encode_binary_columns] Encoded: {yes_no_cols + ['gender']}")
    return df


# ─── One-Hot Encode Multi-Category Columns ──────────────────────────────────────
def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode multi-category columns with drop_first=False to allow
    explicit interpretation of all categories via SHAP values (e.g. explicitly
    modeling the impact of Month-to-month contracts vs missing them).

    Columns encoded:
        MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        Contract, PaymentMethod
    """
    df = df.copy()
    cols_to_encode = [c for c in CATEGORICAL_MULTI if c in df.columns]

    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
    print(f"[encode_categorical_columns] One-hot encoded {len(cols_to_encode)} columns")
    print(f"[encode_categorical_columns] New shape: {df.shape}")
    return df


# ─── Full Clean Pipeline ────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """
    Run the full data cleaning pipeline in order:
        1. Fix TotalCharges dtype
        2. Encode Churn target (Yes/No → 1/0)
        3. Drop customerID
        4. Encode binary columns
        5. One-hot encode multi-category columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from load_data().
    drop_id : bool
        Whether to drop the customerID column (default True).

    Returns
    -------
    pd.DataFrame
        Fully cleaned and encoded dataframe, ready for feature engineering.
    """
    print("\n[clean_data] Starting cleaning pipeline...")
    df = fix_total_charges(df)
    df = encode_target(df)
    if drop_id:
        df = drop_id_column(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    print(f"[clean_data] Done. Final shape: {df.shape}\n")
    return df


# ─── Save Cleaned Data ──────────────────────────────────────────────────────────
def save_clean_data(df: pd.DataFrame, filepath: str = "data/churn_clean.csv") -> None:
    """Save cleaned dataframe to CSV."""
    df.to_csv(filepath, index=False)
    print(f"[save_clean_data] Saved to {filepath}")