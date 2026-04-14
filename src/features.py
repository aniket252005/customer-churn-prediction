"""
src/features.py
===============
Feature engineering functions for the Churn Prediction project.
All functions take a DataFrame and return a DataFrame with new columns added.
Always call AFTER preprocess.clean_data().

Usage:
    from src.features import engineer_features
    df_feat = engineer_features(df_clean)
"""

import pandas as pd
import numpy as np


# ─── 1. Tenure Segmentation ─────────────────────────────────────────────────────
def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Segment customers by tenure in months.
    Groups:
        0–12  → New Customer       (highest churn risk)
        13–24 → Early Customer
        25–48 → Growing Customer
        49–60 → Mature Customer
        61+   → Loyal Customer     (lowest churn risk)

    Creates: tenure_group (object), tenure_group_num (int, ordinal)
    """
    df = df.copy()
    bins   = [0, 12, 24, 48, 60, 72]
    labels = [0, 1, 2, 3, 4]          # ordinal for ML
    label_names = ["New", "Early", "Growing", "Mature", "Loyal"]

    df["tenure_group_num"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    ).astype(int)

    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=label_names, include_lowest=True
    ).astype(str)

    print(f"[add_tenure_group] Distribution:\n{df['tenure_group'].value_counts()}")
    return df


# ─── 2. Service Count ───────────────────────────────────────────────────────────
def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count total number of add-on services per customer.
    Higher service count → more invested → less likely to churn.

    Services counted (binary 1/0 in cleaned data):
        PhoneService, MultipleLines_Yes, OnlineSecurity_Yes,
        OnlineBackup_Yes, DeviceProtection_Yes, TechSupport_Yes,
        StreamingTV_Yes, StreamingMovies_Yes

    Creates: service_count (int, range 0–8)
    """
    df = df.copy()
    service_cols = [
        c for c in [
            "PhoneService",
            "MultipleLines_Yes",
            "OnlineSecurity_Yes",
            "OnlineBackup_Yes",
            "DeviceProtection_Yes",
            "TechSupport_Yes",
            "StreamingTV_Yes",
            "StreamingMovies_Yes",
        ]
        if c in df.columns
    ]
    df["service_count"] = df[service_cols].sum(axis=1)
    print(f"[add_service_count] Range: {df['service_count'].min()}–{df['service_count'].max()}")
    return df


# ─── 3. High Charges Risk Flag ──────────────────────────────────────────────────
def add_high_charges_flag(df: pd.DataFrame, threshold: float = 65.0) -> pd.DataFrame:
    """
    Flag customers paying high monthly charges without add-on protection.
    Based on Finding 03: customers paying >$65/month without security/tech
    support have a 58% churn probability.

    Creates: high_charges_no_support (binary 0/1)
    """
    df = df.copy()

    has_support = pd.Series(False, index=df.index)
    for col in ["OnlineSecurity_Yes", "TechSupport_Yes"]:
        if col in df.columns:
            has_support = has_support | (df[col] == 1)

    df["high_charges_no_support"] = (
        (df["MonthlyCharges"] > threshold) & (~has_support)
    ).astype(int)

    pct = df["high_charges_no_support"].mean() * 100
    print(f"[add_high_charges_flag] {pct:.1f}% of customers flagged as high-charges/no-support")
    return df


# ─── 4. Charge-to-Tenure Ratio ──────────────────────────────────────────────────
def add_charge_per_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly charges per month of tenure.
    New customers with high charges are especially at risk.

    Creates: charge_per_tenure (float)
    """
    df = df.copy()
    # Avoid division by zero for tenure = 0
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    print(f"[add_charge_per_tenure] Mean: {df['charge_per_tenure'].mean():.2f}")
    return df


# ─── 5. TotalCharges vs Expected ────────────────────────────────────────────────
def add_charges_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ratio of actual TotalCharges to expected (MonthlyCharges × tenure).
    Ratio < 1 may indicate discounts or billing issues.
    Ratio > 1 may indicate one-time fees or overages.

    Creates: charges_ratio (float)
    """
    df = df.copy()
    expected = df["MonthlyCharges"] * df["tenure"]
    df["charges_ratio"] = df["TotalCharges"] / (expected + 1)  # +1 avoids /0
    print(f"[add_charges_ratio] Mean: {df['charges_ratio'].mean():.3f}")
    return df


# ─── 7. Auto Payment Flag ───────────────────────────────────────────────────────
def add_auto_payment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finding 05: electronic check customers churn at 45% vs 15-18% for
    automatic payment methods. Flag customers NOT using auto-pay.

    Creates: non_auto_payment (binary 0/1)
    """
    df = df.copy()
    candidate_cols = [
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Bank transfer (automatic)",
    ]
    auto_cols = [c for c in candidate_cols if c in df.columns]
    
    if auto_cols:
        df["non_auto_payment"] = (df[auto_cols].sum(axis=1) == 0).astype(int)
    else:
        df["non_auto_payment"] = 0
    print(f"[add_auto_payment_flag] Non-auto payment share: {df['non_auto_payment'].mean():.1%}")
    return df


# ─── 8. Senior + No Partner Flag ────────────────────────────────────────────────
def add_senior_alone_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Senior customers with no partner/dependents have fewer support systems
    and may be more price-sensitive.

    Creates: senior_alone (binary 0/1)
    """
    df = df.copy()
    if "SeniorCitizen" in df.columns and "Partner" in df.columns:
        df["senior_alone"] = (
            (df["SeniorCitizen"] == 1) & (df["Partner"] == 0)
        ).astype(int)
    else:
        df["senior_alone"] = 0
    return df


# ─── Full Feature Engineering Pipeline ─────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from preprocess.clean_data().

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features added.

    New columns added:
        tenure_group_num, tenure_group, service_count,
        high_charges_no_support, charge_per_tenure,
        charges_ratio, non_auto_payment, senior_alone
    """
    print("\n[engineer_features] Starting feature engineering...")
    df = add_tenure_group(df)
    df = add_service_count(df)
    df = add_high_charges_flag(df)
    df = add_charge_per_tenure(df)
    df = add_charges_ratio(df)
    df = add_auto_payment_flag(df)
    df = add_senior_alone_flag(df)
    print(f"[engineer_features] Done. Final shape: {df.shape}\n")
    return df


# ─── Get Feature Matrix ─────────────────────────────────────────────────────────
def get_feature_matrix(df: pd.DataFrame, target: str = "Churn"):
    """
    Separate features (X) from target (y).
    Drops string/object columns (e.g. tenure_group label).

    Parameters
    ----------
    df : pd.DataFrame
    target : str

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    # Drop target and any remaining string columns
    drop_cols = [target] + [c for c in df.select_dtypes("object").columns]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target]
    print(f"[get_feature_matrix] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[get_feature_matrix] Features: {X.columns.tolist()}")
    return X, y