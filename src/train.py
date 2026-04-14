"""
src/train.py
============
Full training pipeline: load → clean → engineer → split → scale → train → save.

Run as a script:
    python src/train.py

Or import into a notebook:
    from src.train import run_training_pipeline
    results = run_training_pipeline()
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    recall_score,
)
from xgboost import XGBClassifier

# Local modules
from src.preprocess import load_data, clean_data
from src.features import engineer_features, get_feature_matrix

warnings.filterwarnings("ignore")

# ─── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR  = "models"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# Columns to scale (continuous numeric)
SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
              "charge_per_tenure", "charges_ratio"]


# ─── Helpers ────────────────────────────────────────────────────────────────────
def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


def print_section(title: str):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ─── Data Preparation ───────────────────────────────────────────────────────────
def prepare_data(data_path: str = DATA_PATH):
    """
    Load, clean, and engineer features.
    Returns X_train, X_test, y_train, y_test, scaler.
    """
    print_section("STEP 1: Data Preparation")

    # Load & clean
    df_raw   = load_data(data_path)
    df_clean = clean_data(df_raw)

    # Feature engineering
    df_feat = engineer_features(df_clean)

    # Separate features and target
    X, y = get_feature_matrix(df_feat)

    # Train / test split — stratified to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain size : {X_train.shape[0]} rows")
    print(f"Test size  : {X_test.shape[0]} rows")
    print(f"Features   : {X_train.shape[1]}")
    print(f"Class balance (train): {y_train.value_counts(normalize=True).round(3).to_dict()}")

    # Scale — fit ONLY on train, transform both
    scaler = StandardScaler()
    scale_cols_present = [c for c in SCALE_COLS if c in X_train.columns]

    X_train[scale_cols_present] = scaler.fit_transform(X_train[scale_cols_present])
    X_test[scale_cols_present]  = scaler.transform(X_test[scale_cols_present])

    print(f"\nScaling applied to: {scale_cols_present}")

    return X_train, X_test, y_train, y_test, scaler


# ─── Model 1: Logistic Regression Baseline ──────────────────────────────────────
def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression baseline with class_weight='balanced'.
    Run 5-fold stratified cross-validation and report AUC mean ± std.
    """
    print_section("STEP 2a: Logistic Regression Baseline")

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    lr.fit(X_train, y_train)
    return lr


# ─── Model 2: Random Forest with GridSearch ─────────────────────────────────────
def train_random_forest(X_train, y_train):
    """
    Train Random Forest with GridSearchCV.
    Optimises for roc_auc with stratified k-fold.
    """
    print_section("STEP 2b: Random Forest + GridSearchCV")

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [5, 10, None],
        "min_samples_split": [2, 5],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        rf, param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best params : {grid_search.best_params_}")
    print(f"Best CV AUC : {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


# ─── Model 3: XGBoost (Best Model) ─────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test):
    """
    Train XGBoost with:
    - scale_pos_weight to handle class imbalance natively
    - early stopping to prevent overfitting
    - 5-fold stratified cross-validation
    """
    print_section("STEP 2c: XGBoost — Best Model")

    # scale_pos_weight = negative / positive count ratio
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.3f}  ({neg} neg / {pos} pos)")

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0,
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print(f"Best iteration : {xgb.best_iteration}")

    # Cross-validation
    cv_xgb = XGBClassifier(
        n_estimators=xgb.best_iteration,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(cv_xgb, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return xgb


# ─── Compare Models ─────────────────────────────────────────────────────────────
def compare_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Build a comparison table: AUC, Recall, Precision, F1 for each model.

    Parameters
    ----------
    models : dict  {name: fitted_model}
    X_test, y_test : test set

    Returns
    -------
    pd.DataFrame
    """
    print_section("MODEL COMPARISON")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)["1"]
        results[name] = {
            "AUC":       round(roc_auc_score(y_test, y_prob), 4),
            "Recall":    round(report["recall"], 4),
            "Precision": round(report["precision"], 4),
            "F1":        round(report["f1-score"], 4),
        }

    df_results = pd.DataFrame(results).T.sort_values("AUC", ascending=False)
    print(df_results.to_string())
    df_results.to_csv("outputs/model_comparison.csv")
    print("\n[compare_models] Saved to outputs/model_comparison.csv")
    return df_results


# ─── Save Models ────────────────────────────────────────────────────────────────
def save_models(models: dict, scaler: StandardScaler):
    """
    Save all models and scaler using joblib.

    Files saved:
        models/xgb_model.pkl
        models/rf_model.pkl
        models/lr_model.pkl
        models/scaler.pkl
    """
    print_section("SAVING MODELS")

    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"Saved: {path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved: {scaler_path}")


# ─── Full Pipeline ───────────────────────────────────────────────────────────────
def run_training_pipeline(data_path: str = DATA_PATH):
    """
    End-to-end training pipeline.
    Returns dict of fitted models, scaler, test set, and comparison table.
    """
    ensure_dirs()

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(data_path)

    # Train models
    lr  = train_logistic_regression(X_train, y_train)
    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, X_test, y_train, y_test)

    models = {
        "lr_model":  lr,
        "rf_model":  rf,
        "xgb_model": xgb,
    }

    # Compare
    comparison = compare_models(models, X_test, y_test)

    # Save
    save_models(models, scaler)

    print_section("TRAINING COMPLETE ✓")
    print("Best model: XGBoost (xgb_model.pkl)")
    print("Use src/predict.py or api/app.py for inference.\n")

    return {
        "models":     models,
        "scaler":     scaler,
        "X_test":     X_test,
        "y_test":     y_test,
        "comparison": comparison,
    }


# ─── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_training_pipeline()