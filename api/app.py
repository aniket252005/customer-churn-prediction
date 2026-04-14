"""
api/app.py
==========
Flask REST API for real-time customer churn scoring.
Now configured for standard Flask layouts (templates/static) and Render deployment.
"""

import os
import sys
import json
import logging
import time
from functools import wraps

# Ensure project root is in path so src.* imports work
# In production on Render, the root will be the current directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd

from src.predict import load_pipeline, predict_single, predict_batch

# ─── App Setup ──────────────────────────────────────────────────────────────────
# Ensure paths are absolute for serverless compatibility
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# ─── CORS (allow dashboard to call the API) ─────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Load Model at Startup ──────────────────────────────────────────────────────
MODEL  = None
SCALER = None
FEATURE_COLS = None
MODEL_LOADED_AT = None

def load_model_once():
    global MODEL, SCALER, FEATURE_COLS, MODEL_LOADED_AT
    try:
        MODEL, SCALER, FEATURE_COLS = load_pipeline()
        MODEL_LOADED_AT = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        logger.info(f"✓ Model loaded successfully. Features: {len(FEATURE_COLS)}")
    except FileNotFoundError as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("Run `python src/train.py` first to train and save the model.")

# Call immediately when module is loaded by Gunicorn/Flask
load_model_once()

# ─── Decorators ─────────────────────────────────────────────────────────────────
def require_model(f):
    """Return 503 if model hasn't been loaded."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if MODEL is None:
            return jsonify({
                "error": "Model not loaded. Run src/train.py first.",
                "status": "unavailable",
            }), 503
        return f(*args, **kwargs)
    return wrapper

def require_json(f):
    """Return 400 if Content-Type is not application/json."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
            }), 400
        return f(*args, **kwargs)
    return wrapper

# ─── Input Validation ───────────────────────────────────────────────────────────
REQUIRED_FIELDS = {
    "tenure":         (int, float),
    "MonthlyCharges": (int, float),
}

def validate_customer(data: dict) -> tuple[bool, str]:
    """Validate required fields and types."""
    for field, types in REQUIRED_FIELDS.items():
        if field not in data:
            return False, f"Missing required field: '{field}'"
        if not isinstance(data[field], types):
            return False, f"Field '{field}' must be numeric, got {type(data[field]).__name__}"
        if data[field] < 0:
            return False, f"Field '{field}' must be >= 0"
    return True, ""

# ─── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
@app.route("/dashboard/", methods=["GET"])
def index():
    """Serve the dashboard UI home page using Flask templates."""
    return render_template("index.html")

@app.route("/dashboard", methods=["GET"])
def dashboard_redirect():
    """Redirect /dashboard to /dashboard/ for consistency."""
    return redirect(url_for('index'))

@app.route("/api/health", methods=["GET"])
def api_health():
    """Service metadata for the frontend."""
    return jsonify({
        "service": "Customer Churn Prediction API",
        "status":  "running",
        "version": "1.1.0"
    }), 200

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check with model metadata."""
    return jsonify({
        "status":          "healthy" if MODEL is not None else "degraded",
        "model_loaded":    MODEL is not None,
        "model_loaded_at": MODEL_LOADED_AT,
        "n_features":      len(FEATURE_COLS) if FEATURE_COLS else 0,
        "model_type":      type(MODEL).__name__ if MODEL else None,
    }), 200 if MODEL is not None else 503

@app.route("/predict", methods=["POST"])
@require_model
@require_json
def predict():
    """Score a single customer."""
    data = request.get_json()
    is_valid, error_msg = validate_customer(data)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    try:
        t0 = time.time()
        result = predict_single(MODEL, SCALER, FEATURE_COLS, data)
        latency_ms = round((time.time() - t0) * 1000, 2)
        return jsonify({**result, "latency_ms": latency_ms}), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route("/predict/batch", methods=["POST"])
@require_model
@require_json
def predict_batch_endpoint():
    """Score multiple customers in a single request."""
    data = request.get_json()
    if not isinstance(data, list) or len(data) == 0:
        return jsonify({"error": "Request body must be a non-empty JSON array"}), 400

    try:
        t0 = time.time()
        df_raw = pd.DataFrame(data)
        df_out = predict_batch(MODEL, SCALER, FEATURE_COLS, df_raw)
        latency_ms = round((time.time() - t0) * 1000, 2)
        
        predictions = []
        for i, row in df_out.iterrows():
            predictions.append({
                "index": int(i),
                "churn_probability": float(row["ChurnProbability"]),
                "predicted_churn": int(row["PredictedChurn"]),
                "risk_level": row["RiskLevel"],
            })
        
        summary = df_out["RiskLevel"].value_counts().to_dict()
        return jsonify({
            "n_customers": len(data),
            "predictions": predictions,
            "summary": {
                "high_risk": summary.get("High", 0),
                "medium_risk": summary.get("Medium", 0),
                "low_risk": summary.get("Low", 0),
            },
            "latency_ms": latency_ms,
        }), 200
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

# ─── Error Handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

# ─── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Default port set to 5000 as requested for local dev/Render
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info(f"Starting Churn Prediction API on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)