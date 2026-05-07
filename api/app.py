# api/app.py
from fastapi import FastAPI, Request
import joblib
import pandas as pd
import numpy as np
import json
from api.utils import FEATURE_COLUMNS, NUMERIC_COLUMNS
from api.schema import CustomerData
from api.cache import make_cache_key, get_cached, set_cached
from api.logger import get_logger
import shap
import uuid
import time

logger = get_logger("churn_api")

app = FastAPI(title="Customer Churn Prediction API")

# --- Model artifacts ---
model = joblib.load("models/xgb_tuned_model.pkl")
scaler = joblib.load("models/scaler.pkl")
calibrator = joblib.load("models/isotonic_calibrator.pkl")   # NEW

# --- PSI baseline ---
with open("models/psi_baseline.json") as f:
    PSI_BASELINE = json.load(f)

# --- SHAP ---
background_df = pd.read_csv("data/processed/telco_final_processed.csv")
background_data = background_df[FEATURE_COLUMNS].sample(50, random_state=42)
background_data[NUMERIC_COLUMNS] = scaler.transform(background_data[NUMERIC_COLUMNS])
shap_explainer = shap.Explainer(model.predict_proba, background_data)

# Optimal cost-based threshold (FN=10, FP=1 sweep → 0.22 post-calibration)
THRESHOLD = 0.22


def _compute_psi(expected_pcts, observed_pcts):
    """PSI = sum((O - E) * ln(O/E))"""
    eps = 1e-6
    psi = 0.0
    for e, o in zip(expected_pcts, observed_pcts):
        o = max(o, eps)
        e = max(e, eps)
        psi += (o - e) * np.log(o / e)
    return psi


@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}


@app.post("/predict")
async def predict(data: CustomerData):
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    input_dict = data.dict()
    cache_key = make_cache_key(input_dict)

    cached = await get_cached(cache_key)
    if cached:
        cached["x_cache"] = "HIT"
        logger.info("", extra={
            "request_id": request_id,
            "event": "predict",
            "x_cache": "HIT",
            "churn_probability": cached["churn_probability"],
            "risk_flag": cached["churn_risk"],
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
        return cached

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

    raw_prob = model.predict_proba(df)[0][1]
    prob = float(calibrator.predict([raw_prob])[0])       # calibrated
    risk = "High" if prob > THRESHOLD else "Low"

    result = {
        "churn_probability": round(prob, 4),
        "churn_risk": risk,
    }

    await set_cached(cache_key, result)
    result["x_cache"] = "MISS"

    logger.info("", extra={
        "request_id": request_id,
        "event": "predict",
        "x_cache": "MISS",
        "churn_probability": round(prob, 4),
        "risk_flag": risk,
        "tenure": input_dict["tenure"],
        "MonthlyCharges": input_dict["MonthlyCharges"],
        "TotalCharges": input_dict["TotalCharges"],
        "latency_ms": round((time.time() - start) * 1000, 2)
    })

    return result


@app.post("/explain")
async def explain(data: CustomerData):
    request_id = str(uuid.uuid4())[:8]
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

        sv = shap_explainer(df)

        logger.info("", extra={
            "request_id": request_id,
            "event": "explain",
            "status": "ok"
        })

        return {
            "base_value": float(sv.base_values[0, 1]),
            "shap_values": sv.values[0, :, 1].tolist(),
            "feature_names": FEATURE_COLUMNS,
            "feature_values": input_dict
        }

    except Exception as e:
        logger.error("", extra={
            "request_id": request_id,
            "event": "explain",
            "status": "error",
            "error": str(e)
        })
        return {"error": str(e)}


@app.get("/drift")
async def drift(data: list = None):
    """
    Compute PSI for tenure, MonthlyCharges, TotalCharges against training baseline.
    Pass recent inference data as JSON body, or this endpoint reads from logs in production.
    For now, accepts query param ?window=N to simulate — replace with real log replay later.

    Returns per-feature PSI and overall drift status.
    PSI < 0.10 → stable | 0.10–0.20 → moderate | > 0.20 → significant drift
    """
    # In production: pull recent feature values from structured logs.
    # Here we expose the endpoint stub with the computation ready to wire in.
    return {
        "message": "Wire recent_data into this endpoint from log replay.",
        "instructions": "POST a JSON list of dicts with keys: tenure, MonthlyCharges, TotalCharges",
        "psi_thresholds": {"stable": "<0.10", "moderate": "0.10-0.20", "significant": ">0.20"}
    }


@app.post("/drift")
async def drift_check(request: Request):
    """
    POST body: list of recent inference records (dicts with numeric feature keys).
    Example: [{"tenure": 5, "MonthlyCharges": 70.0, "TotalCharges": 350.0}, ...]
    """
    try:
        body = await request.json()
        recent_df = pd.DataFrame(body)
    except Exception as e:
        return {"error": f"Invalid body: {e}"}

    results = {}
    alert = False

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in recent_df.columns:
            results[col] = {"error": "column missing"}
            continue

        baseline = PSI_BASELINE[col]
        bin_edges = baseline["bin_edges"]
        expected_pcts = baseline["bin_pcts"]

        obs_counts, _ = np.histogram(recent_df[col].dropna(), bins=bin_edges)
        n = obs_counts.sum()
        if n == 0:
            results[col] = {"error": "no data"}
            continue
        observed_pcts = (obs_counts / n).tolist()

        psi_val = _compute_psi(expected_pcts, observed_pcts)
        status = "stable" if psi_val < 0.10 else ("moderate" if psi_val < 0.20 else "significant")
        if status == "significant":
            alert = True

        results[col] = {
            "psi": round(psi_val, 4),
            "status": status,
            "n_records": int(n)
        }

    return {
        "drift_alert": alert,
        "retraining_recommended": alert,
        "features": results
    }


@app.get("/health")
async def health():
    if model is None or scaler is None or calibrator is None:
        return {"status": "degraded", "detail": "model artifacts missing"}
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost Churn Classifier",
        "version": "1.1",
        "calibration": "isotonic",
        "threshold": THRESHOLD,
        "threshold_basis": "cost-optimized (FN=10, FP=1)",
        "features": len(FEATURE_COLUMNS)
    }