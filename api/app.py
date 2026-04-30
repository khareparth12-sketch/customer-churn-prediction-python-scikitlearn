# api/app.py
from fastapi import FastAPI, Request
import joblib
import pandas as pd
from api.utils import FEATURE_COLUMNS, NUMERIC_COLUMNS
from api.schema import CustomerData
from api.cache import make_cache_key, get_cached, set_cached
from api.logger import get_logger
import shap
import uuid
import time

logger = get_logger("churn_api")

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("models/xgb_tuned_model.pkl")
scaler = joblib.load("models/scaler.pkl")

background_df = pd.read_csv("data/processed/telco_final_processed.csv")
background_data = background_df[FEATURE_COLUMNS].sample(50, random_state=42)
background_data[NUMERIC_COLUMNS] = scaler.transform(background_data[NUMERIC_COLUMNS])

shap_explainer = shap.Explainer(model.predict_proba, background_data)

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

    prob = model.predict_proba(df)[0][1]
    risk = "High" if prob > 0.35 else "Low"

    result = {
        "churn_probability": float(prob),
        "churn_risk": risk,
    }

    await set_cached(cache_key, result)
    result["x_cache"] = "MISS"

    logger.info("", extra={
        "request_id": request_id,
        "event": "predict",
        "x_cache": "MISS",
        "churn_probability": float(prob),
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

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost Churn Classifier",
        "version": "1.0",
        "threshold": 0.35,
        "features": len(FEATURE_COLUMNS)
    }