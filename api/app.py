#api\app.py
from fastapi import FastAPI, Request
import joblib
import pandas as pd
from api.utils import FEATURE_COLUMNS, NUMERIC_COLUMNS
from api.schema import CustomerData
import shap
from api.cache import make_cache_key, get_cached, set_cached

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("models/xgb_tuned_model.pkl")
scaler = joblib.load("models/scaler.pkl")

background_df = pd.read_csv(
    "data/processed/telco_final_processed.csv"
)

background_data = background_df[FEATURE_COLUMNS].sample(
    50,
    random_state=42
)

background_data[NUMERIC_COLUMNS] = scaler.transform(
    background_data[NUMERIC_COLUMNS]
)

shap_explainer = shap.Explainer(
    model.predict_proba,
    background_data
)

#shap_explainer = None

@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}

# AFTER
@app.post("/predict")
async def predict(data: CustomerData):
    input_dict = data.dict()
    cache_key = make_cache_key(input_dict)

    # Cache HIT
    cached = await get_cached(cache_key)
    if cached:
        cached["x_cache"] = "HIT"
        return cached

    # Cache MISS — run inference
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
    return result

@app.post("/explain")
async def explain(data: CustomerData):

    try:
        input_dict = data.dict()

        df = pd.DataFrame([input_dict])

        df = df.reindex(
            columns=FEATURE_COLUMNS,
            fill_value=0
        )

        df[NUMERIC_COLUMNS] = scaler.transform(
            df[NUMERIC_COLUMNS]
        )

        sv = shap_explainer(df)

        return {
            "base_value": float(sv.base_values[0,1]),

            "shap_values": sv.values[0,:,1].tolist(),

            "feature_names":
                FEATURE_COLUMNS,

            "feature_values":
                input_dict
        }

    except Exception as e:
        return {
            "error": str(e)
        }

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