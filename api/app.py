#api\app.py
from fastapi import FastAPI
import joblib
import pandas as pd
from api.utils import FEATURE_COLUMNS, NUMERIC_COLUMNS
from api.schema import CustomerData
import shap

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("models/xgb_tuned_model.pkl")
scaler = joblib.load("models/scaler.pkl")

shap_explainer = None

@app.get("/")
def home():
    return {"message": "Churn Prediction API running"}

@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()

    df = pd.DataFrame([input_dict])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

    prob = model.predict_proba(df)[0][1]

    risk = "High" if prob > 0.35 else "Low"

    booster = model.get_booster()

    explainer = shap.TreeExplainer(
        booster,
        model_output="probability"
    )

    sv = explainer(df)

    return {
        "churn_probability": float(prob),
        "churn_risk": risk,

        "shap_explanation": {
            "base_value": float(sv.base_values[0]),

            "shap_values": sv.values[0].tolist(),

            "feature_names": FEATURE_COLUMNS,

            "feature_values": input_dict
        }
    }

@app.get("/health")
def health_check():
    return {"status": "API running"}

@app.get("/model-info")
def model_info():
    return {
        "model": "XGBoost Churn Classifier",
        "version": "1.0",
        "threshold": 0.35,
        "features": len(FEATURE_COLUMNS)
    }