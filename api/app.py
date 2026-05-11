# api/app.py
from fastapi import FastAPI, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
import json
import io
from api.utils import FEATURE_COLUMNS, NUMERIC_COLUMNS
from api.schema import CustomerData
from api.cache import make_cache_key, get_cached, set_cached
from api.logger import get_logger
import shap
import uuid
import time

logger = get_logger("churn_api")

MODEL_VERSION = "1.1"

app = FastAPI(title="Customer Churn Prediction API", version=MODEL_VERSION)

# -----------------------------------------------------------------------
# Artifact loading — lazy with readiness flag
# -----------------------------------------------------------------------
model         = None
scaler        = None
calibrator    = None
shap_explainer = None
PSI_BASELINE  = None
_ready        = False
_ready_error  = None

def _load_artifacts():
    global model, scaler, calibrator, shap_explainer, PSI_BASELINE, _ready, _ready_error
    try:
        model      = joblib.load("models/xgb_tuned_model.pkl")
        scaler     = joblib.load("models/scaler.pkl")
        calibrator = joblib.load("models/isotonic_calibrator.pkl")

        with open("models/psi_baseline.json") as f:
            PSI_BASELINE = json.load(f)

        background_df   = pd.read_csv("data/processed/telco_final_processed.csv")
        background_data = background_df[FEATURE_COLUMNS].sample(50, random_state=42)
        background_data[NUMERIC_COLUMNS] = scaler.transform(background_data[NUMERIC_COLUMNS])
        shap_explainer  = shap.Explainer(model.predict_proba, background_data)

        _ready = True
        logger.info("", extra={"event": "artifacts_loaded", "status": "ok"})
    except Exception as e:
        _ready_error = str(e)
        logger.error("", extra={"event": "artifacts_load_failed", "error": str(e)})


@app.on_event("startup")
async def startup_event():
    _load_artifacts()


THRESHOLD = 0.22

# -----------------------------------------------------------------------
# Middleware — inject X-Model-Version on every response
# -----------------------------------------------------------------------
@app.middleware("http")
async def add_model_version_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Model-Version"] = MODEL_VERSION
    return response


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _compute_psi(expected_pcts, observed_pcts):
    eps = 1e-6
    return sum(
        (max(o, eps) - max(e, eps)) * np.log(max(o, eps) / max(e, eps))
        for e, o in zip(expected_pcts, observed_pcts)
    )


def _predict_df(df: pd.DataFrame):
    """Run inference on a pre-processed DataFrame. Returns prob + risk columns."""
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])
    raw_probs = model.predict_proba(df)[:, 1]
    cal_probs = calibrator.predict(raw_probs)
    risks     = ["High" if p > THRESHOLD else "Low" for p in cal_probs]
    return cal_probs, risks


def _run_batch(records: list[dict], job_id: str):
    """Background worker for large batch jobs — logs results."""
    logger.info("", extra={"event": "batch_start", "job_id": job_id, "n": len(records)})
    df = pd.DataFrame(records)
    probs, risks = _predict_df(df)
    logger.info("", extra={"event": "batch_done", "job_id": job_id, "n": len(records)})


# -----------------------------------------------------------------------
# Unversioned — discovery + health
# -----------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API", "version": MODEL_VERSION, "docs": "/docs"}


@app.get("/health")
async def health():
    """
    Liveness probe — returns 200 if process is alive.
    Does NOT check model readiness. Use /ready for that.
    """
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """
    Readiness probe — returns 200 only when all model artifacts are loaded.
    Docker / Render should use this before routing traffic.
    Returns 503 if still loading or if load failed.
    """
    if _ready:
        return {"status": "ready", "model_version": MODEL_VERSION}
    if _ready_error:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": _ready_error},
        )
    return JSONResponse(
        status_code=503,
        content={"status": "loading", "detail": "artifacts not yet loaded"},
    )


# -----------------------------------------------------------------------
# v1 routes
# -----------------------------------------------------------------------
@app.post("/v1/predict")
async def predict(data: CustomerData):
    if not _ready:
        return JSONResponse(status_code=503, content={"detail": "Service warming up, please wait…"})

    request_id = str(uuid.uuid4())[:8]
    start      = time.time()
    input_dict = data.dict()
    cache_key  = make_cache_key(input_dict)

    cached = await get_cached(cache_key)
    if cached:
        cached["x_cache"] = "HIT"
        logger.info("", extra={
            "request_id": request_id, "event": "predict",
            "x_cache": "HIT", "churn_probability": cached["churn_probability"],
            "risk_flag": cached["churn_risk"],
            "latency_ms": round((time.time() - start) * 1000, 2)
        })
        return cached

    df           = pd.DataFrame([input_dict])
    probs, risks = _predict_df(df)
    prob, risk   = float(probs[0]), risks[0]

    result = {"churn_probability": round(prob, 4), "churn_risk": risk}
    await set_cached(cache_key, result)
    result["x_cache"] = "MISS"

    logger.info("", extra={
        "request_id": request_id, "event": "predict", "x_cache": "MISS",
        "churn_probability": round(prob, 4), "risk_flag": risk,
        "tenure": input_dict["tenure"],
        "MonthlyCharges": input_dict["MonthlyCharges"],
        "TotalCharges": input_dict["TotalCharges"],
        "latency_ms": round((time.time() - start) * 1000, 2)
    })
    return result


@app.post("/v1/predict/batch")
async def predict_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not _ready:
        return JSONResponse(status_code=503, content={"detail": "Service warming up, please wait…"})

    request_id = str(uuid.uuid4())[:8]
    start      = time.time()

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": f"CSV parse failed: {e}"})

    if "customer_id" in df.columns:
        customer_ids = df["customer_id"].astype(str).tolist()
        df = df.drop(columns=["customer_id"])
    else:
        customer_ids = [str(i) for i in df.index]

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        return JSONResponse(
            status_code=422,
            content={"error": "Missing columns", "missing": missing}
        )

    n = len(df)

    if n > 1000:
        job_id  = str(uuid.uuid4())[:12]
        records = df[FEATURE_COLUMNS].to_dict(orient="records")
        background_tasks.add_task(_run_batch, records, job_id)
        logger.info("", extra={
            "request_id": request_id, "event": "batch_queued",
            "job_id": job_id, "n": n
        })
        return {
            "status": "queued",
            "job_id": job_id,
            "message": f"{n} rows queued for background processing.",
        }

    probs, risks = _predict_df(df[FEATURE_COLUMNS].copy())
    results = [
        {"customer_id": cid, "churn_probability": round(float(p), 4), "risk_flag": r}
        for cid, p, r in zip(customer_ids, probs, risks)
    ]

    logger.info("", extra={
        "request_id": request_id, "event": "batch_inline",
        "n": n, "latency_ms": round((time.time() - start) * 1000, 2)
    })
    return {"status": "ok", "n": n, "results": results}


@app.post("/v1/explain")
async def explain(data: CustomerData):
    if not _ready:
        return JSONResponse(status_code=503, content={"detail": "Service warming up, please wait…"})

    request_id = str(uuid.uuid4())[:8]
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

        sv = shap_explainer(df)

        logger.info("", extra={"request_id": request_id, "event": "explain", "status": "ok"})
        return {
            "base_value":     float(sv.base_values[0, 1]),
            "shap_values":    sv.values[0, :, 1].tolist(),
            "feature_names":  FEATURE_COLUMNS,
            "feature_values": input_dict,
        }
    except Exception as e:
        logger.error("", extra={"request_id": request_id, "event": "explain", "status": "error", "error": str(e)})
        return {"error": str(e)}


@app.get("/v1/drift")
async def drift_info():
    return {
        "message": "POST recent inference records to this endpoint.",
        "instructions": "Body: list of dicts with keys: tenure, MonthlyCharges, TotalCharges",
        "psi_thresholds": {"stable": "<0.10", "moderate": "0.10-0.20", "significant": ">0.20"},
    }


@app.post("/v1/drift")
async def drift_check(request: Request):
    if not _ready:
        return JSONResponse(status_code=503, content={"detail": "Service warming up, please wait…"})

    try:
        body      = await request.json()
        recent_df = pd.DataFrame(body)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": f"Invalid body: {e}"})

    results = {}
    alert   = False

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in recent_df.columns:
            results[col] = {"error": "column missing"}
            continue

        baseline      = PSI_BASELINE[col]
        bin_edges     = baseline["bin_edges"]
        expected_pcts = baseline["bin_pcts"]

        obs_counts, _ = np.histogram(recent_df[col].dropna(), bins=bin_edges)
        n = obs_counts.sum()
        if n == 0:
            results[col] = {"error": "no data"}
            continue

        observed_pcts = (obs_counts / n).tolist()
        psi_val       = _compute_psi(expected_pcts, observed_pcts)
        status        = "stable" if psi_val < 0.10 else ("moderate" if psi_val < 0.20 else "significant")
        if status == "significant":
            alert = True

        results[col] = {"psi": round(psi_val, 4), "status": status, "n_records": int(n)}

    return {"drift_alert": alert, "retraining_recommended": alert, "features": results}


@app.get("/v1/model-info")
def model_info():
    if not _ready:
        return JSONResponse(status_code=503, content={"detail": "Service warming up, please wait…"})
    return {
        "model":           "XGBoost Churn Classifier",
        "version":         MODEL_VERSION,
        "calibration":     "isotonic",
        "threshold":       THRESHOLD,
        "threshold_basis": "cost-optimized (FN=10, FP=1)",
        "features":        len(FEATURE_COLUMNS),
    }