"""
scripts/calibrate_model.py

Run this once after any retraining to regenerate isotonic_calibrator.pkl and psi_baseline.json.
Usage: python scripts/calibrate_model.py
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
DATA_PATH = "data/processed/telco_final_processed.csv"
MODEL_PATH = "models/xgb_tuned_model.pkl"
OUT_CAL = "models/isotonic_calibrator.pkl"
OUT_PSI = "models/psi_baseline.json"

df = pd.read_csv(DATA_PATH)
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Reproduce original split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Calibration holdout from train
X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

scaler = StandardScaler()
X_tr_s = X_tr.copy()
X_cal_s = X_cal.copy()
X_tr_s[NUMERIC_COLS] = scaler.fit_transform(X_tr[NUMERIC_COLS])
X_cal_s[NUMERIC_COLS] = scaler.transform(X_cal[NUMERIC_COLS])

model = joblib.load(MODEL_PATH)

# Fit isotonic calibrator on calibration holdout
raw_probs = model.predict_proba(X_cal_s)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_probs, y_cal)
joblib.dump(iso, OUT_CAL)
print(f"Saved calibrator → {OUT_CAL}")

# PSI baseline from full training set
psi_baseline = {}
for col in NUMERIC_COLS:
    vals = X_train[col]
    bins = np.percentile(vals, np.linspace(0, 100, 11))
    bins[0] -= 0.001
    bins[-1] += 0.001
    counts, _ = np.histogram(vals, bins=bins)
    psi_baseline[col] = {
        "bin_edges": bins.tolist(),
        "bin_pcts": (counts / counts.sum()).tolist(),
    }

with open(OUT_PSI, "w") as f:
    json.dump(psi_baseline, f, indent=2)
print(f"Saved PSI baseline → {OUT_PSI}")

# Quick validation: Brier score improvement
from sklearn.metrics import brier_score_loss
X_test_s = X_test.copy()
X_test_s[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])
raw_test = model.predict_proba(X_test_s)[:, 1]
cal_test = iso.predict(raw_test)
print(f"Brier (raw):        {brier_score_loss(y_test, raw_test):.4f}")
print(f"Brier (calibrated): {brier_score_loss(y_test, cal_test):.4f}")
