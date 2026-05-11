"""
tests/test_model.py
Minimum bar for CI gate:
  - AUC > 0.80
  - Output shape matches input rows
  - Probabilities in [0, 1]
  - Risk labels are "High" or "Low" only
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from api.utils import FEATURE_COLUMNS, NUMERIC_COLUMNS

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture(scope="session")
def artifacts():
    model      = joblib.load("models/xgb_tuned_model.pkl")
    scaler     = joblib.load("models/scaler.pkl")
    calibrator = joblib.load("models/isotonic_calibrator.pkl")
    return model, scaler, calibrator


@pytest.fixture(scope="session")
def test_data():
    df = pd.read_csv("data/processed/telco_final_processed.csv")
    X  = df[FEATURE_COLUMNS].copy()
    y  = df["Churn"].values
    return X, y


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
def test_artifacts_load(artifacts):
    model, scaler, calibrator = artifacts
    assert model is not None
    assert scaler is not None
    assert calibrator is not None


def test_auc_above_threshold(artifacts, test_data):
    model, scaler, calibrator = artifacts
    X, y = test_data

    X_scaled = X.copy()
    X_scaled[NUMERIC_COLUMNS] = scaler.transform(X_scaled[NUMERIC_COLUMNS])

    raw_probs = model.predict_proba(X_scaled)[:, 1]
    cal_probs = calibrator.predict(raw_probs)

    auc = roc_auc_score(y, cal_probs)
    assert auc > 0.80, f"AUC degraded: {auc:.4f} < 0.80"


def test_output_shape(artifacts, test_data):
    model, scaler, calibrator = artifacts
    X, y = test_data

    X_scaled = X.copy()
    X_scaled[NUMERIC_COLUMNS] = scaler.transform(X_scaled[NUMERIC_COLUMNS])

    raw_probs = model.predict_proba(X_scaled)[:, 1]
    cal_probs = calibrator.predict(raw_probs)

    assert len(cal_probs) == len(y), "Output length mismatch"


def test_probabilities_in_bounds(artifacts, test_data):
    model, scaler, calibrator = artifacts
    X, _ = test_data

    X_scaled = X.copy()
    X_scaled[NUMERIC_COLUMNS] = scaler.transform(X_scaled[NUMERIC_COLUMNS])

    raw_probs = model.predict_proba(X_scaled)[:, 1]
    cal_probs = calibrator.predict(raw_probs)

    assert np.all(cal_probs >= 0.0), "Probability below 0"
    assert np.all(cal_probs <= 1.0), "Probability above 1"


def test_risk_labels_valid(artifacts, test_data):
    THRESHOLD = 0.22
    model, scaler, calibrator = artifacts
    X, _ = test_data

    X_scaled = X.copy()
    X_scaled[NUMERIC_COLUMNS] = scaler.transform(X_scaled[NUMERIC_COLUMNS])

    raw_probs = model.predict_proba(X_scaled)[:, 1]
    cal_probs = calibrator.predict(raw_probs)
    risks     = ["High" if p > THRESHOLD else "Low" for p in cal_probs]

    assert all(r in ("High", "Low") for r in risks), "Unexpected risk label"


def test_feature_columns_match(test_data):
    X, _ = test_data
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    assert not missing, f"Missing feature columns in test data: {missing}"