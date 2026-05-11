"""
train.py — Churn model training with MLflow tracking
Run: python train.py
MLflow UI: mlflow ui --port 5000
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATA_PATH      = "data/processed/telco_final_processed.csv"
MODEL_OUT      = "models/xgb_tuned_model.pkl"
SCALER_OUT     = "models/scaler.pkl"
CALIBRATOR_OUT = "models/isotonic_calibrator.pkl"
PSI_OUT        = "models/psi_baseline.json"

THRESHOLD      = 0.22          # cost-optimized (FN=10, FP=1)
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
N_ITER         = 25
CV_FOLDS       = 5

NUMERIC_COLS   = ["tenure", "MonthlyCharges", "TotalCharges"]

PARAM_GRID = {
    "n_estimators":     [200, 300, 400],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def compute_psi_baseline(df: pd.DataFrame, cols: list, n_bins: int = 10) -> dict:
    """Compute PSI baseline buckets from training data."""
    baseline = {}
    for col in cols:
        counts, bin_edges = np.histogram(df[col].dropna(), bins=n_bins)
        pcts = (counts / counts.sum()).tolist()
        baseline[col] = {
            "bin_edges": bin_edges.tolist(),
            "bin_pcts":  pcts,
        }
    return baseline


def eval_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold":  threshold,
        "roc_auc":    round(roc_auc_score(y_true, y_prob), 4),
        "recall":     round(recall_score(y_true, y_pred), 4),
        "precision":  round(precision_score(y_true, y_pred), 4),
        "f1":         round(f1_score(y_true, y_pred), 4),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def train():
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run(run_name="xgb-tuned") as run:
        print(f"MLflow run: {run.info.run_id}")

        # ── 1. Load data ───────────────────────────────────────────
        df = pd.read_csv(DATA_PATH)
        X  = df.drop("Churn", axis=1)
        y  = df["Churn"]

        mlflow.log_param("data_path",  DATA_PATH)
        mlflow.log_param("n_samples",  len(df))
        mlflow.log_param("churn_rate", round(float(y.mean()), 4))
        mlflow.log_param("n_features", X.shape[1])

        # ── 2. Train/test split ────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        mlflow.log_param("test_size",    TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # ── 3. Scaling — only numeric cols ─────────────────────────
        scaler    = StandardScaler()
        X_train_s = X_train.copy()
        X_test_s  = X_test.copy()
        X_train_s[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
        X_test_s[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

        joblib.dump(scaler, SCALER_OUT)
        mlflow.log_artifact(SCALER_OUT, artifact_path="artifacts")

        # ── 4. Hyperparameter search ───────────────────────────────
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        xgb_base  = XGBClassifier(
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        )

        search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=PARAM_GRID,
            n_iter=N_ITER,
            scoring="roc_auc",
            cv=CV_FOLDS,
            verbose=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train_s, y_train)

        best_params = search.best_params_
        mlflow.log_params(best_params)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("n_iter",   N_ITER)
        mlflow.log_metric("best_cv_roc_auc", round(search.best_score_, 4))
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
        print(f"Best params: {best_params}")

        # ── 5. Isotonic calibration (prefit — no refitting XGBoost) ─
        # Fit IsotonicRegression directly on train raw probs → train labels.
        # This avoids CalibratedClassifierCV API changes across sklearn versions.
        xgb_tuned  = search.best_estimator_
        train_raw  = xgb_tuned.predict_proba(X_train_s)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(train_raw, y_train)

        mlflow.log_param("calibration_method", "isotonic")

        # ── 6. Evaluation ──────────────────────────────────────────
        test_raw  = xgb_tuned.predict_proba(X_test_s)[:, 1]
        cal_probs = calibrator.predict(test_raw)

        metrics = eval_at_threshold(y_test, cal_probs, THRESHOLD)
        mlflow.log_metrics({
            "test_roc_auc":   metrics["roc_auc"],
            "test_recall":    metrics["recall"],
            "test_precision": metrics["precision"],
            "test_f1":        metrics["f1"],
            "tp":             metrics["tp"],
            "fp":             metrics["fp"],
            "tn":             metrics["tn"],
            "fn":             metrics["fn"],
        })
        mlflow.log_param("decision_threshold", THRESHOLD)

        print("\n── Evaluation ──────────────────────────────────")
        print(f"  ROC-AUC:   {metrics['roc_auc']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  F1:        {metrics['f1']}")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  TN={metrics['tn']}  FN={metrics['fn']}")

        # Classification report as artifact (Windows-safe path)
        report_str  = classification_report(
            y_test,
            (cal_probs >= THRESHOLD).astype(int),
            target_names=["No Churn", "Churn"],
        )
        report_path = os.path.join("models", "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_path, artifact_path="reports")

        # ── 7. Save + log artifacts ────────────────────────────────
        joblib.dump(xgb_tuned,  MODEL_OUT)
        joblib.dump(calibrator, CALIBRATOR_OUT)

        mlflow.xgboost.log_model(xgb_tuned,  artifact_path="xgb_model")
        mlflow.sklearn.log_model(calibrator, artifact_path="calibrator")
        mlflow.log_artifact(MODEL_OUT,      artifact_path="artifacts")
        mlflow.log_artifact(CALIBRATOR_OUT, artifact_path="artifacts")

        print(f"\nModel saved      → {MODEL_OUT}")
        print(f"Calibrator saved → {CALIBRATOR_OUT}")

        # ── 8. PSI baseline ────────────────────────────────────────
        psi_baseline = compute_psi_baseline(X_train, NUMERIC_COLS)
        with open(PSI_OUT, "w") as f:
            json.dump(psi_baseline, f, indent=2)
        mlflow.log_artifact(PSI_OUT, artifact_path="artifacts")
        print(f"PSI baseline saved → {PSI_OUT}")

        # ── 9. Tags ────────────────────────────────────────────────
        mlflow.set_tags({
            "model_type":      "XGBoost",
            "dataset":         "telco-churn",
            "stage":           "production",
            "threshold_basis": "cost-optimized",
        })

        print(f"\nMLflow run complete: {run.info.run_id}")
        print("View UI: mlflow ui --port 5000")

        return metrics


if __name__ == "__main__":
    train()