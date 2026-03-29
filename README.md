# Customer Churn Analysis – Telecommunications Industry

## Overview

This project analyzes customer churn patterns in a telecom company using exploratory data analysis, data cleaning, and machine learning techniques. We have identified high-risk customer profiles and deployed an XGBoost classification model via a FastAPI application to provide real-time churn predictions.

## Tools & Technologies

- **Programming:** Python (Pandas, NumPy, Scikit-learn, XGBoost)
- **Data Visualization:** Matplotlib, Seaborn
- **API Development:** FastAPI, Uvicorn
- **Environment:** Jupyter Notebook

## Project Structure

```text
api/
 ├── app.py                     # FastAPI application for serving the model
 ├── schema.py                  # Pydantic schemas for API inputs
 └── utils.py                   # API utilities and feature definitions

data/
 ├── raw/                       # Raw, unprocessed datasets
 └── processed/                 # Cleaned data ready for modeling

models/
 ├── xgb_tuned_model.pkl        # Best performing tuned XGBoost model
 ├── rf_model.pkl               # Random Forest model
 ├── logistic_model.pkl         # Logistic Regression model
 └── scaler.pkl                 # Data scaler for preprocessing

notebooks/
 ├── 01_data_overview.ipynb     # Initial data review
 ├── 02_data_cleaning.ipynb     # Handling missing values and outliers
 ├── 03_eda_churn.ipynb         # Exploratory Data Analysis
 ├── 04_preprocessing.ipynb     # Feature engineering and scaling
 ├── 05_modeling_and_tuning.ipynb # Training Models & Hyperparameter Tuning
 └── 06_model_interpretation.ipynb# Feature Importance & SHAP analysis

reports/
 ├── churn_analysis_report.md   # Business insights from EDA
 └── model_report.md            # Model evaluation metrics

sql/                            # (Pending) SQL scripts for data extraction
dashboard/                      # (Pending) PowerBI/Tableau dashboard files
```

## Methodology

1. **Data Collection & Exploration:** Understanding the raw data structure.
2. **Data Cleaning:** Handling missing variables and correcting data types.
3. **Exploratory Data Analysis (EDA):** Identifying trends among churned vs. retained customers.
4. **Data Preprocessing:** Feature engineering, encoding, scaling, and handling class imbalances.
5. **Machine Learning Modeling:** Training and tuning Logistic Regression, Random Forest, and XGBoost models.
6. **Model Deployment:** Serving the best-performing model (XGBoost) as a REST API using FastAPI.

## Key Insights

- **Early Attrition:** Nearly 48% churn observed among first-year customers.
- **Contract Type:** Month-to-month contracts showed the highest attrition rate.
- **Payment Method:** Electronic check users exhibited elevated churn.
- **Pricing:** Mid-to-high pricing tiers faced increased customer dissatisfaction, whereas long-term contracts demonstrated strong retention.

## API Usage

The XGBoost model is served at the `/predict` endpoint via FastApi.
You can run the API locally using:

```bash
uvicorn api.app:app --reload
```

## Future Work

- **SQL Data Pipeline:** Implement SQL scripts for robust data extraction.
- **Interactive Dashboards:** Develop an interactive PowerBI or Tableau dashboard for stakeholder reporting (within the `dashboard/` folder).
- **A/B Testing:** Design and perform A/B testing on retention campaigns.
- **Monitoring:** Integrate drift monitoring for the deployed API.
