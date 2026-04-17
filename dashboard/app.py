#dashboard\app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://api:8000")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction & Analytics Dashboard")

tab1, tab2, tab3 = st.tabs([
    "Analytics",
    "Prediction",
    "Explorer"
])

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/telco_final_processed.csv")

df = load_data()
df["ChurnLabel"] = df["Churn"].map({0: "No Churn", 1: "Churn"})

# -----------------------------
# CONTRACT TYPE
# -----------------------------
df["ContractType"] = "Month-to-month"
df.loc[df["Contract_One year"] == 1, "ContractType"] = "One year"
df.loc[df["Contract_Two year"] == 1, "ContractType"] = "Two year"

# -----------------------------
# KPI METRICS
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", df.shape[0])
col2.metric("Churn Rate", f"{df['Churn'].mean()*100:.2f}%")
col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
col4.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")

# -----------------------------
# ANALYTICS TAB
# -----------------------------
with tab1:
    st.subheader("Customer Churn Analysis")

    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df["ChurnLabel"].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x="ChurnLabel", y="tenure")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(df, x="ChurnLabel", y="MonthlyCharges")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(df, x="ContractType", color="ChurnLabel", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# EXPLORER TAB
# -----------------------------
with tab3:
    st.subheader("Customer Explorer")

    customer_index = st.selectbox("Select a customer", df.index)
    customer_data = df.loc[customer_index]

    st.dataframe(customer_data.to_frame().astype(str))

# -----------------------------
# API CALL FUNCTION (FIXED)
# -----------------------------
def call_api(payload):
    for _ in range(5):
        try:
            res = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=5
            )
            return res.json()
        except:
            time.sleep(2)
    return {"error": "API not reachable"}

# -----------------------------
# PREDICTION TAB
# -----------------------------
with tab2:
    st.subheader("Live Customer Churn Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input("Tenure", 0, 100, int(customer_data["tenure"]))
            monthly = st.number_input("Monthly Charges", 0.0, 200.0, float(customer_data["MonthlyCharges"]))
            total = st.number_input("Total Charges", 0.0, 10000.0, float(customer_data["TotalCharges"]))

        with col2:
            senior = st.selectbox("Senior Citizen", [0,1])
            partner = st.selectbox("Partner", [0,1])
            dependents = st.selectbox("Dependents", [0,1])

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        payload = {
            "gender": 1,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": 1,
            "PaperlessBilling": 1,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "MultipleLines_No phone service": 0,
            "MultipleLines_Yes": 1,
            "InternetService_Fiber optic": 1,
            "InternetService_No": 0,
            "OnlineSecurity_Yes": 0,
            "OnlineBackup_Yes": 1,
            "DeviceProtection_Yes": 1,
            "TechSupport_Yes": 0,
            "StreamingTV_Yes": 1,
            "StreamingMovies_Yes": 1,
            "Contract_One year": 0,
            "Contract_Two year": 0,
            "PaymentMethod_Credit card (automatic)": 0,
            "PaymentMethod_Electronic check": 1,
            "PaymentMethod_Mailed check": 0
        }

        result = call_api(payload)

        if "error" in result:
            st.error("API not reachable")
        else:
            prob = result["churn_probability"]
            risk = result["churn_risk"]

            shap_data = result["shap_explanation"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 35], 'color': "green"},
                        {'range': [35, 65], 'color': "orange"},
                        {'range': [65, 100], 'color': "red"}
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            if risk == "High":
                st.error("High Churn Risk")
            else:
                st.success("Low Churn Risk")

            st.subheader("Why the Model Made This Prediction")

            shap_vals = shap_data["shap_values"]
            feat_names = shap_data["feature_names"]

            ranked = sorted(
                zip(shap_vals, feat_names),
                key=lambda x: abs(x[0]),
                reverse=True
            )[:10]

            vals, names = zip(*ranked)

            colors = [
                "red" if v > 0 else "green"
                for v in vals
            ]

            fig = go.Figure(go.Bar(
                x=list(vals),
                y=list(names),
                orientation="h",
                marker_color=colors
            ))

            fig.update_layout(
                title="Top Drivers of Churn Prediction",
                yaxis=dict(autorange="reversed")
            )

            st.plotly_chart(fig, use_container_width=True)