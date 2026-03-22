import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title(" Customer Churn Prediction & Analytics Dashboard")

tab1, tab2, tab3 = st.tabs([
    " Analytics",
    " Prediction",
    " Explorer"
])

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/telco_final_processed.csv")
    return df

df = load_data()

# Create readable churn labels
df["ChurnLabel"] = df["Churn"].map({0: "No Churn", 1: "Churn"})

# -----------------------------
# REBUILD CONTRACT TYPE
# -----------------------------
df["ContractType"] = "Month-to-month"
df.loc[df["Contract_One year"] == 1, "ContractType"] = "One year"
df.loc[df["Contract_Two year"] == 1, "ContractType"] = "Two year"

# -----------------------------
# KPI METRICS
# -----------------------------
total_customers = df.shape[0]
churn_rate = df["Churn"].mean() * 100
avg_monthly = df["MonthlyCharges"].mean()
avg_tenure = df["tenure"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churn Rate", f"{churn_rate:.2f}%")
col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
col4.metric("Avg Tenure", f"{avg_tenure:.1f} months")

#st.divider()

# -----------------------------
# CHURN ANALYSIS
# -----------------------------
with tab1:
    st.markdown("### ")
    st.subheader("Customer Churn Analysis")

    col1, col2 = st.columns(2)

    # Churn distribution
    with col1:
        churn_counts = df["ChurnLabel"].value_counts()

        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            title="Customer Churn Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Tenure vs churn
    with col2:
        fig = px.box(
            df,
            x="ChurnLabel",
            y="tenure",
            title="Tenure vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Monthly charges vs churn
    with col3:
        fig = px.box(
            df,
            x="ChurnLabel",
            y="MonthlyCharges",
            title="Monthly Charges vs Churn"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Contract type vs churn
    with col4:
        fig = px.histogram(
            df,
            x="ContractType",
            color="ChurnLabel",
            title="Contract Type vs Churn",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# CUSTOMER EXPLORER
# -----------------------------
#st.divider()
with tab3:
    st.markdown("### ")
    st.subheader(" Customer Explorer")

    customer_index = st.selectbox(
        "Select a customer from dataset",
        df.index
    )

    customer_data = df.loc[customer_index]

    st.write("### Selected Customer Profile")

    st.dataframe(customer_data.to_frame().astype(str))

# -----------------------------
# LIVE CHURN PREDICTION
# -----------------------------

st.divider()
with tab2:
    st.markdown("### ")
    st.subheader(" Live Customer Churn Prediction")

    with st.form("prediction_form"):

        st.write("Enter customer information")

        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input(
                "Tenure (months)",
                0,
                100,
                int(customer_data["tenure"])
            )

            monthly = st.number_input(
                "Monthly Charges",
                0.0,
                200.0,
                float(customer_data["MonthlyCharges"])
            )

            total = st.number_input(
                "Total Charges",
                0.0,
                10000.0,
                float(customer_data["TotalCharges"])
            )

        with col2:
            senior = st.selectbox("Senior Citizen", [0,1])
            partner = st.selectbox("Has Partner", [0,1])
            dependents = st.selectbox("Has Dependents", [0,1])

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

        try:
            response = requests.post("http://api:8000/predict", json=payload)

            result = response.json()

            prob = result["churn_probability"]
            risk = result["churn_risk"]

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Churn Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 35], 'color': "green"},
                        {'range': [35, 65], 'color': "orange"},
                        {'range': [65, 100], 'color': "red"}
                    ],
                }
            ))

            st.plotly_chart(fig, width="stretch")

            # Risk message
            if risk == "High":
                st.error(" High Churn Risk")
            else:
                st.success(" Low Churn Risk")

        except Exception as e:
            st.error("API not reachable")