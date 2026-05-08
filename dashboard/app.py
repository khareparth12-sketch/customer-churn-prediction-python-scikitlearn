# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
import time

API_URL = os.getenv("API_URL", "http://api:8000")

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Customer Churn Prediction & Analytics Dashboard")

# TIER 4: Added "Segments" tab (tab4)
tab1, tab2, tab3, tab4 = st.tabs(["Analytics", "Prediction", "Explorer", "Segments"])

# -----------------------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/processed/telco_final_processed.csv")

df = load_data()
df["ChurnLabel"] = df["Churn"].map({0: "No Churn", 1: "Churn"})

df["ContractType"] = "Month-to-month"
df.loc[df["Contract_One_year"] == 1, "ContractType"] = "One year"
df.loc[df["Contract_Two_year"] == 1, "ContractType"] = "Two year"

# TIER 4: Tenure bands for segment analysis
df["TenureBand"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, float("inf")],
    labels=["0–12 mo", "12–24 mo", "24+ mo"],
    right=True,
)

# -----------------------------------------------------------------------
# KPI METRICS
# -----------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",     df.shape[0])
col2.metric("Churn Rate",          f"{df['Churn'].mean()*100:.2f}%")
col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
col4.metric("Avg Tenure",          f"{df['tenure'].mean():.1f} months")

# -----------------------------------------------------------------------
# API HELPERS — exponential backoff, max 3 retries
# -----------------------------------------------------------------------
def call_api(payload: dict) -> dict:
    for attempt in range(3):
        try:
            res = requests.post(
                f"{API_URL}/v1/predict",
                json=payload,
                timeout=5,
            )
            return res.json()
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            if attempt < 2:
                st.toast(f"Service warming up, please wait… (retry {attempt+1}/3)")
                time.sleep(wait)
    return {"error": str(last_error)}


def call_explain_api(payload: dict) -> dict:
    for attempt in range(3):
        try:
            res = requests.post(
                f"{API_URL}/v1/explain",
                json=payload,
                timeout=10,
            )
            return res.json()
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            if attempt < 2:
                time.sleep(wait)
    return {"error": str(last_error)}

# -----------------------------------------------------------------------
# ANALYTICS TAB
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# EXPLORER TAB
# -----------------------------------------------------------------------
with tab3:
    st.subheader("Customer Explorer")
    customer_index = st.selectbox("Select a customer", df.index)
    customer_data  = df.loc[customer_index]
    st.dataframe(customer_data.to_frame().astype(str))

# -----------------------------------------------------------------------
# PREDICTION TAB
# -----------------------------------------------------------------------
with tab2:
    st.subheader("Live Customer Churn Prediction")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            tenure  = st.number_input("Tenure",          0,    100,  int(customer_data["tenure"]))
            monthly = st.number_input("Monthly Charges", 0.0,  200.0, float(customer_data["MonthlyCharges"]))
            total   = st.number_input("Total Charges",   0.0, 10000.0, float(customer_data["TotalCharges"]))

        with col2:
            senior     = st.selectbox("Senior Citizen", [0, 1])
            partner    = st.selectbox("Partner",        [0, 1])
            dependents = st.selectbox("Dependents",     [0, 1])

        submitted = st.form_submit_button("Predict Churn Risk")

    if submitted:
        payload = {
            "gender":                               1,
            "SeniorCitizen":                        senior,
            "Partner":                              partner,
            "Dependents":                           dependents,
            "tenure":                               tenure,
            "PhoneService":                         1,
            "PaperlessBilling":                     1,
            "MonthlyCharges":                       monthly,
            "TotalCharges":                         total,
            "MultipleLines_No_phone_service":       0,
            "MultipleLines_Yes":                    1,
            "InternetService_Fiber_optic":          1,
            "InternetService_No":                   0,
            "OnlineSecurity_Yes":                   0,
            "OnlineBackup_Yes":                     1,
            "DeviceProtection_Yes":                 1,
            "TechSupport_Yes":                      0,
            "StreamingTV_Yes":                      1,
            "StreamingMovies_Yes":                  1,
            "Contract_One_year":                    0,
            "Contract_Two_year":                    0,
            "PaymentMethod_Credit_card_automatic":  0,
            "PaymentMethod_Electronic_check":       1,
            "PaymentMethod_Mailed_check":           0,
        }

        with st.spinner("Predicting…"):
            result         = call_api(payload)
            explain_result = call_explain_api(payload)

        if "error" in result or "detail" in result:
            st.error(f"API error: {result.get('detail', result.get('error'))}")
        else:
            prob = result["churn_probability"]
            risk = result["churn_risk"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 22],   "color": "green"},
                        {"range": [22, 55],  "color": "orange"},
                        {"range": [55, 100], "color": "red"},
                    ],
                },
            ))
            st.plotly_chart(fig, use_container_width=True)

            if risk == "High":
                st.error("⚠️ High Churn Risk")
            else:
                st.success("✅ Low Churn Risk")

            # ------------------------------------------------------------------
            # TIER 4: SHAP Waterfall (replaces horizontal bar chart)
            # ------------------------------------------------------------------
            if explain_result and "shap_values" in explain_result:
                st.subheader("Top Factors Driving This Prediction")

                shap_vals   = explain_result["shap_values"]
                feat_names  = explain_result["feature_names"]
                base_value  = explain_result.get("base_value", 0)

                # Top 10 by absolute magnitude
                ranked = sorted(
                    zip(shap_vals, feat_names),
                    key=lambda x: abs(x[0]),
                    reverse=True,
                )[:10]
                vals, names = zip(*ranked)

                # Build running totals for waterfall
                running = base_value
                measure = []
                x_vals  = []
                text    = []
                colors  = []

                for v in vals:
                    measure.append("relative")
                    x_vals.append(v)
                    text.append(f"{v:+.3f}")
                    colors.append("#EF4444" if v > 0 else "#3B82F6")   # red=churn, blue=retain
                    running += v

                # Final total bar
                measure.append("total")
                x_vals.append(running)
                text.append(f"{running:.3f}")
                colors.append("#F59E0B")

                display_names = list(names) + ["Final Score"]

                fig = go.Figure(go.Waterfall(
                    orientation="h",
                    measure=measure,
                    x=x_vals,
                    y=display_names,
                    text=text,
                    textposition="outside",
                    connector={"line": {"color": "rgba(150,150,150,0.4)", "width": 1}},
                    increasing={"marker": {"color": "#EF4444"}},   # red → pushes toward churn
                    decreasing={"marker": {"color": "#3B82F6"}},   # blue → pulls away from churn
                    totals={"marker": {"color": "#F59E0B"}},       # amber → final score
                ))

                fig.update_layout(
                    title=dict(
                        text=(
                            f"<b>SHAP Waterfall</b>  —  base {base_value:.3f} → "
                            f"prediction {running:.3f}"
                        ),
                        font=dict(size=14),
                    ),
                    xaxis_title="SHAP contribution",
                    yaxis=dict(autorange="reversed"),
                    shapes=[
                        # Vertical line at base value
                        dict(
                            type="line",
                            x0=base_value, x1=base_value,
                            y0=-0.5, y1=len(display_names) - 0.5,
                            line=dict(color="grey", width=1, dash="dot"),
                        ),
                        # Vertical line at 0.35 decision threshold
                        dict(
                            type="line",
                            x0=0.35, x1=0.35,
                            y0=-0.5, y1=len(display_names) - 0.5,
                            line=dict(color="#F59E0B", width=1.5, dash="dash"),
                        ),
                    ],
                    annotations=[
                        dict(
                            x=0.35, y=-0.5,
                            text="threshold 0.35",
                            showarrow=False,
                            font=dict(size=10, color="#F59E0B"),
                            xanchor="left",
                        )
                    ],
                    height=460,
                    margin=dict(l=10, r=40, t=60, b=40),
                )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning("Explanation unavailable")

# -----------------------------------------------------------------------
# TIER 4: SEGMENT ANALYSIS TAB
# -----------------------------------------------------------------------
with tab4:
    st.subheader("Churn Patterns by Segment")

    # --- Row 1: Contract type churn rate ---------------------------------
    st.markdown("#### Contract Type")
    contract_churn = (
        df.groupby("ContractType")["Churn"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "ChurnRate", "sum": "Churned", "count": "Total"})
    )
    contract_churn["ChurnRate%"] = (contract_churn["ChurnRate"] * 100).round(1)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            contract_churn,
            x="ContractType",
            y="ChurnRate%",
            color="ContractType",
            text="ChurnRate%",
            labels={"ChurnRate%": "Churn Rate (%)"},
            title="Churn Rate by Contract Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            contract_churn,
            x="ContractType",
            y=["Churned", "Total"],
            barmode="overlay",
            title="Churned vs Total Customers by Contract",
            labels={"value": "Customers", "variable": ""},
            color_discrete_map={"Churned": "#EF4444", "Total": "#CBD5E1"},
        )
        fig.update_layout(yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Row 2: Tenure band churn rate -----------------------------------
    st.markdown("#### Tenure Band")
    tenure_churn = (
        df.groupby("TenureBand", observed=True)["Churn"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"mean": "ChurnRate", "sum": "Churned", "count": "Total"})
    )
    tenure_churn["ChurnRate%"] = (tenure_churn["ChurnRate"] * 100).round(1)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.bar(
            tenure_churn,
            x="TenureBand",
            y="ChurnRate%",
            color="TenureBand",
            text="ChurnRate%",
            title="Churn Rate by Tenure Band",
            color_discrete_sequence=["#EF4444", "#F59E0B", "#3B82F6"],
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.box(
            df,
            x="TenureBand",
            y="MonthlyCharges",
            color="ChurnLabel",
            title="Monthly Charges by Tenure & Churn",
            color_discrete_map={"Churn": "#EF4444", "No Churn": "#3B82F6"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Row 3: Cross-segment heatmap ------------------------------------
    st.markdown("#### Contract × Tenure Churn Rate Heatmap")
    pivot = (
        df.groupby(["ContractType", "TenureBand"], observed=True)["Churn"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .pivot(index="ContractType", columns="TenureBand", values="Churn")
    )

    fig = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="RdYlGn_r",
        title="Churn Rate % (Contract × Tenure)",
        labels=dict(color="Churn Rate %"),
        zmin=0,
        zmax=100,
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # --- Row 4: Raw segment table ----------------------------------------
    st.markdown("#### Full Segment Table")
    cross = (
        df.groupby(["ContractType", "TenureBand"], observed=True)["Churn"]
        .agg(ChurnRate="mean", Churned="sum", Total="count")
        .reset_index()
    )
    cross["ChurnRate"] = (cross["ChurnRate"] * 100).round(1).astype(str) + "%"
    st.dataframe(cross, use_container_width=True, hide_index=True)