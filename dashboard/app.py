# dashboard/app.py  — Tier 4 Restyled: Dark Intelligence
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
    page_title="Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------
# GLOBAL CSS — Dark Intelligence Theme
# -----------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg:        #080B12;
  --surface:   #0E1420;
  --card:      #131929;
  --border:    #1E2D45;
  --cyan:      #00D4FF;
  --cyan-dim:  rgba(0,212,255,0.12);
  --red:       #FF4560;
  --red-dim:   rgba(255,69,96,0.12);
  --green:     #00E396;
  --green-dim: rgba(0,227,150,0.12);
  --amber:     #FFB800;
  --text:      #E2EAF4;
  --muted:     #5A7194;
  --font-head: 'Syne', sans-serif;
  --font-body: 'DM Sans', sans-serif;
}

/* ── Streamlit chrome wipe ─────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

[data-testid="stHeader"]          { background: transparent !important; }
[data-testid="stSidebar"]         { background: var(--surface) !important; }
[data-testid="stToolbar"]         { display: none !important; }
footer                            { display: none !important; }
[data-testid="stDecoration"]      { display: none !important; }

/* scrollbar */
::-webkit-scrollbar              { width: 4px; height: 4px; }
::-webkit-scrollbar-track        { background: var(--bg); }
::-webkit-scrollbar-thumb        { background: var(--border); border-radius: 2px; }

/* ── Grid background ───────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size: 48px 48px;
  opacity: 0.18;
  pointer-events: none;
  z-index: 0;
}

/* ── Typography ────────────────────────────────────────────── */
h1, h2, h3,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  font-family: var(--font-head) !important;
  color: var(--text) !important;
  letter-spacing: -0.02em;
}

/* ── Tab bar ───────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  gap: 4px;
  border-bottom: 1px solid var(--border) !important;
  padding-bottom: 0;
}

[data-testid="stTabs"] [role="tab"] {
  font-family: var(--font-head) !important;
  font-size: 0.8rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 6px 6px 0 0 !important;
  padding: 8px 20px !important;
  transition: color .2s, border-color .2s;
}

[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--cyan) !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--cyan) !important;
  border-color: var(--border) var(--border) var(--bg) !important;
  background: var(--card) !important;
  box-shadow: 0 0 12px var(--cyan-dim) !important;
}

/* ── Cards ─────────────────────────────────────────────────── */
.ci-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 22px 26px;
  position: relative;
  overflow: hidden;
  transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
  animation: fadeSlideUp .5s ease both;
}
.ci-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(0,0,0,.4);
  border-color: var(--cyan);
}
.ci-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-color, var(--cyan)), transparent);
}

/* KPI card variants */
.ci-card.kpi-cyan  { --accent-color: var(--cyan);  }
.ci-card.kpi-red   { --accent-color: var(--red);   }
.ci-card.kpi-green { --accent-color: var(--green); }
.ci-card.kpi-amber { --accent-color: var(--amber); }

.ci-card .kpi-icon {
  font-size: 1.4rem;
  margin-bottom: 10px;
  display: block;
}
.ci-card .kpi-label {
  font-family: var(--font-body);
  font-size: 0.7rem;
  font-weight: 500;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
}
.ci-card .kpi-value {
  font-family: var(--font-head);
  font-size: 2rem;
  font-weight: 800;
  color: var(--text);
  line-height: 1;
  letter-spacing: -0.03em;
}
.ci-card .kpi-value span {
  color: var(--accent-color, var(--cyan));
}
.ci-card .kpi-sub {
  font-size: 0.72rem;
  color: var(--muted);
  margin-top: 6px;
}

/* Section header */
.ci-section {
  font-family: var(--font-head);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  border-left: 3px solid var(--cyan);
  padding-left: 10px;
  margin: 28px 0 16px;
}

/* ── Risk badge ────────────────────────────────────────────── */
.risk-high {
  display: inline-block;
  background: var(--red-dim);
  color: var(--red);
  border: 1px solid var(--red);
  border-radius: 6px;
  padding: 6px 18px;
  font-family: var(--font-head);
  font-weight: 700;
  font-size: 0.95rem;
  letter-spacing: 0.05em;
  animation: pulse-red 1.6s ease-in-out infinite;
}
.risk-low {
  display: inline-block;
  background: var(--green-dim);
  color: var(--green);
  border: 1px solid var(--green);
  border-radius: 6px;
  padding: 6px 18px;
  font-family: var(--font-head);
  font-weight: 700;
  font-size: 0.95rem;
  letter-spacing: 0.05em;
}

/* ── Form inputs ───────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"]   > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
  transition: border-color .2s;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"]   > div > div:focus-within {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px var(--cyan-dim) !important;
}

/* Submit button */
[data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(135deg, #00A8CC, #00D4FF) !important;
  color: #080B12 !important;
  font-family: var(--font-head) !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 36px !important;
  transition: transform .15s, box-shadow .15s !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 24px rgba(0,212,255,.35) !important;
}

/* Divider */
hr { border-color: var(--border) !important; opacity: 1 !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden;
}

/* ── Animations ────────────────────────────────────────────── */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0);    }
}
@keyframes pulse-red {
  0%, 100% { box-shadow: 0 0 0 0 rgba(255,69,96,.4); }
  50%       { box-shadow: 0 0 0 8px rgba(255,69,96,0); }
}

/* staggered card animation delays */
.ci-card:nth-child(1) { animation-delay: .05s; }
.ci-card:nth-child(2) { animation-delay: .10s; }
.ci-card:nth-child(3) { animation-delay: .15s; }
.ci-card:nth-child(4) { animation-delay: .20s; }

/* ── Spinner ───────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
  border-top-color: var(--cyan) !important;
}

/* Toast */
[data-testid="stToast"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# PLOTLY DARK TEMPLATE
# -----------------------------------------------------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#5A7194", size=11),
    title_font=dict(family="Syne, sans-serif", color="#E2EAF4", size=13),
    xaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45", tickcolor="#1E2D45"),
    yaxis=dict(gridcolor="#1E2D45", linecolor="#1E2D45", tickcolor="#1E2D45"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1E2D45"),
    margin=dict(l=10, r=10, t=48, b=10),
    colorway=["#00D4FF","#FF4560","#00E396","#FFB800","#775DD0","#3B82F6"],
)

def dark(fig, **extra):
    fig.update_layout(**{**PLOT_LAYOUT, **extra})
    return fig

# -----------------------------------------------------------------------
# PAGE HEADER
# -----------------------------------------------------------------------
st.markdown("""
<div style="padding: 32px 0 8px; animation: fadeSlideUp .4s ease both;">
  <div style="font-family:'Syne',sans-serif; font-size:1.75rem; font-weight:800;
              letter-spacing:-0.03em; color:#E2EAF4;">
    ⚡ Churn <span style="color:#00D4FF;">Intelligence</span>
  </div>
  <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem;
              color:#5A7194; margin-top:4px; letter-spacing:0.04em;">
    TELCO CUSTOMER RETENTION PLATFORM &nbsp;·&nbsp; XGBOOST MODEL &nbsp;·&nbsp; ROC-AUC 0.84
  </div>
</div>
""", unsafe_allow_html=True)

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
df["TenureBand"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, float("inf")],
    labels=["0–12 mo", "12–24 mo", "24+ mo"],
    right=True,
)

# -----------------------------------------------------------------------
# KPI CARDS
# -----------------------------------------------------------------------
total     = df.shape[0]
churn_rt  = df["Churn"].mean() * 100
avg_mo    = df["MonthlyCharges"].mean()
avg_ten   = df["tenure"].mean()

k1, k2, k3, k4 = st.columns(4)
cards = [
    (k1, "kpi-cyan",  "👥", "Total Customers",    f"{total:,}",         "", ""),
    (k2, "kpi-red",   "📉", "Churn Rate",          f"{churn_rt:.1f}",   "%", "of all customers"),
    (k3, "kpi-green", "💳", "Avg Monthly Charges", f"${avg_mo:.2f}",    "",  "per customer"),
    (k4, "kpi-amber", "📅", "Avg Tenure",          f"{avg_ten:.1f}",    "",  "months"),
]
for col, cls, icon, label, val, unit, sub in cards:
    with col:
        st.markdown(f"""
        <div class="ci-card {cls}">
          <span class="kpi-icon">{icon}</span>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}<span>{unit}</span></div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊  Analytics", "🎯  Prediction", "🔍  Explorer", "📂  Segments"])

# -----------------------------------------------------------------------
# API HELPERS
# -----------------------------------------------------------------------
def call_api(payload: dict) -> dict:
    for attempt in range(3):
        try:
            res = requests.post(f"{API_URL}/v1/predict", json=payload, timeout=5)
            return res.json()
        except Exception as e:
            last_error = e
            if attempt < 2:
                st.toast(f"Service warming up… (retry {attempt+1}/3)")
                time.sleep(2 ** attempt)
    return {"error": str(last_error)}

def call_explain_api(payload: dict) -> dict:
    for attempt in range(3):
        try:
            res = requests.post(f"{API_URL}/v1/explain", json=payload, timeout=10)
            return res.json()
        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {"error": str(last_error)}

# -----------------------------------------------------------------------
# TAB 1 — ANALYTICS
# -----------------------------------------------------------------------
with tab1:
    st.markdown('<div class="ci-section">Churn Distribution</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        churn_counts = df["ChurnLabel"].value_counts()
        fig = go.Figure(go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            hole=0.6,
            marker=dict(colors=["#00E396","#FF4560"],
                        line=dict(color="#080B12", width=3)),
            textfont=dict(family="DM Sans, sans-serif", color="#E2EAF4"),
        ))
        fig.update_layout(**{**PLOT_LAYOUT,
            "title": "Churn Split",
            "annotations": [dict(text=f"{churn_rt:.1f}%<br><span style='font-size:10px'>churn</span>",
                              font=dict(size=18, family="Syne, sans-serif", color="#FF4560"),
                              showarrow=False)],
            "showlegend": True,
        })
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x="ChurnLabel", y="tenure",
                     color="ChurnLabel",
                     color_discrete_map={"Churn":"#FF4560","No Churn":"#00E396"})
        dark(fig, title="Tenure by Churn Status", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="ci-section">Revenue Signals</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(df, x="ChurnLabel", y="MonthlyCharges",
                     color="ChurnLabel",
                     color_discrete_map={"Churn":"#FF4560","No Churn":"#00E396"})
        dark(fig, title="Monthly Charges by Churn", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(df, x="ContractType", color="ChurnLabel", barmode="group",
                           color_discrete_map={"Churn":"#FF4560","No Churn":"#00D4FF"})
        dark(fig, title="Churn by Contract Type")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# TAB 3 — EXPLORER
# -----------------------------------------------------------------------
with tab3:
    st.markdown('<div class="ci-section">Customer Record</div>', unsafe_allow_html=True)
    customer_index = st.selectbox("Select a customer", df.index)
    customer_data  = df.loc[customer_index]
    st.dataframe(customer_data.to_frame().astype(str), use_container_width=True)

# -----------------------------------------------------------------------
# TAB 2 — PREDICTION
# -----------------------------------------------------------------------
with tab2:
    st.markdown('<div class="ci-section">Customer Input</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure  = st.number_input("Tenure (months)",   0,    100,   int(customer_data["tenure"]))
            monthly = st.number_input("Monthly Charges",   0.0,  200.0, float(customer_data["MonthlyCharges"]))
            total   = st.number_input("Total Charges",     0.0, 10000.0,float(customer_data["TotalCharges"]))
        with col2:
            senior     = st.selectbox("Senior Citizen", [0, 1])
            partner    = st.selectbox("Partner",        [0, 1])
            dependents = st.selectbox("Dependents",     [0, 1])
        submitted = st.form_submit_button("⚡  Run Prediction")

    if submitted:
        payload = {
            "gender":1,"SeniorCitizen":senior,"Partner":partner,"Dependents":dependents,
            "tenure":tenure,"PhoneService":1,"PaperlessBilling":1,
            "MonthlyCharges":monthly,"TotalCharges":total,
            "MultipleLines_No_phone_service":0,"MultipleLines_Yes":1,
            "InternetService_Fiber_optic":1,"InternetService_No":0,
            "OnlineSecurity_Yes":0,"OnlineBackup_Yes":1,"DeviceProtection_Yes":1,
            "TechSupport_Yes":0,"StreamingTV_Yes":1,"StreamingMovies_Yes":1,
            "Contract_One_year":0,"Contract_Two_year":0,
            "PaymentMethod_Credit_card_automatic":0,
            "PaymentMethod_Electronic_check":1,"PaymentMethod_Mailed_check":0,
        }

        with st.spinner("Running inference…"):
            result         = call_api(payload)
            explain_result = call_explain_api(payload)

        if "error" in result or "detail" in result:
            st.error(f"API error: {result.get('detail', result.get('error'))}")
        else:
            prob = result["churn_probability"]
            risk = result["churn_risk"]

            st.markdown('<div class="ci-section">Result</div>', unsafe_allow_html=True)

            rc1, rc2 = st.columns([2, 1])
            with rc1:
                # Gauge
                gauge_color = "#FF4560" if prob >= 0.35 else "#00E396"
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number=dict(suffix="%", font=dict(family="Syne, sans-serif",
                                                       color=gauge_color, size=36)),
                    title=dict(text="Churn Probability",
                               font=dict(family="Syne, sans-serif", color="#5A7194", size=13)),
                    gauge=dict(
                        axis=dict(range=[0,100], tickcolor="#1E2D45",
                                  tickfont=dict(color="#5A7194")),
                        bar=dict(color=gauge_color, thickness=0.25),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 22],   color="#0E1420"),
                            dict(range=[22, 55],  color="#131929"),
                            dict(range=[55, 100], color="#1A1020"),
                        ],
                        threshold=dict(
                            line=dict(color="#FFB800", width=2),
                            thickness=0.75, value=35,
                        ),
                    ),
                ))
                fig.update_layout(**{**PLOT_LAYOUT,
                                    "height": 280,
                                    "margin": dict(l=20,r=20,t=40,b=0)})
                st.plotly_chart(fig, use_container_width=True)

            with rc2:
                st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)
                if risk == "High":
                    st.markdown('<div class="risk-high">⚠️ HIGH RISK</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="risk-low">✅ LOW RISK</div>',
                                unsafe_allow_html=True)
                st.markdown(f"""
                <div style="margin-top:16px; font-family:'DM Sans',sans-serif;
                             font-size:0.78rem; color:#5A7194; line-height:1.7;">
                  Probability<br>
                  <span style="color:#E2EAF4; font-size:1.1rem;
                               font-family:'Syne',sans-serif; font-weight:700;">
                    {prob*100:.1f}%
                  </span><br><br>
                  Threshold<br>
                  <span style="color:#FFB800; font-size:1rem;
                               font-family:'Syne',sans-serif; font-weight:600;">
                    35%
                  </span>
                </div>
                """, unsafe_allow_html=True)

            # ── SHAP Waterfall ─────────────────────────────────────────
            if explain_result and "shap_values" in explain_result:
                st.markdown('<div class="ci-section">SHAP Waterfall — Feature Contributions</div>',
                            unsafe_allow_html=True)

                shap_vals  = explain_result["shap_values"]
                feat_names = explain_result["feature_names"]
                base_value = explain_result.get("base_value", 0)

                ranked = sorted(zip(shap_vals, feat_names),
                                key=lambda x: abs(x[0]), reverse=True)[:10]
                vals, names = zip(*ranked)

                running  = base_value
                measure, x_vals, text_vals = [], [], []
                for v in vals:
                    measure.append("relative")
                    x_vals.append(v)
                    text_vals.append(f"{v:+.3f}")
                    running += v
                measure.append("total")
                x_vals.append(running)
                text_vals.append(f"{running:.3f}")
                display_names = list(names) + ["▶ Final Score"]

                fig = go.Figure(go.Waterfall(
                    orientation="h",
                    measure=measure,
                    x=x_vals,
                    y=display_names,
                    text=text_vals,
                    textposition="outside",
                    textfont=dict(family="DM Sans, sans-serif", color="#E2EAF4", size=10),
                    connector=dict(line=dict(color="#1E2D45", width=1, dash="dot")),
                    increasing=dict(marker=dict(color="#FF4560",
                                                line=dict(color="#FF4560",width=0))),
                    decreasing=dict(marker=dict(color="#00D4FF",
                                                line=dict(color="#00D4FF",width=0))),
                    totals=dict(marker=dict(color="#FFB800",
                                            line=dict(color="#FFB800",width=0))),
                ))

                fig.update_layout(**{**PLOT_LAYOUT,
                    "title": dict(
                        text=(f"Base {base_value:.3f} → Prediction {running:.3f}"
                              f"  <span style='color:#5A7194;font-size:10px'>"
                              f"(🔴 toward churn · 🔵 toward retain · 🟡 final)</span>"),
                        font=dict(family="Syne, sans-serif", color="#E2EAF4", size=12),
                    ),
                    "xaxis_title": "SHAP contribution",
                    "yaxis": dict(autorange="reversed", gridcolor="#1E2D45",
                               tickfont=dict(size=10)),
                    "height": 460,
                    "margin": dict(l=10, r=50, t=56, b=20),
                    "shapes": [
                        dict(type="line", x0=base_value, x1=base_value,
                             y0=-0.5, y1=len(display_names)-0.5,
                             line=dict(color="#5A7194", width=1, dash="dot")),
                        dict(type="line", x0=0.35, x1=0.35,
                             y0=-0.5, y1=len(display_names)-0.5,
                             line=dict(color="#FFB800", width=1.5, dash="dash")),
                    ],
                    "annotations": [dict(x=0.35, y=-0.5, text="threshold",
                                      showarrow=False,
                                      font=dict(size=9, color="#FFB800"),
                                      xanchor="left")],
                })
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Explanation unavailable.")

# -----------------------------------------------------------------------
# TAB 4 — SEGMENTS
# -----------------------------------------------------------------------
with tab4:
    # ── Contract type ────────────────────────────────────────────────
    st.markdown('<div class="ci-section">Contract Type Analysis</div>',
                unsafe_allow_html=True)

    contract_churn = (
        df.groupby("ContractType")["Churn"]
        .agg(["mean","sum","count"]).reset_index()
        .rename(columns={"mean":"ChurnRate","sum":"Churned","count":"Total"})
    )
    contract_churn["ChurnRate%"] = (contract_churn["ChurnRate"]*100).round(1)

    sc1, sc2 = st.columns(2)
    with sc1:
        fig = px.bar(contract_churn, x="ContractType", y="ChurnRate%",
                     color="ContractType", text="ChurnRate%",
                     color_discrete_sequence=["#FF4560","#FFB800","#00E396"])
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_width=0)
        dark(fig, title="Churn Rate by Contract", showlegend=False,
             yaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        fig = px.bar(contract_churn, x="ContractType",
                     y=["Churned","Total"], barmode="overlay",
                     color_discrete_map={"Churned":"#FF4560","Total":"#1E2D45"})
        dark(fig, title="Churned vs Total by Contract")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tenure band ──────────────────────────────────────────────────
    st.markdown('<div class="ci-section">Tenure Band Analysis</div>',
                unsafe_allow_html=True)

    tenure_churn = (
        df.groupby("TenureBand", observed=True)["Churn"]
        .agg(["mean","sum","count"]).reset_index()
        .rename(columns={"mean":"ChurnRate","sum":"Churned","count":"Total"})
    )
    tenure_churn["ChurnRate%"] = (tenure_churn["ChurnRate"]*100).round(1)

    tc1, tc2 = st.columns(2)
    with tc1:
        fig = px.bar(tenure_churn, x="TenureBand", y="ChurnRate%",
                     color="TenureBand", text="ChurnRate%",
                     color_discrete_sequence=["#FF4560","#FFB800","#00D4FF"])
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_width=0)
        dark(fig, title="Churn Rate by Tenure Band", showlegend=False,
             yaxis_range=[0,100])
        st.plotly_chart(fig, use_container_width=True)

    with tc2:
        fig = px.box(df, x="TenureBand", y="MonthlyCharges",
                     color="ChurnLabel",
                     color_discrete_map={"Churn":"#FF4560","No Churn":"#00D4FF"})
        dark(fig, title="Monthly Charges by Tenure & Churn")
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────
    st.markdown('<div class="ci-section">Contract × Tenure Heatmap</div>',
                unsafe_allow_html=True)

    pivot = (
        df.groupby(["ContractType","TenureBand"], observed=True)["Churn"]
        .mean().mul(100).round(1).reset_index()
        .pivot(index="ContractType", columns="TenureBand", values="Churn")
    )
    fig = px.imshow(pivot, text_auto=True,
                    color_continuous_scale=[[0,"#00293B"],[0.5,"#FFB800"],[1,"#FF4560"]],
                    title="Churn Rate % (Contract × Tenure)",
                    zmin=0, zmax=100)
    dark(fig, height=300, margin=dict(l=10,r=10,t=48,b=10))
    fig.update_traces(textfont=dict(family="Syne, sans-serif",
                                    color="#E2EAF4", size=13))
    st.plotly_chart(fig, use_container_width=True)

    # ── Table ─────────────────────────────────────────────────────────
    st.markdown('<div class="ci-section">Segment Table</div>',
                unsafe_allow_html=True)
    cross = (
        df.groupby(["ContractType","TenureBand"], observed=True)["Churn"]
        .agg(ChurnRate="mean", Churned="sum", Total="count").reset_index()
    )
    cross["ChurnRate"] = (cross["ChurnRate"]*100).round(1).astype(str) + "%"
    st.dataframe(cross, use_container_width=True, hide_index=True)