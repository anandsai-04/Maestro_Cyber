import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Advanced Cyber Actuarial Dashboard", layout="wide", page_icon="🛡️")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #00d2ff;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Advanced Cyber Risk & AI Pricing Dashboard")
st.markdown("*Use Case 1: Dynamic Frequency-Severity Pricing & Engineered Features Insights*")

@st.cache_data
def load_data():
    features_file = DATA_DIR / "09_cyber_pricing_features.csv"
    if features_file.exists():
        return pd.read_csv(features_file)
    return None

df = load_data()

if df is None:
    st.error("Data not found. Please ensure `09_cyber_pricing_features.csv` is in the `data/` directory.")
    st.stop()

# ==========================================
# MACHINE LEARNING ENGINE (TRAINED ON FLY)
# ==========================================
# We train the Random Forest directly in Streamlit to enable 100% dynamic user-input pricing
categorical_cols = ["sub_sector", "cloud_provider_primary"]
numeric_cols = [
    "exposure_size_score", "cyber_control_score", "control_gap_score", 
    "vendor_control_pressure", "regulatory_findings_pressure", 
    "critical_operations_score", "payment_trading_flag", "hybrid_cloud_flag"
]

# Prepare Data
X = df[numeric_cols + categorical_cols].copy()
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
features_list = X.columns.tolist()

y_freq = df["had_claim"]
severity_mask = df["total_loss"] > 0
X_sev = X[severity_mask]
y_sev = np.log1p(df[severity_mask]["total_loss"])

# Train Frequency Model (Random Forest & GLM for comparison)
rf_freq = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf_freq.fit(X, y_freq)

glm_freq = LogisticRegression(max_iter=1000)
glm_freq.fit(X, y_freq)

# Train Severity Model (Random Forest)
rf_sev = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_sev.fit(X_sev, y_sev)

# Calculate AUC
df["rf_prob"] = rf_freq.predict_proba(X)[:, 1]
df["glm_prob"] = glm_freq.predict_proba(X)[:, 1]
rf_auc = roc_auc_score(y_freq, df["rf_prob"])
glm_auc = roc_auc_score(y_freq, df["glm_prob"])


# ==========================================
# TABS
# ==========================================
tab_calc, tab_features, tab_models = st.tabs([
    "🧮 Interactive Pricing Engine", 
    "🧠 Engineered Features Analytics", 
    "📊 Model Comparison & Tail Risk"
])


# ------------------------------------------
# TAB 1: INTERACTIVE PRICING ENGINE
# ------------------------------------------
with tab_calc:
    st.markdown("### Dynamically Price a New Policy Profile")
    st.write("Adjust the features below. The Random Forest AI will calculate the expected frequency, expected severity, and final technical premium on the fly.")
    
    col_in1, col_in2, col_in3 = st.columns(3)
    
    with col_in1:
        st.subheader("Profile")
        sub_sector = st.selectbox("Type of Institution", df["sub_sector"].unique())
        cloud_provider = st.selectbox("Cloud Architecture", df["cloud_provider_primary"].unique())
        revenue = st.slider("Revenue ($M)", min_value=10, max_value=50000, value=500)
        
    with col_in2:
        st.subheader("Controls & Security")
        nist = st.slider("NIST Control Maturity", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
        mfa = st.slider("MFA Coverage %", min_value=0, max_value=100, value=50, step=5)
        edr = st.checkbox("EDR Deployed", value=True)
        soc = st.checkbox("24/7 SOC", value=False)
        n_vendors = st.slider("Number of 3rd Party Vendors", min_value=0, max_value=150, value=30)
        hybrid_flag = 1 if cloud_provider == "Hybrid" else 0
        
    with col_in3:
        st.subheader("Operations & Regulatory")
        has_trading = st.checkbox("Has Trading Desk", value=False)
        processes_payments = st.checkbox("Processes Payments", value=True)
        
        st.markdown("**Regulatory Audit History**")
        n_findings = st.slider("Total Regulatory Findings", min_value=0, max_value=10, value=2)
        n_high = st.slider("High Severity Findings", min_value=0, max_value=n_findings, value=0)
        n_med = st.slider("Medium Severity Findings", min_value=0, max_value=n_findings-n_high if n_findings-n_high > 0 else 0, value=0)
        nlp_prob = st.slider("DistilBERT NLP High-Sev Probability", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    # Transform Raw Inputs to Engineered Features
    exp_size = np.log1p(revenue) / 15.0 # Rough scaling
    crit_ops = int(has_trading) + int(processes_payments)
    pay_trad = 1 if (has_trading and processes_payments) else 0
    
    # 1. Cyber Control Score
    nist_score = nist / 5.0
    mfa_score = mfa / 100.0
    cyber_control_score = (0.40 * nist_score) + (0.25 * mfa_score) + (0.20 * int(edr)) + (0.15 * int(soc))
    control_gap = 1.0 - cyber_control_score
    
    # 2. Vendor Control Pressure
    vendor_pressure = n_vendors / (nist + 0.1)
    
    # 3. Regulatory Findings Pressure
    high_sev_rate = n_high / (n_findings + 1.0)
    med_sev_rate = n_med / (n_findings + 1.0)
    reg_pressure = np.log1p(n_findings) * (1.0 + high_sev_rate + nlp_prob) * (1.0 + 0.25 * med_sev_rate)
    
    st.markdown("### 📊 Live Engineered Feature Calculations")
    st.write("These scores are dynamically calculated from your raw inputs and fed directly into the Random Forest AI.")
    e_col1, e_col2, e_col3 = st.columns(3)
    e_col1.metric("Calculated Cyber Control Score", f"{cyber_control_score:.2f} / 1.0", help="Merged from NIST, MFA, EDR, and SOC")
    e_col2.metric("Calculated Vendor Risk Pressure", f"{vendor_pressure:.1f}", help="Vendors divided by NIST score")
    e_col3.metric("Calculated Regulatory Pressure", f"{reg_pressure:.2f}", help="Merged finding counts with AI severity probability")
    
    input_dict = {
        "exposure_size_score": exp_size,
        "cyber_control_score": cyber_control_score,
        "control_gap_score": control_gap,
        "vendor_control_pressure": vendor_pressure,
        "regulatory_findings_pressure": reg_pressure,
        "critical_operations_score": crit_ops,
        "payment_trading_flag": pay_trad,
        "hybrid_cloud_flag": hybrid_flag
    }
    
    # Add categoricals exactly as they appear in dummy columns
    for col in categorical_cols:
        val = sub_sector if col == "sub_sector" else cloud_provider
        dummy_col = f"{col}_{val}"
        if dummy_col in features_list:
            input_dict[dummy_col] = 1
            
    # Fill remaining dummy columns with 0
    input_array = []
    for f in features_list:
        input_array.append(input_dict.get(f, 0))
        
    input_df = pd.DataFrame([input_array], columns=features_list)
    
    # Predict
    pred_freq = rf_freq.predict_proba(input_df)[0][1]
    pred_log_sev = rf_sev.predict(input_df)[0]
    pred_sev = np.expm1(pred_log_sev)
    
    pure_premium = pred_freq * pred_sev
    
    # Risk Load (Capital + Expense)
    expense_ratio = 0.25
    risk_margin = 0.20 # Base margin, scales with pure premium in this dynamic context
    tech_premium = pure_premium / (1 - expense_ratio - risk_margin)
    
    st.markdown("---")
    st.markdown("### Pricing Output")
    o_col1, o_col2, o_col3, o_col4 = st.columns(4)
    
    with o_col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Predicted Claim Frequency</div><div class="metric-value">{pred_freq:.2%}</div></div>', unsafe_allow_html=True)
    with o_col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Expected Claim Severity</div><div class="metric-value">${pred_sev:,.0f}</div></div>', unsafe_allow_html=True)
    with o_col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Modeled Pure Premium</div><div class="metric-value">${pure_premium:,.0f}</div></div>', unsafe_allow_html=True)
    with o_col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Technical Premium (Charged)</div><div class="metric-value">${tech_premium:,.0f}</div></div>', unsafe_allow_html=True)
        

# ------------------------------------------
# TAB 2: ENGINEERED FEATURES ANALYTICS
# ------------------------------------------
with tab_features:
    st.markdown("### The Power of Merged Engineered Features")
    st.write("""
    Instead of using raw, unscaled variables, we mathematically merged highly correlated variables to create robust index scores. 
    Here is how these newly engineered features directly impact historical claims.
    """)
    
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        st.subheader("1. Cyber Control Score vs Claim Rate")
        st.info("**What is this?** We merged `NIST Maturity`, `MFA %`, `EDR flag`, and `SOC flag` into a single 0-to-1 Control Score.")
        # Bin the score
        df["control_bin"] = pd.qcut(df["cyber_control_score"], q=4, labels=["Weak", "Developing", "Strong", "Excellent"])
        fig_c = px.bar(df.groupby("control_bin", observed=False)["had_claim"].mean().reset_index(), 
                       x="control_bin", y="had_claim", color="had_claim", color_continuous_scale="Reds",
                       labels={"control_bin": "Merged Control Score", "had_claim": "Claim Probability"})
        st.plotly_chart(fig_c, use_container_width=True)
        
    with f_col2:
        st.subheader("2. Vendor Control Pressure vs Total Loss")
        st.info("**What is this?** We divided `Number of Vendors` by the `NIST Maturity` to create a ratio indicating Third-Party Supply Chain Risk.")
        fig_v = px.scatter(df, x="vendor_control_pressure", y="total_loss", color="sub_sector", 
                           log_y=True,
                           labels={"vendor_control_pressure": "Vendor Risk Pressure Score", "total_loss": "Historical Claim Loss ($)"})
        st.plotly_chart(fig_v, use_container_width=True)

# ------------------------------------------
# TAB 3: MODEL Comparison & TAIL RISK
# ------------------------------------------
with tab_models:
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.subheader("Why Random Forest beats GLM")
        st.write("Generalized Linear Models (GLMs) struggle with complex, non-linear cyber risks (e.g., when a bank uses a Hybrid Cloud *and* has weak MFA). The Random Forest implicitly captures these interactions.")
        
        comp_df = pd.DataFrame({
            "Model": ["GLM (Logistic)", "Random Forest (AI)"],
            "AUC-ROC Score": [glm_auc, rf_auc]
        })
        fig_comp = px.bar(comp_df, x="Model", y="AUC-ROC Score", color="Model", text_auto='.3f',
                          color_discrete_sequence=["#555555", "#00d2ff"])
        st.plotly_chart(fig_comp, use_container_width=True)
        
    with m_col2:
        st.subheader("Value at Risk (VaR) & TVaR")
        st.write("Using the portfolio's total losses, we can identify the Tail Risk for Capital allocation.")
        
        losses = df[df["total_loss"] > 0]["total_loss"]
        p95 = np.percentile(losses, 95)
        p99 = np.percentile(losses, 99)
        tvar = np.mean(losses[losses >= p99])
        
        fig_tail = px.histogram(losses, nbins=50, log_y=True, 
                                labels={"value": "Loss Amount ($)", "count": "Frequency"},
                                color_discrete_sequence=['#FF4B4B'])
        fig_tail.add_vline(x=p95, line_dash="dash", line_color="orange", annotation_text="VaR 95%")
        fig_tail.add_vline(x=p99, line_dash="dash", line_color="red", annotation_text="VaR 99%")
        fig_tail.add_vline(x=tvar, line_dash="solid", line_color="darkred", annotation_text="TVaR 99%")
        fig_tail.update_layout(showlegend=False, title="Tail Distribution of Historical Losses")
        st.plotly_chart(fig_tail, use_container_width=True)
