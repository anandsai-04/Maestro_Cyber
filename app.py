import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    pass

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Advanced Cyber Actuarial Dashboard", layout="wide", page_icon="🛡️")

# Premium Glassmorphism CSS & Modern Typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #38bdf8;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    /* Chat bubbles styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
    }
    
    /* Input fields */
    .stTextInput input {
        background: rgba(0,0,0,0.2) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: white !important;
        border-radius: 8px !important;
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
tab_calc, tab_features, tab_models, tab_agent = st.tabs([
    "🧮 Interactive Pricing Engine", 
    "🧠 Engineered Features Analytics", 
    "📊 Model Comparison & Tail Risk",
    "🤖 AI Actuarial Chatbot"
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
        
        st.info("🤖 **NLP Note:** DistilBERT automatically extracts the severity probability from unstructured text (baseline 10% applied).")
        nlp_prob = 0.10

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
    st.write("These merged features are mathematically derived from your raw inputs above and fed directly into the Random Forest AI.")
    
    e_col1, e_col2, e_col3 = st.columns(3)
    
    with e_col1:
        st.markdown("**Cyber Control Score** (Merged NIST, MFA, EDR, SOC)")
        st.progress(cyber_control_score)
        st.caption(f"Score: {cyber_control_score:.2f} / 1.00")
        
    with e_col2:
        st.markdown("**Vendor Risk Pressure** (Vendors / NIST)")
        # Normalize for progress bar display (max expected around 50)
        norm_vendor = min(vendor_pressure / 50.0, 1.0)
        st.progress(norm_vendor)
        st.caption(f"Score: {vendor_pressure:.1f}")
        
    with e_col3:
        st.markdown("**Regulatory Pressure** (Findings + NLP AI)")
        # Normalize for progress bar display (max expected around 20)
        norm_reg = min(reg_pressure / 20.0, 1.0)
        st.progress(norm_reg)
        st.caption(f"Score: {reg_pressure:.2f}")
    
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
# TAB 3: MODEL COMPARISON & TAIL RISK
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
        
        st.plotly_chart(fig_tail, use_container_width=True)
        st.markdown(f"**99% Tail Value at Risk (TVaR):** The average loss of the worst 1% of claims is **${tvar:,.0f}**. This is the capital required to survive systemic events like widespread vendor ransomware.")


# ------------------------------------------
# TAB 4: AI ACTUARIAL CHATBOT
# ------------------------------------------
with tab_agent:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 AI Actuarial Report & Chat")
    st.write("This agent uses Gemini ADK to explain the portfolio's BI loss, Vendor Control Pressure, and Regulatory Pressure.")
    
    api_key = st.text_input("Enter Gemini API Key to run Agent:", type="password", key="agent_key_app1")
    
    if api_key:
        client = genai.Client(api_key=api_key)
        
        # Load SHAP global feature importances
        try:
            shap_df = pd.read_csv("outputs/model_outputs/shap_importances.csv")
            shap_importances = shap_df.set_index('Feature')['SHAP_Importance'].to_dict()
        except FileNotFoundError:
            shap_importances = {"Error": "SHAP importances not generated yet."}
            
        stats = {
            "avg_loss_ratio": df['loss_ratio'].mean(),
            "avg_bi_loss": df['bi_loss'].mean(),
            "AI_SHAP_Feature_Importances": shap_importances
        }
        
        # Auto-generate report if not in session state
        if "agent_report_app1" not in st.session_state:
            with st.spinner("Agent is analyzing deterministic stats and SHAP importances..."):
                prompt = f"""
                You are an expert Chief Actuary reviewing a cyber insurance portfolio. 
                I have already run the heavy deterministic calculations to save your tokens. 
                Crucially, I have also included the SHAP Feature Importances from the XGBoost pricing engine.
                
                Here are the statistical effects and SHAP importances:
                {stats}
                
                Write a concise executive report for the underwriters. 
                Explicitly study and explain the effect of Vendor Control Pressure, Regulatory Findings, and Cyber Control Score on BI Loss.
                **CRITICAL:** Explicitly use the AI_SHAP_Feature_Importances to explain *why* the AI Pricing Engine cares about certain features over others.
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                st.session_state["agent_report_app1"] = response.text
                
                # Initialize chat history
                st.session_state.messages_app1 = [
                    {"role": "model", "content": "I have completed the portfolio analysis. Ask me follow-up questions about BI loss, specific vendors, or risk drivers below!"}
                ]
        
        st.markdown(st.session_state["agent_report_app1"])
        
        # Chat interface
        st.markdown("---")
        st.subheader("💬 Chat with the Agent")
        
        if "messages_app1" in st.session_state:
            # Display chat messages
            for msg in st.session_state.messages_app1:
                with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                    st.write(msg["content"])
                    
            # Chat input
            if prompt_input := st.chat_input("Ask about BI Loss or Vendor Pressure..."):
                st.session_state.messages_app1.append({"role": "user", "content": prompt_input})
                with st.chat_message("user"):
                    st.write(prompt_input)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        chat = client.chats.create(model="gemini-2.5-flash")
                        chat.send_message(f"Here is the context: {stats}. Report: {st.session_state['agent_report_app1']}. Answer the user.")
                        response = chat.send_message(prompt_input)
                        st.write(response.text)
                
                st.session_state.messages_app1.append({"role": "model", "content": response.text})
    else:
        st.info("Provide API key to automatically generate the AI report and start chatting.")
    
    st.markdown('</div>', unsafe_allow_html=True)
