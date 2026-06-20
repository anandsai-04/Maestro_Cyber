import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from pathlib import Path

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda_visuals"
MODEL_DIR = OUTPUTS_DIR / "model_outputs"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Cyber Pricing Dashboard", layout="wide", page_icon="🛡️")
st.title("🛡️ Cyber Pricing Dashboard (Use Case 1)")

# Create Tabs
tab_eda, tab_model, tab_patterns = st.tabs(["📊 EDA Visuals", "🧠 Model Outputs", "🧮 Pricing Calculator & Patterns"])

# --- EDA Visuals Tab ---
with tab_eda:
    st.header("Exploratory Data Analysis")
    html_file = EDA_DIR / "cyber_pricing_eda_dashboard.html"
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            components.html(f.read(), height=800, scrolling=True)
    else:
        st.warning(f"Dashboard file not found: {html_file.name}")

# --- Model Outputs Tab ---
with tab_model:
    st.header("Model Outputs & Diagnostics")
    diag_file = MODEL_DIR / "model_diagnostics.txt"
    if diag_file.exists():
        with open(diag_file, 'r', encoding='utf-8') as f:
            st.text(f.read())
    
    st.markdown("---")
    mc_file = MODEL_DIR / "pure_premium_indications_with_mc.csv"
    if mc_file.exists():
        df_mc = pd.read_csv(mc_file)
        st.subheader("Premium Indications Data")
        st.dataframe(df_mc, use_container_width=True)

# --- Pricing Calculator & Patterns Tab ---
with tab_patterns:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Pricing Calculator")
        st.write("Lookup a policy profile to view its risk and calculated technical premium.")
        
        mc_file = MODEL_DIR / "pure_premium_indications_with_mc.csv"
        if mc_file.exists():
            df_mc = pd.read_csv(mc_file)
            
            # Input Selection
            selected_policy = st.selectbox("Select Policy ID", df_mc["policy_id"].unique())
            pol_data = df_mc[df_mc["policy_id"] == selected_policy].iloc[0]
            
            # Display Inputs (Feature Profile)
            st.markdown("### Profile Inputs")
            i_col1, i_col2, i_col3 = st.columns(3)
            i_col1.metric("Sub-Sector", pol_data["sub_sector"])
            i_col2.metric("Revenue ($M)", f"${pol_data['revenue_mm']:,}")
            i_col3.metric("NIST Maturity", pol_data["control_maturity_nist"])
            
            # Display Extracted NLP Metrics if available
            if "high_sev_prob" in pol_data:
                st.metric("NLP Extracted High-Severity Risk", f"{pol_data['high_sev_prob']:.1%}")
                
            st.markdown("---")
            st.markdown("### Premium Calculation Output")
            st.metric("Pure Premium (Expected Risk)", f"${pol_data['pure_premium']:,.0f}")
            
            if 'risk_adjusted_technical_premium' in pol_data:
                tech_prem = pol_data['risk_adjusted_technical_premium']
            else:
                tech_prem = pol_data['technical_premium_template']
                
            st.metric("Technical Premium (Charged Price)", f"${tech_prem:,.0f}", 
                      help="Calculated by loading Pure Premium with Risk Margin & Expenses")
            
            st.metric("Current Old Premium", f"${pol_data['premium_usd']:,.0f}")
            
            diff = tech_prem - pol_data['premium_usd']
            diff_pct = (diff / pol_data['premium_usd']) * 100
            st.markdown(f"**Pricing Correction:** {'🔴 Underpriced' if diff > 0 else '🟢 Overpriced'} by ${abs(diff):,.0f} ({diff_pct:+.1f}%)")

        else:
            st.warning("Please run Use Case 1 EDA script first to generate pricing outputs.")

    with col2:
        st.header("Feature Engineering Patterns")
        features_file = DATA_DIR / "09_cyber_pricing_features.csv"
        
        if features_file.exists():
            features_df = pd.read_csv(features_file)
            
            st.subheader("1. Control Maturity vs Claim Frequency")
            st.write("Does a higher NIST score actually prevent claims? This plot proves the efficacy of the engineered features.")
            # Group by control maturity and calculate claim rate
            if "has_claim" in features_df.columns:
                claim_rates = features_df.groupby("control_maturity_nist")["has_claim"].mean().reset_index()
                fig1 = px.bar(claim_rates, x="control_maturity_nist", y="has_claim", 
                              labels={"has_claim": "Historical Claim Probability", "control_maturity_nist": "NIST Score"},
                              color="has_claim", color_continuous_scale="Reds")
                st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("2. Regulatory Finding Severity vs Pure Premium")
            st.write("How does the DistilBERT AI severity score impact the modeled pure premium?")
            if mc_file.exists() and "high_sev_prob" in df_mc.columns:
                fig2 = px.scatter(df_mc, x="high_sev_prob", y="pure_premium", 
                                  color="sub_sector", hover_data=["policy_id", "revenue_mm"],
                                  labels={"high_sev_prob": "AI High-Severity Finding Probability", "pure_premium": "Pure Premium ($)"},
                                  log_y=True)
                st.plotly_chart(fig2, use_container_width=True)
                
        else:
            st.warning("Features data not found. Run Use Case 1 EDA script.")
