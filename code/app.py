import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from pathlib import Path

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda_visuals"
MODEL_DIR = OUTPUTS_DIR / "model_outputs"

st.set_page_config(page_title="Cyber Pricing Dashboard", layout="wide")
st.title("Cyber Pricing Dashboard")

# Create Tabs
tab_eda, tab_model = st.tabs(["📊 EDA Visuals", "🧠 Model Outputs"])

# --- EDA Visuals Tab ---
with tab_eda:
    st.header("Exploratory Data Analysis")
    
    # Check if HTML dashboard exists
    html_file = EDA_DIR / "cyber_pricing_eda_dashboard.html"
    if html_file.exists():
        st.subheader("EDA Interactive Dashboard")
        with open(html_file, 'r', encoding='utf-8') as f:
            html_data = f.read()
        components.html(html_data, height=800, scrolling=True)
    else:
        st.warning(f"Dashboard file not found: {html_file.name}")
    
    st.markdown("---")
    st.subheader("Summary Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sector Claim Loss Premium Summary**")
        file1 = EDA_DIR / "sector_claim_loss_premium_summary.csv"
        if file1.exists():
            st.dataframe(pd.read_csv(file1), use_container_width=True)
            
        st.write("**Prior Incident Claim Summary**")
        file2 = EDA_DIR / "prior_incident_claim_summary.csv"
        if file2.exists():
            st.dataframe(pd.read_csv(file2), use_container_width=True)
            
    with col2:
        st.write("**Control Score Claim Loss Summary**")
        file3 = EDA_DIR / "control_score_claim_loss_summary.csv"
        if file3.exists():
            st.dataframe(pd.read_csv(file3), use_container_width=True)
            
        st.write("**Vendor Pressure Summary**")
        file4 = EDA_DIR / "vendor_pressure_summary.csv"
        if file4.exists():
            st.dataframe(pd.read_csv(file4), use_container_width=True)

# --- Model Outputs Tab ---
with tab_model:
    st.header("Model Outputs & Diagnostics")
    
    # Model Diagnostics Text
    diag_file = MODEL_DIR / "model_diagnostics.txt"
    if diag_file.exists():
        st.subheader("Model Diagnostics")
        with open(diag_file, 'r', encoding='utf-8') as f:
            diagnostics = f.read()
        st.text(diagnostics)
    
    st.markdown("---")
    
    # Predictions
    st.subheader("Premium Comparison Visualizations")
    mc_file = MODEL_DIR / "pure_premium_indications_with_mc.csv"
    if mc_file.exists():
        df_mc = pd.read_csv(mc_file)
        
        # Select relevant columns for plotting
        plot_df = df_mc[['policy_id', 'premium_usd', 'pure_premium', 'technical_premium_template', 'risk_adjusted_technical_premium']].copy()
        plot_df = plot_df.head(50) # Limit to 50 for visualization clarity
        plot_df.set_index('policy_id', inplace=True)
        
        st.write("Comparing Current Premium vs Technical Premium for Top 50 Policies")
        st.bar_chart(plot_df[['premium_usd', 'technical_premium_template', 'risk_adjusted_technical_premium']])
        
        st.subheader("Premium Indications Data")
        st.dataframe(df_mc, use_container_width=True)
    else:
        ind_file = MODEL_DIR / "pure_premium_indications.csv"
        if ind_file.exists():
            df_ind = pd.read_csv(ind_file)
            st.subheader("Premium Indications Data")
            st.dataframe(df_ind, use_container_width=True)
