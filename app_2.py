import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Portfolio BI Analytics", layout="wide", page_icon="📈")

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
    
    /* Glassmorphism Cards */
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
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #38bdf8;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
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

# Ensure required libraries
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Please run `pip install google-genai` to use the AI Agent features.")
    st.stop()

# ============================================================
# 1. Load & Preprocess Data (Deterministic Steps)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data/09_cyber_pricing_features.csv")
    if "had_claim" in df.columns:
        df = df.drop(columns=["had_claim"])
    
    # Calculate deterministic correlations for LLM
    numeric_cols = [
        'cyber_control_score', 'vendor_control_pressure', 'control_gap_score',
        'regulatory_findings_pressure', 'high_sev_rate', 'limit_to_revenue', 
        'prior_incident_score', 'exposure_size_score', 'premium_usd'
    ]
    # Filter numeric cols that actually exist
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    corr_bi = df[numeric_cols + ['bi_loss']].corr()['bi_loss'].drop('bi_loss').to_dict()
    corr_lr = df[numeric_cols + ['loss_ratio']].corr()['loss_ratio'].drop('loss_ratio').to_dict()
    
    # Groupings
    regulator_stats = df.groupby('primary_regulator')[['loss_ratio', 'bi_loss', 'n_claims', 'total_loss']].mean().round(2).to_dict('index')
    vendor_band_stats = df.groupby('vendor_pressure_band')[['loss_ratio', 'bi_loss']].mean().round(2).to_dict('index')
    sector_stats = df.groupby('sub_sector')[['loss_ratio', 'bi_loss', 'premium_usd']].mean().round(2).to_dict('index')
    cloud_stats = df.groupby('cloud_provider_primary')[['loss_ratio', 'bi_loss']].mean().round(2).to_dict('index')
    core_banking_stats = df.groupby('core_banking_vendor')[['loss_ratio', 'bi_loss']].mean().round(2).to_dict('index')
    
    # Overall averages
    overall = {
        "avg_premium": df['premium_usd'].mean(),
        "avg_loss_ratio": df['loss_ratio'].mean(),
        "avg_bi_loss": df['bi_loss'].mean(),
        "avg_bi_share_of_loss": df['bi_share_of_loss'].mean(),
        "total_policies": len(df)
    }
    
    # Load SHAP global feature importances
    try:
        shap_df = pd.read_csv("outputs/model_outputs/shap_importances.csv")
        shap_importances = shap_df.set_index('Feature')['SHAP_Importance'].to_dict()
    except FileNotFoundError:
        shap_importances = {"Error": "SHAP importances not generated yet."}
    
    # Condense stats for LLM
    condensed_stats = {
        "overall": overall,
        "correlations_with_bi_loss": corr_bi,
        "correlations_with_loss_ratio": corr_lr,
        "regulator_impact": regulator_stats,
        "vendor_pressure_impact": vendor_band_stats,
        "sector_impact": sector_stats,
        "cloud_provider_impact": cloud_stats,
        "core_banking_vendor_impact": core_banking_stats,
        "AI_SHAP_Feature_Importances": shap_importances
    }
    
    return df, condensed_stats

df, stats = load_data()

st.title("🛡️ Use Case 2: AI Portfolio & BI Explainer")
st.markdown("*Premium Analytics Engine with Gemini ADK Integration*")

tab1, tab2 = st.tabs(["🤖 AI Portfolio Overview", "🧬 Feature Derivation Explainer"])

# ============================================================
# Tab 1: Overview & Agent Chat
# ============================================================
with tab1:
    
    st.markdown(f"""
    <div class="glass-card" style="display: flex; justify-content: space-around;">
        <div>
            <div class="metric-title">Total Policies</div>
            <div class="metric-value">{stats['overall']['total_policies']:,}</div>
        </div>
        <div>
            <div class="metric-title">Avg Loss Ratio</div>
            <div class="metric-value">{stats['overall']['avg_loss_ratio']:.2%}</div>
        </div>
        <div>
            <div class="metric-title">Avg BI Share</div>
            <div class="metric-value">{stats['overall']['avg_bi_share_of_loss']:.2%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Data Visualizations")
        fig_lr = px.box(df, x='vendor_pressure_band', y='loss_ratio', color='primary_regulator', 
                        title="Loss Ratio by Vendor Pressure", template="plotly_dark")
        fig_lr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_lr, use_container_width=True)
        
        fig_bi = px.scatter(df, x='cyber_control_score', y='bi_loss', size='premium_usd', color='vendor_pressure_band', 
                            title="BI Loss vs Cyber Control Score", template="plotly_dark")
        fig_bi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bi, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🤖 AI Actuarial Report")
        
        api_key = st.text_input("Enter Gemini API Key to run Agent:", type="password", key="agent_key")
        
        if api_key:
            client = genai.Client(api_key=api_key)
            
            # Auto-generate report if not in session state
            if "agent_report" not in st.session_state:
                with st.spinner("Agent is analyzing deterministic stats and SHAP importances..."):
                    prompt = f"""
                    You are an expert Chief Actuary reviewing a cyber insurance portfolio. 
                    I have already run the heavy deterministic calculations to save your tokens. 
                    Crucially, I have also included the SHAP Feature Importances from the XGBoost pricing engine.
                    
                    Here are the statistical effects and SHAP importances:
                    
                    {stats}
                    
                    Write a concise executive report for the underwriters. 
                    Explicitly study and explain the effect of:
                    1. Regulator based effects
                    2. Vendor Control Pressure / Vendor Pressure Band
                    3. Cyber Control Score and Control Gap Score
                    4. Cloud Providers and Core Banking Vendors
                    5. Regulatory Finding Pressure and High Severity Rate (NLP output)
                    6. Limit to Revenue and Prior Incidents
                    
                    Explain how severely these factors affect BI Loss and Loss Ratio. Compare Loss Ratio to the BI share of loss ratio.
                    **CRITICAL:** Explicitly use the AI_SHAP_Feature_Importances to explain *why* the AI Pricing Engine cares about certain features over others.
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt
                    )
                    st.session_state["agent_report"] = response.text
                    
                    # Initialize chat history
                    st.session_state.messages = [
                        {"role": "model", "content": "I have completed the portfolio analysis. You can ask me follow-up questions about the correlations, specific vendors, or risk drivers below."}
                    ]
            
            st.markdown(st.session_state["agent_report"])
            
        else:
            st.info("Provide API key to automatically generate the AI report.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface at the bottom
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("💬 Chat with the Actuarial Agent")
    
    if api_key and "messages" in st.session_state:
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message("assistant" if msg["role"] == "model" else "user"):
                st.write(msg["content"])
                
        # Chat input
        if prompt_input := st.chat_input("Ask a follow-up question about the portfolio..."):
            st.session_state.messages.append({"role": "user", "content": prompt_input})
            with st.chat_message("user"):
                st.write(prompt_input)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat = client.chats.create(model="gemini-2.5-flash")
                    # Pre-load context
                    chat.send_message(f"Here is the context of the portfolio stats: {stats}. Here is the report you just wrote: {st.session_state['agent_report']}. Answer the user's next question based on this.")
                    
                    response = chat.send_message(prompt_input)
                    st.write(response.text)
            
            st.session_state.messages.append({"role": "model", "content": response.text})
    elif not api_key:
        st.warning("API key required to use the chat functionality.")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# Tab 2: Feature Engineering Explainer
# ============================================================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🧬 How We Derived the Features")
    
    st.markdown("### 1. Merged Features (Mathematical Engineering)")
    st.info("The raw dataset contained hundreds of disparate security and policy variables. We mathematically condensed these into actionable 'Merged Features'.")
    st.markdown("""
    * **`vendor_control_pressure`:** We evaluated the number of third-party vendors (`n_third_party_vendors`) combined with their access levels and the presence of multi-factor authentication (`mfa_coverage_pct`). High vendor count with low MFA creates immense systemic 'pressure'.
    * **`vendor_pressure_band`:** A categorical binning of the above score into 'Low', 'Medium', 'High', and 'Critical' to make it easy for underwriters to segment risk.
    * **`control_gap_score`:** We took the baseline expected `cyber_control_score` for a company of a specific revenue size and sector, and subtracted their actual NIST score. A positive gap means they are lagging behind their peers.
    * **`limit_to_revenue`:** Simple financial ratio: `limit_mm / revenue_mm`. High ratios indicate severe moral hazard or over-insurance.
    * **`bi_share_of_loss`:** `bi_loss / total_loss`. Shows how much of the overall claim severity is purely driven by business interruption versus forensics or ransoms.
    """)
    
    st.markdown("### 2. Features Extracted from Text Data (NLP Engine)")
    st.success("Cyber risk is often buried in unstructured auditor reports. We deployed an AI NLP pipeline to read this text and extract quantitative risk scores.")
    st.markdown("""
    * **The NLP Engine:** We used **DistilBERT**, a powerful transformer-based language model.
    * **`high_sev_rate`:** DistilBERT reads the raw text from `regulatory_findings` (e.g., *"Critical vulnerability found in core banking database..."*). It generates 768-dimensional vector embeddings of the text's semantic meaning. We pass these embeddings into a Random Forest classifier that predicts the probability (0% to 100%) that the finding represents a "High Severity" systemic threat. This probability becomes the `high_sev_rate` feature.
    * **`regulatory_findings_pressure`:** We combine the `high_sev_rate` with the *count* of unresolved findings to create a single pressure metric indicating the likelihood of regulatory fines and extended downtime during an incident.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
