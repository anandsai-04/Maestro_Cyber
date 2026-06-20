import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Try to import our simulation engine
try:
    import sys
    sys.path.append(str(Path(__file__).parent / "code"))
    from code.02_use_case_2_bi_simulation import simulate_bi_loss_vectorized, calculate_premium, downtime_models
except ImportError:
    st.error("Could not import BI Simulation engine. Ensure app_2.py is run from the root directory.")

st.set_page_config(page_title="BI Ransomware Pricing Engine", layout="wide", page_icon="🏦")
st.title("🏦 Advanced BI Ransomware Pricing Engine")
st.markdown("Use Case 2: Stochastic Monte Carlo Simulation & Cost of Capital (TVaR) Pricing")

# ============================================================
# Sidebar: User Inputs
# ============================================================
st.sidebar.header("Policyholder Profile")

revenue_mm = st.sidebar.slider("Annual Revenue ($M)", min_value=10, max_value=50000, value=500, step=10)
control_maturity = st.sidebar.slider("NIST Control Maturity (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
primary_regulator = st.sidebar.selectbox("Primary Regulator", ["OCC", "FRB", "FDIC", "SEC", "State", "None"])

st.sidebar.markdown("---")
st.sidebar.header("Actuarial Parameters")
capital_charge = st.sidebar.slider("Capital Charge Rate (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.5) / 100.0
expense_ratio = st.sidebar.slider("Expense Ratio (%)", min_value=10.0, max_value=40.0, value=25.0, step=1.0) / 100.0
n_sims = st.sidebar.selectbox("Number of Simulations (Years)", [10000, 50000, 100000], index=1)

# Build a policy row to feed the simulation
policy_input = pd.Series({
    "revenue_mm": revenue_mm,
    "control_maturity_nist": control_maturity,
    "primary_regulator": primary_regulator
})

# ============================================================
# Tabs
# ============================================================
tab_calc, tab_viz, tab_ai = st.tabs(["🧮 Interactive Pricing Calculator", "📈 System Distributions (GMM)", "🤖 AI Explainer"])

# ------------------------------------------------------------
# Tab 1: Interactive Pricing Calculator
# ------------------------------------------------------------
with tab_calc:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Run Simulation")
        st.write("Click below to run a massive vectorized Monte Carlo simulation modeling thousands of potential ransomware outage scenarios.")
        if st.button("▶️ Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Simulating {n_sims:,} years of cyber risk..."):
                sim_losses = simulate_bi_loss_vectorized(policy_input, n_sims=n_sims)
                metrics = calculate_premium(sim_losses, capital_charge=capital_charge, expense_ratio=expense_ratio)
                
            st.success("Simulation Complete!")
            
            st.markdown("### 💰 Pricing Indications")
            st.metric("Pure Premium (Expected Loss)", f"${metrics['expected_loss']:,.0f}")
            
            st.markdown("#### Actuarial Risk Adjustments")
            st.metric(f"Value at Risk (99th Percentile)", f"${metrics['p99_loss']:,.0f}")
            st.metric(f"Tail Value at Risk (TVaR 99%)", f"${metrics['tvar_99']:,.0f}")
            st.metric(f"Required Capital Base", f"${(metrics['tvar_99'] - metrics['expected_loss']):,.0f}")
            st.metric(f"Cost of Capital (Risk Load)", f"${metrics['risk_load_dollars']:,.0f}", help=f"Capital Base × {capital_charge*100}%")
            
            st.markdown("---")
            st.metric("Final Technical Premium", f"${metrics['technical_premium']:,.0f}", 
                      help=f"(Pure Premium + Risk Load) / (1 - {expense_ratio*100}% Expense Ratio)")
            
            # Save for viz and AI
            st.session_state["sim_losses"] = sim_losses
            st.session_state["metrics"] = metrics
            st.session_state["policy_input"] = policy_input

    with col2:
        st.subheader("Loss Distribution Tail Risk")
        if "sim_losses" in st.session_state:
            losses = st.session_state["sim_losses"]
            non_zero_losses = losses[losses > 0]
            
            if len(non_zero_losses) > 0:
                fig = px.histogram(
                    non_zero_losses, 
                    nbins=100, 
                    title="Simulated Ransomware BI Losses (Excluding $0 Years)",
                    labels={"value": "Loss Amount ($)", "count": "Frequency"},
                    color_discrete_sequence=['#FF4B4B']
                )
                
                # Add vertical lines for percentiles
                p95 = np.percentile(losses, 95)
                p99 = np.percentile(losses, 99)
                tvar = np.mean(losses[losses >= p99]) if len(losses[losses >= p99]) > 0 else p99
                
                fig.add_vline(x=p95, line_dash="dash", line_color="orange", annotation_text="VaR 95%")
                fig.add_vline(x=p99, line_dash="dash", line_color="red", annotation_text="VaR 99%")
                fig.add_vline(x=tvar, line_dash="solid", line_color="darkred", annotation_text="TVaR 99%")
                
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No losses generated in this simulation (highly secure profile).")
        else:
            st.info("👈 Run the simulation to see the loss distribution tail risk.")

# ------------------------------------------------------------
# Tab 2: System Distributions (GMM)
# ------------------------------------------------------------
with tab_viz:
    st.subheader("Bimodal Downtime Recovery Profiles")
    st.write("These distributions model the fact that recovering from ransomware is not a simple bell curve. Systems usually experience a fast 'clean recovery' (if backups are viable) or a long 'full rebuild' (if backups are encrypted/destroyed).")
    
    try:
        # Generate plot for each GMM model
        systems = list(downtime_models.keys())
        selected_sys = st.selectbox("Select System Class to View Downtime Distribution", systems)
        
        if selected_sys:
            gmm = downtime_models[selected_sys]
            
            # Generate points to evaluate the PDF
            x = np.linspace(0, 10, 1000).reshape(-1, 1) # Log hours
            logprob = gmm.score_samples(x)
            pdf = np.exp(logprob)
            
            # Convert x to actual hours for the x-axis
            x_hours = np.exp(x).flatten()
            
            # Filter to reasonable display range (up to 4 weeks = 672 hours)
            mask = x_hours <= 1000
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x_hours[mask], 
                y=pdf[mask], 
                mode='lines', 
                fill='tozeroy',
                name='Combined Mixture PDF',
                line=dict(color='#1f77b4', width=3)
            ))
            
            # Plot individual components
            for i, (mean, covar, weight) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
                from scipy.stats import norm
                comp_pdf = weight * norm.pdf(x, mean[0], np.sqrt(covar[0][0])).flatten()
                fig2.add_trace(go.Scatter(
                    x=x_hours[mask],
                    y=comp_pdf[mask],
                    mode='lines',
                    line=dict(dash='dash', width=2),
                    name=f'Component {i+1} (Weight: {weight:.2f})'
                ))
                
            fig2.update_layout(
                title=f"Downtime Probability Density for {selected_sys}",
                xaxis_title="Downtime (Hours)",
                yaxis_title="Probability Density",
                hovermode="x unified"
            )
            st.plotly_chart(fig2, use_container_width=True)
            
    except NameError:
        st.warning("GMM Models not loaded. Please ensure simulation script ran successfully.")

# ------------------------------------------------------------
# Tab 3: AI Explainer
# ------------------------------------------------------------
with tab_ai:
    st.subheader("🤖 Actuarial AI Explainer")
    st.write("Generate a plain-English explanation of why the calculated premium is high or low based on the exact simulation metrics and the policyholder's risk profile.")
    
    api_key = st.text_input("Enter your Gemini API Key:", type="password", help="Get a free key from Google AI Studio")
    
    if st.button("Generate Actuarial Explanation", type="primary"):
        if "metrics" not in st.session_state:
            st.warning("Please run the Monte Carlo simulation first in Tab 1!")
        elif not api_key:
            st.error("Please enter a Gemini API Key to generate the explanation.")
        else:
            with st.spinner("Analyzing simulation metrics..."):
                try:
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    
                    metrics = st.session_state["metrics"]
                    inputs = st.session_state["policy_input"]
                    
                    prompt = f"""
                    You are an expert Cyber Actuary. Explain to a non-technical underwriter why this policy's Business Interruption premium is what it is.
                    
                    Here are the inputs:
                    - Revenue: ${inputs['revenue_mm']} Million
                    - NIST Maturity Score: {inputs['control_maturity_nist']}/5
                    - Primary Regulator: {inputs['primary_regulator']}
                    
                    Here are the simulation outputs (based on 50,000 Monte Carlo years):
                    - Expected Average Loss (Pure Premium): ${metrics['expected_loss']:,.0f}
                    - TVaR 99% (Worst 1% of simulated years): ${metrics['tvar_99']:,.0f}
                    - Risk Load (Cost to hold reserve capital): ${metrics['risk_load_dollars']:,.0f}
                    - Final Technical Premium: ${metrics['technical_premium']:,.0f}
                    
                    Focus your explanation on HOW the inputs (especially the NIST maturity and regulator) influenced the TVaR 99% tail risk, and why the Risk Load is causing the final premium to be much higher than just the "average" expected loss.
                    Be concise, professional, and act as an actuarial co-pilot.
                    """
                    
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt,
                    )
                    
                    st.success("Analysis Complete")
                    st.markdown("### AI Actuarial Assessment")
                    st.write(response.text)
                    
                except ImportError:
                    st.error("The `google-genai` library is not installed. Please run `pip install google-genai`.")
                except Exception as e:
                    st.error(f"Error generating explanation: {e}")
