"""
Use Case 1: Cyber Risk Pricing Model — Starter EDA Script
============================================================
Run from: /home/claude/intern_package/

This script walks through:
  1. Loading all six datasets
  2. Basic data quality checks
  3. Exposure & loss summary statistics
  4. Loss ratio and frequency/severity decomposition
  5. Identifying baseline rating factors

Intern team: extend this script into a Jupyter notebook with visualizations,
add the modeling pipeline (frequency model, severity model, premium calculator),
and produce a model output table that joins back to policies.r

"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
from pathlib import Path
import plotly.express as px


DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================
# 1. Load Data
# ============================================================
print("Loading datasets...")
policies = pd.read_csv(DATA_DIR / "01_policies.csv", parse_dates=["inception_date", "expiration_date"])
claims = pd.read_csv(DATA_DIR / "02_claims.csv", parse_dates=["loss_date"])
findings = pd.read_csv(DATA_DIR / "03_regulatory_findings.csv", parse_dates=["exam_date"])
responses = pd.read_csv(DATA_DIR / "04_questionnaire_responses.csv")
systems = pd.read_csv(DATA_DIR / "05_system_recovery_profiles.csv")
outages = pd.read_csv(DATA_DIR / "06_outage_events.csv", parse_dates=["outage_start_date"])

print(f"  policies:  {len(policies):>6,} rows")
print(f"  claims:    {len(claims):>6,} rows")
print(f"  findings:  {len(findings):>6,} rows")
print(f"  responses: {len(responses):>6,} rows")
print(f"  systems:   {len(systems):>6,} rows")
print(f"  outages:   {len(outages):>6,} rows")

# ============================================================
# 2. DATA QUALITY & SANITY CHECKS
# ============================================================
print("--- 2. Executing Actuarial Quality Checks ---")

# Check for un-billable policies
missing_prem = policies['premium_usd'].isna().sum()
print(f"Policies missing premium data: {missing_prem}")

# Claim Math Check (BI + Non-BI must equal Gross Incurred)
mismatch = claims[claims['bi_loss_usd'].fillna(0) + claims['non_bi_loss_usd'].fillna(0) != claims['gross_incurred_usd']]
print(f"Claims where BI + Non-BI != Gross Loss: {len(mismatch)}")

# Aggregate claims up to the Policy Level
claim_agg = claims.groupby('policy_id').agg(
    n_claims=('claim_id', 'count'),
    total_loss=('gross_incurred_usd', 'sum')
).reset_index()

# Merge targets back to the main policy dataframe
model_df = policies.merge(claim_agg, on='policy_id', how='left').fillna({'n_claims': 0, 'total_loss': 0})
model_df['had_claim'] = (model_df['n_claims'] > 0).astype(int)

# ============================================================
# 3. HIGH-SPEED FEATURE ENGINEERING & INTENSIVE EDA PREP
# ============================================================
print("--- 3. Engineering Domain Signals & Actuarial Metrics ---")

# A. Missingness Imputation
model_df['mfa_coverage_pct'] = model_df['mfa_coverage_pct'].fillna(0)
model_df['n_third_party_vendors'] = model_df['n_third_party_vendors'].fillna(0)

# B. Target Transformation (C-level NumPy speed)
model_df['log_revenue'] = np.log1p(model_df['revenue_mm'])
model_df['log_employees'] = np.log1p(model_df['employees'])

# C. Actuarial Ratios & Domain Proxies
model_df['control_gap'] = 5.0 - model_df['control_maturity_nist']
model_df['vendor_risk'] = model_df['n_third_party_vendors'] * (1 - (model_df['mfa_coverage_pct'] / 100.0))
model_df['retention_ratio'] = model_df['retention_mm'] / model_df['limit_mm'].clip(lower=1)

# Intensive EDA Addition: Calculate Policy Loss Ratio
model_df['loss_ratio'] = model_df['total_loss'] / model_df['premium_usd'].clip(lower=1)

# D. Regulatory Mismatch Logic
expected_regulators = {'Regional Bank': ['FDIC', 'FRB', 'OCC'], 'Broker-Dealer': ['SEC', 'FINRA']}
def check_mismatch(row):
    sector, reg = row.get('sub_sector', ''), row.get('primary_regulator', '')
    if sector in expected_regulators:
        return 0 if reg in expected_regulators[sector] else 1
    return 0
model_df['regulator_mismatch'] = model_df.apply(check_mismatch, axis=1)

# E. Boolean Composites
bool_cols = ['edr_deployed', 'soc_24_7', 'has_trading_desk']
for col in bool_cols:
    if col in model_df.columns:
        model_df[col] = model_df[col].astype(int)
model_df['strong_controls'] = ((model_df['edr_deployed'] == 1) & (model_df['soc_24_7'] == 1)).astype(int)

# ============================================================
# 4. INTENSIVE 6-PANEL PLOTLY DASHBOARD (BUG-FIXED)
# ============================================================
print("--- 4. Generating Interactive 6-Panel EDA Dashboard ---")

# Pre-calculate Sector Metrics
sector_metrics = model_df.groupby('sub_sector').agg(
    avg_freq=('had_claim', 'mean'),
    median_sev=('total_loss', lambda x: x[x>0].median() if len(x[x>0]) > 0 else 0),
    avg_loss_ratio=('loss_ratio', 'mean')
).reset_index().sort_values(by='avg_loss_ratio', ascending=False)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        "1. Claim Frequency (Zero-Inflated)",
        "2. Claim Severity (Log10 Tail)",
        "3. Average Loss Ratio by Sector",
        "4. Risk Landscape: Frequency vs Median Severity",
        "5. Validation: Vendor Risk vs Claims",
        "6. Validation: Regulatory Mismatch Penalty"
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

# --- PANEL 1: Frequency Histogram ---
fig.add_trace(go.Histogram(x=model_df['n_claims'], marker_color='#3498db', name="Policy Count"), row=1, col=1)

# --- PANEL 2: Severity Histogram (The Bug Fix) ---
severity_df = model_df[model_df['total_loss'] > 0].copy()
# Manually apply the log to bypass the Plotly rendering bug
severity_df['log10_loss'] = np.log10(severity_df['total_loss']) 

fig.add_trace(go.Histogram(x=severity_df['log10_loss'], nbinsx=40, marker_color='#9b59b6', name="Claims"), row=1, col=2)
fig.update_xaxes(title_text="Total Loss USD (Base-10 Log Scale)", row=1, col=2)

# --- PANEL 3: Loss Ratio by Sector (Bar) ---
fig.add_trace(
    go.Bar(x=sector_metrics['sub_sector'], y=sector_metrics['avg_loss_ratio'], 
           marker=dict(color=sector_metrics['avg_loss_ratio'], colorscale='Reds'), name="Loss Ratio"),
    row=2, col=1
)
fig.update_yaxes(title_text="Avg Loss Ratio", row=2, col=1)

# --- PANEL 4: Risk Landscape (Bubble Scatter) ---
fig.add_trace(
    go.Scatter(
        x=sector_metrics['avg_freq'], 
        y=sector_metrics['median_sev'], 
        mode='markers+text',
        text=sector_metrics['sub_sector'],
        textposition='top center',
        marker=dict(size=12, color=sector_metrics['avg_loss_ratio'], colorscale='Reds', showscale=True),
        name="Sector Risk"
    ),
    row=2, col=2
)
fig.update_xaxes(title_text="Probability of Claim", row=2, col=2)
fig.update_yaxes(title_text="Median Severity USD", row=2, col=2)

# --- PANEL 5: Vendor Risk (Box Plot) ---
claims_no = model_df[model_df['had_claim'] == 0]['vendor_risk']
claims_yes = model_df[model_df['had_claim'] == 1]['vendor_risk']
fig.add_trace(go.Box(y=claims_no, name="No Claim", marker_color='#2ecc71'), row=3, col=1)
fig.add_trace(go.Box(y=claims_yes, name="Had Claim", marker_color='#e74c3c'), row=3, col=1)
fig.update_yaxes(title_text="Vendor Risk Score", row=3, col=1)

# --- PANEL 6: Regulatory Mismatch vs Claims (Added Text Labels) ---
mismatch_risk = model_df.groupby('regulator_mismatch')['had_claim'].mean().reset_index()
mismatch_risk['regulator_mismatch'] = mismatch_risk['regulator_mismatch'].map({0: 'Matched', 1: 'Mismatched'})
fig.add_trace(
    go.Bar(
        x=mismatch_risk['regulator_mismatch'], 
        y=mismatch_risk['had_claim'], 
        text=np.round(mismatch_risk['had_claim'], 3), # Show the exact math
        textposition='auto',
        marker_color=['#95a5a6', '#c0392b'], 
        name="Reg Risk"
    ),
    row=3, col=2
)
fig.update_yaxes(title_text="Probability of Claim", row=3, col=2)

# Layout Formatting
fig.update_layout(
    title_text="Comprehensive Cyber Risk Actuarial Diagnostics",
    title_font_size=24,
    height=1100, 
    showlegend=False,
    template="plotly_white"
)
fig.show()


# ============================================================
# 5. HIGH-SPEED ENCODING & THE NUMPY HANDOFF
# ============================================================
print("\n--- 5. Securing matrices and passing to NumPy ---")

# Prevent Leakage
leakage_cols = ['insured_id', 'inception_date', 'expiration_date', 'total_loss', 'n_claims', 'bi_loss', 'loss_ratio']
df_clean = model_df.drop(columns=[col for col in leakage_cols if col in model_df.columns])

# High-Speed Label Encoding (.cat.codes)
cat_cols = ['sub_sector', 'region', 'primary_regulator', 'core_banking_vendor', 'cloud_provider_primary']
for col in cat_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype('category').cat.codes

# Standardize names for XGBoost
df_clean.columns = [col.replace(' ', '_').replace('-', '_').lower() for col in df_clean.columns]

# The NumPy Switch
y_freq = df_clean['had_claim'].to_numpy()
X_matrix = df_clean.drop(columns=['policy_id', 'had_claim']).to_numpy()

print(f"\n✅ FULL PIPELINE COMPLETE")
print(f"Features (X): {X_matrix.shape} -> Dense NumPy Array")
print(f"Target (y):   {y_freq.shape}")

# ============================================================
# 4. INTENSIVE 8-PANEL PLOTLY DASHBOARD (INCLUDING NLP)
# ============================================================
print("--- 4. Generating Interactive 8-Panel EDA Dashboard ---")

# Pre-calculate Sector Metrics
sector_metrics = model_df.groupby('sub_sector').agg(
    avg_freq=('had_claim', 'mean'),
    median_sev=('total_loss', lambda x: x[x>0].median() if len(x[x>0]) > 0 else 0),
    avg_loss_ratio=('loss_ratio', 'mean')
).reset_index().sort_values(by='avg_loss_ratio', ascending=False)

# Pre-calculate NLP Tiers (Binning the continuous probability into 4 risk groups)
if 'nlp_high_risk_prob' in model_df.columns:
    model_df['nlp_risk_tier'] = pd.qcut(model_df['nlp_high_risk_prob'], q=4, duplicates='drop', labels=['Low', 'Medium', 'High', 'Severe'])
    nlp_tier_risk = model_df.groupby('nlp_risk_tier')['had_claim'].mean().reset_index()
else:
    print("Warning: nlp_high_risk_prob not found. NLP charts will be blank.")

# Expand to 4 Rows
fig = make_subplots(
    rows=4, cols=2,
    subplot_titles=(
        "1. Claim Frequency (Zero-Inflated)",
        "2. Claim Severity (Log10 Tail)",
        "3. Average Loss Ratio by Sector",
        "4. Risk Landscape: Frequency vs Median Severity",
        "5. Domain Valid: Vendor Risk vs Claims",
        "6. Domain Valid: Regulatory Mismatch Penalty",
        "7. NLP Valid: Claim Probability by Text Risk Tier",
        "8. NLP Valid: FinBERT Probability Distribution"
    ),
    vertical_spacing=0.10,
    horizontal_spacing=0.10
)

# --- PANEL 1 & 2: Targets ---
fig.add_trace(go.Histogram(x=model_df['n_claims'], marker_color='#3498db', name="Policy Count"), row=1, col=1)

severity_df = model_df[model_df['total_loss'] > 0].copy()
severity_df['log10_loss'] = np.log10(severity_df['total_loss']) 
fig.add_trace(go.Histogram(x=severity_df['log10_loss'], nbinsx=40, marker_color='#9b59b6', name="Claims"), row=1, col=2)
fig.update_xaxes(title_text="Total Loss USD (Base-10 Log Scale)", row=1, col=2)

# --- PANEL 3 & 4: Sector Risk ---
fig.add_trace(
    go.Bar(x=sector_metrics['sub_sector'], y=sector_metrics['avg_loss_ratio'], 
           marker=dict(color=sector_metrics['avg_loss_ratio'], colorscale='Reds'), name="Loss Ratio"),
    row=2, col=1
)
fig.update_yaxes(title_text="Avg Loss Ratio", row=2, col=1)

fig.add_trace(
    go.Scatter(
        x=sector_metrics['avg_freq'], y=sector_metrics['median_sev'], mode='markers+text',
        text=sector_metrics['sub_sector'], textposition='top center',
        marker=dict(size=12, color=sector_metrics['avg_loss_ratio'], colorscale='Reds', showscale=True),
        name="Sector Risk"
    ),
    row=2, col=2
)
fig.update_xaxes(title_text="Probability of Claim", row=2, col=2)
fig.update_yaxes(title_text="Median Severity USD", row=2, col=2)

# --- PANEL 5 & 6: Domain Features ---
claims_no = model_df[model_df['had_claim'] == 0]['vendor_risk']
claims_yes = model_df[model_df['had_claim'] == 1]['vendor_risk']
fig.add_trace(go.Box(y=claims_no, name="No Claim", marker_color='#2ecc71'), row=3, col=1)
fig.add_trace(go.Box(y=claims_yes, name="Had Claim", marker_color='#e74c3c'), row=3, col=1)

mismatch_risk = model_df.groupby('regulator_mismatch')['had_claim'].mean().reset_index()
mismatch_risk['regulator_mismatch'] = mismatch_risk['regulator_mismatch'].map({0: 'Matched', 1: 'Mismatched'})
fig.add_trace(
    go.Bar(
        x=mismatch_risk['regulator_mismatch'], y=mismatch_risk['had_claim'], 
        text=np.round(mismatch_risk['had_claim'], 3), textposition='auto',
        marker_color=['#95a5a6', '#c0392b'], name="Reg Risk"
    ),
    row=3, col=2
)

# --- PANEL 7 & 8: THE NEW NLP VISUALIZATIONS ---
if 'nlp_high_risk_prob' in model_df.columns:
    # Panel 7: Bar chart showing how claims jump as NLP risk gets more severe
    fig.add_trace(
        go.Bar(
            x=nlp_tier_risk['nlp_risk_tier'], y=nlp_tier_risk['had_claim'],
            text=np.round(nlp_tier_risk['had_claim'], 3), textposition='auto',
            marker=dict(color=nlp_tier_risk['had_claim'], colorscale='Purples'), name="NLP Risk Tier"
        ),
        row=4, col=1
    )
    fig.update_yaxes(title_text="Probability of Claim", row=4, col=1)
    
    # Panel 8: Box plot showing the exact FinBERT probability distributions
    nlp_no = model_df[model_df['had_claim'] == 0]['nlp_high_risk_prob']
    nlp_yes = model_df[model_df['had_claim'] == 1]['nlp_high_risk_prob']
    fig.add_trace(go.Box(y=nlp_no, name="No Claim", marker_color='#34495e'), row=4, col=2)
    fig.add_trace(go.Box(y=nlp_yes, name="Had Claim", marker_color='#8e44ad'), row=4, col=2)
    fig.update_yaxes(title_text="FinBERT High Risk Prob", row=4, col=2)

# --- Layout Formatting ---
fig.update_layout(
    title_text="Comprehensive Cyber Risk Actuarial Diagnostics (Including NLP)",
    title_font_size=24,
    height=1400, # Increased height to fit 4 rows comfortably
    showlegend=False,
    template="plotly_white"
)
fig.show()

# ============================================================
# 5. NLP Corpus Preview
# ============================================================
print("\n" + "="*60)
print("NLP CORPUS PREVIEW")
print("="*60)

print(f"\nRegulatory findings: {len(findings)} samples")
print(f"Severity label distribution:")
print(findings['severity_label'].value_counts())
print(f"\nSample HIGH severity finding:")
print("  " + findings[findings['severity_label'] == 'High'].iloc[0]['finding_text'])
print(f"\nSample LOW severity finding:")
print("  " + findings[findings['severity_label'] == 'Low'].iloc[0]['finding_text'])

# ============================================================
# 6. Build Modeling Dataset (starter)
# ============================================================
print("\n" + "="*60)
print("BUILDING MODELING DATASET")
print("="*60)

# Claim aggregation per policy
claim_agg = claims.groupby('policy_id').agg(
    n_claims=('claim_id', 'count'),
    total_loss=('gross_incurred_usd', 'sum'),
    bi_loss=('bi_loss_usd', 'sum'),
).reset_index()

# Add NLP-feedable severity counts (proxy until interns build the classifier)
finding_agg = findings.groupby('policy_id').agg(
    n_findings=('finding_id', 'count'),
    n_high_sev=('severity_label', lambda x: (x == 'High').sum()),
    n_med_sev=('severity_label', lambda x: (x == 'Medium').sum()),
).reset_index()

model_df = policies.merge(claim_agg, on='policy_id', how='left').fillna({'n_claims': 0, 'total_loss': 0, 'bi_loss': 0})
model_df = model_df.merge(finding_agg, on='policy_id', how='left').fillna({'n_findings': 0, 'n_high_sev': 0, 'n_med_sev': 0})

model_df['had_claim'] = (model_df['n_claims'] > 0).astype(int)
model_df['log_revenue'] = np.log(model_df['revenue_mm'])

# Save the prepared modeling dataset
output_path = DATA_DIR / "07_modeling_dataset.csv"
model_df.to_csv(output_path, index=False)
print(f"\nWrote modeling dataset: {output_path}")
print(f"Shape: {model_df.shape}")
print(f"Columns: {list(model_df.columns)}")

print("\n" + "="*60)
print("NEXT STEPS FOR INTERN TEAM")
print("="*60)
print("""
1. Visualize the distributions (loss severity, frequency by segment, premium adequacy by segment)
2. Build the NLP severity classifier on regulatory findings:
   - Start with TF-IDF + logistic regression as a baseline
   - Then fine-tune a small transformer (distilbert-base-uncased)
   - Compare F1 scores across both
3. Build the frequency model:
   - Target: had_claim (or n_claims for Poisson)
   - Features: log_revenue, sub_sector, control_maturity_nist, mfa_coverage_pct, n_high_sev, etc.
   - Try logistic regression baseline, then XGBoost
4. Build the severity model:
   - Two-part: attritional (GLM gamma) + tail (GPD over a threshold like $1M)
5. Combine into technical premium: E[freq] * E[severity_attritional] + tail loading
6. Compare technical premium to charged premium — identify under/over-priced cohorts
7. Document everything in a notebook with charts
""")
