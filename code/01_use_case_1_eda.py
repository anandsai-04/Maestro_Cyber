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
and produce a model output table that joins back to policies.
"""

import pandas as pd
import numpy as np
from pathlib import Path

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
# 2. Data Quality Checks (TODO for interns: expand this section)
# ============================================================
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

# Check for missing premium
print(f"\nPolicies with missing premium: {policies['premium_usd'].isna().sum()}")

# Check for negative losses
print(f"Claims with negative gross_incurred: {(claims['gross_incurred_usd'] < 0).sum()}")

# Check for impossible BI splits
mismatch = claims[claims['bi_loss_usd'] + claims['non_bi_loss_usd'] != claims['gross_incurred_usd']]
print(f"Claims with BI+non-BI != gross: {len(mismatch)}")

# Check regulator-subsector consistency (intentional data quality issue)
print("\nRegulator distribution by sub-sector (look for anomalies):")
print(pd.crosstab(policies['sub_sector'], policies['primary_regulator']).iloc[:5, :5])

# ============================================================
# 3. Exposure & Loss Summary
# ============================================================
print("\n" + "="*60)
print("EXPOSURE & LOSS SUMMARY")
print("="*60)

total_premium = policies['premium_usd'].sum()
total_losses = claims['gross_incurred_usd'].sum()
loss_ratio = total_losses / total_premium

print(f"\nTotal earned premium: ${total_premium:>15,.0f}")
print(f"Total incurred loss:  ${total_losses:>15,.0f}")
print(f"Aggregate loss ratio:  {loss_ratio:>15.1%}")

print("\nPremium and losses by sub-sector:")
prem_by_seg = policies.groupby('sub_sector').agg(
    n_policies=('policy_id', 'count'),
    avg_revenue_mm=('revenue_mm', 'mean'),
    total_premium=('premium_usd', 'sum'),
)
loss_by_seg = (
    claims.merge(policies[['policy_id', 'sub_sector']], on='policy_id')
    .groupby('sub_sector')
    .agg(n_claims=('claim_id', 'count'), total_loss=('gross_incurred_usd', 'sum'))
)
summary = prem_by_seg.join(loss_by_seg, how='left').fillna(0)
summary['loss_ratio'] = summary['total_loss'] / summary['total_premium']
summary['frequency'] = summary['n_claims'] / summary['n_policies']
print(summary.round(3))

# ============================================================
# 4. Frequency / Severity Decomposition
# ============================================================
print("\n" + "="*60)
print("FREQUENCY & SEVERITY ANALYSIS")
print("="*60)

# Frequency: claims per policy
freq_overall = len(claims) / len(policies)
print(f"\nOverall annual claim frequency: {freq_overall:.3f}")

# Severity distribution (note the heavy tail)
sev = claims['gross_incurred_usd']
print(f"\nSeverity distribution ($USD):")
print(f"  count:  {len(sev):>15,}")
print(f"  mean:   {sev.mean():>15,.0f}")
print(f"  median: {sev.median():>15,.0f}")
print(f"  p75:    {sev.quantile(0.75):>15,.0f}")
print(f"  p90:    {sev.quantile(0.90):>15,.0f}")
print(f"  p95:    {sev.quantile(0.95):>15,.0f}")
print(f"  p99:    {sev.quantile(0.99):>15,.0f}")
print(f"  max:    {sev.max():>15,.0f}")
print("\nNote the heavy right tail — this is exactly why a single GLM under-prices")
print("excess layers. The intern team should investigate splitting attritional vs.")
print("tail severity (e.g. lognormal + GPD).")

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
