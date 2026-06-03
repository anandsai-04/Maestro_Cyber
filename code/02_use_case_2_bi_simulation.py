"""
Use Case 2: Ransomware BI Pricing — Starter Monte Carlo Skeleton
==================================================================

This starter shows the bones of a BI Monte Carlo pricing engine:
  1. Per-system downtime distributions (bimodal mixture)
  2. Time-of-occurrence revenue multiplier
  3. Regulatory cost overlay (stepwise by downtime)
  4. Simulation -> expected BI loss -> technical premium

The intern team should:
  - Replace stub distributions with parameters fit from outage_events.csv
  - Add multi-system correlation (single ransomware event hits many systems)
  - Add reinsurance / layer attachment for tower pricing
  - Wrap in a function callable per insured for the dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# ============================================================
# Load
# ============================================================
policies = pd.read_csv(DATA_DIR / "01_policies.csv")
outages = pd.read_csv(DATA_DIR / "06_outage_events.csv")
systems = pd.read_csv(DATA_DIR / "05_system_recovery_profiles.csv")

# ============================================================
# Step 1: Fit downtime distributions per system class
# ============================================================
print("="*60)
print("STEP 1: DOWNTIME DISTRIBUTION FITTING")
print("="*60)

# Group historical outages by affected system
downtime_by_system = outages.groupby("systems_affected")["total_downtime_hours"].describe()
print("\nObserved downtime stats by system:")
print(downtime_by_system.round(1))

# Fit a simple lognormal per system as a baseline.
# REAL TASK: Fit a bimodal mixture (clean-recovery vs. rebuild)
# using e.g. scipy.stats or a GaussianMixture on log(downtime).
downtime_params = {}
for sys_class, grp in outages.groupby("systems_affected"):
    log_dt = np.log(grp["total_downtime_hours"].clip(lower=1))
    downtime_params[sys_class] = {"mu": log_dt.mean(), "sigma": log_dt.std()}

print("\nFitted lognormal params (placeholder — switch to mixture):")
for k, v in downtime_params.items():
    print(f"  {k:25s}  mu={v['mu']:.2f}  sigma={v['sigma']:.2f}")

# ============================================================
# Step 2: Time-of-occurrence revenue multiplier
# ============================================================
TIME_MULTIPLIERS = {
    "regular": 1.00,
    "month_end": 1.40,
    "quarter_end": 1.85,
    "weekend": 0.55,  # markets closed, lower revenue impact for some sub-sectors
}

def sample_time_multiplier():
    """Approximate distribution of when outages occur in a calendar year."""
    r = np.random.random()
    if r < 0.04:   return TIME_MULTIPLIERS["quarter_end"]
    elif r < 0.18: return TIME_MULTIPLIERS["month_end"]
    elif r < 0.46: return TIME_MULTIPLIERS["weekend"]
    else:          return TIME_MULTIPLIERS["regular"]

# ============================================================
# Step 3: Regulatory cost overlay (stepwise)
# ============================================================
def regulatory_overlay(downtime_hours, regulator, revenue_mm):
    """Returns additional cost triggered by regulatory thresholds."""
    cost = 0
    # 36-hour notification rule (federal banking agencies)
    if regulator in ("OCC", "FRB", "FDIC") and downtime_hours > 36:
        cost += 250_000  # notification + initial examiner response
    # Extended outage triggers heightened supervision
    if downtime_hours > 168:  # one week
        cost += 1_500_000 + (revenue_mm * 200)
    if downtime_hours > 336:  # two weeks
        cost += 5_000_000  # consent order risk reserve
    return cost

# ============================================================
# Step 4: Monte Carlo per insured
# ============================================================
def simulate_bi_loss(policy_row, n_sims=10_000, seed=None):
    """Run Monte Carlo simulation of annual BI loss for one insured."""
    if seed is not None:
        np.random.seed(seed)

    rev_daily = (policy_row["revenue_mm"] * 1_000_000) / 252  # business days

    # Annual frequency of a material ransomware event
    # Crude prior — should be replaced with output of the UC1 frequency model
    annual_freq = 0.10 * (5 - policy_row["control_maturity_nist"]) / 2.5

    losses = np.zeros(n_sims)

    for i in range(n_sims):
        n_events = np.random.poisson(annual_freq)
        sim_loss = 0
        for _ in range(n_events):
            # Pick which system is hit (weighted by criticality)
            sys_class = np.random.choice(list(downtime_params.keys()))
            params = downtime_params[sys_class]

            # Sample downtime
            downtime = np.random.lognormal(params["mu"], params["sigma"])

            # Lost revenue during downtime
            time_mult = sample_time_multiplier()
            revenue_dep = 0.35  # placeholder — pull from systems table per system class
            lost_rev = (downtime / 24) * rev_daily * revenue_dep * time_mult

            # Extra expense + forensics + restoration (rough multipliers)
            extra_expense = lost_rev * 0.35
            forensics = min(2_500_000, lost_rev * 0.12)
            restoration = downtime * 4_000  # $4k/hr response surge

            # Regulatory overlay
            reg_cost = regulatory_overlay(
                downtime, policy_row["primary_regulator"], policy_row["revenue_mm"]
            )

            event_loss = lost_rev + extra_expense + forensics + restoration + reg_cost
            sim_loss += event_loss

        losses[i] = sim_loss

    return losses

# ============================================================
# Step 5: Apply to a sample of policies and price
# ============================================================
print("\n" + "="*60)
print("STEP 5: PRICING SAMPLE INSUREDS")
print("="*60)

# Sample a few representative policies — one per sub-sector
sample_pols = (
    policies.sort_values("policy_id")
    .groupby("sub_sector", group_keys=False, as_index=False)
    .head(1)
    .head(6)
)

risk_load = 0.30  # 30% load above expected loss
expense_ratio = 0.25  # acquisition + admin

results = []
for _, pol in sample_pols.iterrows():
    sim_losses = simulate_bi_loss(pol, n_sims=5_000, seed=42)
    expected_loss = sim_losses.mean()
    p95_loss = np.percentile(sim_losses, 95)
    p99_loss = np.percentile(sim_losses, 99)
    technical_premium = expected_loss * (1 + risk_load) / (1 - expense_ratio)
    results.append({
        "policy_id": pol["policy_id"],
        "sub_sector": pol["sub_sector"],
        "revenue_mm": pol["revenue_mm"],
        "control_maturity": pol["control_maturity_nist"],
        "expected_bi_loss": int(expected_loss),
        "p95_bi_loss": int(p95_loss),
        "p99_bi_loss": int(p99_loss),
        "bi_technical_premium": int(technical_premium),
        "charged_premium": pol["premium_usd"],
    })

results_df = pd.DataFrame(results)
print("\nBI Pricing Output (Sample):")
print(results_df.to_string(index=False))

results_df.to_csv(DATA_DIR / "08_bi_pricing_output_sample.csv", index=False)
print(f"\nWrote {len(results_df)} sample outputs.")

print("\n" + "="*60)
print("NEXT STEPS FOR INTERN TEAM")
print("="*60)
print("""
1. Replace the lognormal downtime fit with a 2-component mixture model
   (sklearn.mixture.GaussianMixture on log-downtime)
2. Pull revenue_dependency_pct from systems_recovery_profiles per system
   instead of the hardcoded 0.35
3. Build correlation structure: when ransomware hits a bank, it typically
   takes down multiple systems simultaneously, not one in isolation
4. Calibrate annual_freq using the output of the UC1 frequency model
   instead of the crude maturity-based formula
5. Add layer attachment: simulate ground-up losses, then apply attachment
   point and limit to price specific tower layers
6. Vectorize the inner loop — current code is slow at production volumes
7. Build a Streamlit or Dash dashboard that shows:
   - Loss distribution histogram per insured
   - VaR / TVaR at 95% and 99%
   - Premium decomposition (expected loss + risk load + expense)
   - Sensitivity sliders (control maturity, RTO, regulator)
""")
