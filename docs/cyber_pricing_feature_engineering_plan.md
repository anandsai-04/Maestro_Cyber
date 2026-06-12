# Cyber Pricing Feature Engineering Plan

## Purpose

This folder now has a first-pass pricing workflow for cyber insurance:

1. Build a clean, pruned feature dataset.
2. Generate EDA visualizations focused on pricing signals.
3. Train a transparent frequency-severity template.
4. Calculate pure premium and a loaded technical premium indication.

## Run Order

From the project root in VS Code:

```bash
python code/01_use_case_1_eda.py
```

If your default Python does not have `pandas`, use the bundled/project environment that has it installed.

## Main Outputs

- `data/09_cyber_pricing_features.csv`
- `data/09_feature_dictionary.csv`
- `outputs/eda_visuals/cyber_pricing_eda_dashboard.html`
- `outputs/eda_visuals/sector_claim_loss_premium_summary.csv`
- `outputs/eda_visuals/prior_incident_claim_summary.csv`
- `outputs/eda_visuals/control_score_claim_loss_summary.csv`
- `outputs/eda_visuals/vendor_pressure_summary.csv`
- `outputs/model_outputs/pure_premium_indications.csv`
- `outputs/model_outputs/model_diagnostics.txt`

## Redundant Variables Merged

The raw dataset has several variables that are useful but correlated or overlapping. For model stability and easier interpretation, the first template collapses them:

| Combined feature | Inputs merged | Why |
|---|---|---|
| `exposure_size_score` | `log_revenue`, `log_assets`, `log_employees` | Revenue, assets, and employees are all company-size proxies. |
| `cyber_control_score` | `control_maturity_nist`, `mfa_coverage_pct`, `edr_deployed`, `soc_24_7` | Represents overall security posture. |
| `vendor_control_pressure` | `n_third_party_vendors`, `control_maturity_nist` | Captures the high-vendor, weak-control interaction. |
| `regulatory_findings_pressure` | `n_findings`, `n_high_sev`, `n_med_sev` | Preserves findings volume and severity mix in one score. |
| `critical_operations_score` | `has_trading_desk`, `processes_payments`, `has_custodial_aum` | Captures operational criticality. |
| `coverage_structure_score` | `limit_mm`, `retention_mm`, `revenue_mm` | Captures coverage adequacy and self-insured retention structure. |
| `prior_incident_score` | `prior_incidents_3yr` | Smooths prior incident counts for modeling. |

## Variables Excluded From Predictors

These are not used as predictors in the template because they are IDs, outcomes, or direct pricing outputs:

- `policy_id`
- `insured_id`
- `had_claim`
- `n_claims`
- `total_loss`
- `bi_loss`
- `premium_usd`
- `loss_ratio`

They remain in output files for auditing, comparison, and joins.

## Pricing Template

The first version uses:

```text
Expected Claim Frequency = predicted_claim_probability * exposure_years
Pure Premium = Expected Claim Frequency * Expected Claim Severity
Technical Premium = Pure Premium / (1 - total_load)
```

Current load assumptions in the template:

- Expense load: 25%
- Profit load: 10%
- Cyber catastrophe/systemic load: 12%
- Reinsurance load: 8%

These are placeholders. They should be calibrated with actuarial, underwriting, and reinsurance input before any production use.
