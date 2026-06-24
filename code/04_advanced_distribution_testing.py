import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor, GammaRegressor, Ridge
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, mean_squared_error
import statsmodels.api as sm
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

print("="*50)
print("ADVANCED CYBER DISTRIBUTION TESTING")
print("="*50)

# 1. Load Data
try:
    df = pd.read_csv('data/07_final_modeling_dataset.csv')
    claims_df = pd.read_csv('data/02_claims.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Rebuild minimal X and y
X_raw = df[['log_revenue', 'log_employees', 'control_gap', 'vendor_risk', 'retention_ratio', 'strong_controls']]
# Scale it so statsmodels doesn't fail on convergence
X = (X_raw - X_raw.mean()) / X_raw.std()
X = sm.add_constant(X)

y_freq = df['had_claim']

# ---------------------------------------------------------------------------
# FREQUENCY: Poisson vs Negative Binomial
# ---------------------------------------------------------------------------
print("\n[1] FREQUENCY MODELING (Poisson vs. Negative Binomial)")

# Poisson (statsmodels for AIC comparison)
poisson_model = sm.GLM(y_freq, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()
print(f"Poisson GLM AIC:      {poisson_results.aic:.2f}")
print(f"Poisson GLM Deviance: {poisson_results.deviance:.2f}")

# Negative Binomial (tuning alpha)
best_aic = float('inf')
best_alpha = 0
best_nb_model = None

for alpha in np.linspace(0.1, 2.0, 10):
    try:
        nb_model = sm.GLM(y_freq, X, family=sm.families.NegativeBinomial(alpha=alpha))
        nb_results = nb_model.fit()
        if nb_results.aic < best_aic:
            best_aic = nb_results.aic
            best_alpha = alpha
            best_nb_model = nb_results
    except:
        continue

print(f"\nNegative Binomial (alpha={best_alpha:.2f}) AIC:      {best_nb_model.aic:.2f}")
print(f"Negative Binomial (alpha={best_alpha:.2f}) Deviance: {best_nb_model.deviance:.2f}")

if best_nb_model.aic < poisson_results.aic:
    print("-> Result: Negative Binomial performs BETTER than Poisson on this dataset.")
else:
    print("-> Result: Poisson performs equally well or better. (No massive overdispersion).")

# ---------------------------------------------------------------------------
# SEVERITY: Gamma vs Lognormal vs Pareto
# ---------------------------------------------------------------------------
print("\n[2] SEVERITY MODELING (Gamma vs. Lognormal)")

# Filter for severity
df_sev = df[df['had_claim'] > 0].copy()
df_sev = df_sev.merge(claims_df[['policy_id', 'gross_incurred_usd']], on='policy_id', how='left')
y_sev = df_sev['gross_incurred_usd'].dropna()
X_sev_raw = df_sev.loc[y_sev.index, ['log_revenue', 'log_employees', 'control_gap', 'vendor_risk', 'retention_ratio', 'strong_controls']]
X_sev = (X_sev_raw - X_sev_raw.mean()) / X_sev_raw.std()
X_sev = sm.add_constant(X_sev)

# Gamma GLM (Baseline)
gamma_model = sm.GLM(y_sev, X_sev, family=sm.families.Gamma(link=sm.families.links.log()))
try:
    gamma_results = gamma_model.fit()
    print(f"Gamma GLM AIC:      {gamma_results.aic:.2f}")
    print(f"Gamma GLM Deviance: {gamma_results.deviance:.2f}")
except Exception as e:
    print("Gamma GLM failed to converge.")

# Lognormal Regression
log_y = np.log(y_sev)
lognorm_model = sm.OLS(log_y, X_sev)
lognorm_results = lognorm_model.fit()
jacobian = 2 * np.sum(log_y)
lognorm_aic_adjusted = lognorm_results.aic + jacobian
print(f"Lognormal GLM AIC (Adjusted): {lognorm_aic_adjusted:.2f}")

# Pareto Tail Fitting
print("\n[3] EXTREME TAIL RISK (Raw Distribution Fit)")
shape_g, loc_g, scale_g = stats.gamma.fit(y_sev, floc=0)
shape_ln, loc_ln, scale_ln = stats.lognorm.fit(y_sev, floc=0)
b_pareto, loc_pareto, scale_pareto = stats.pareto.fit(y_sev, floc=0)

nll_g = stats.gamma.nnlf((shape_g, loc_g, scale_g), y_sev)
nll_ln = stats.lognorm.nnlf((shape_ln, loc_ln, scale_ln), y_sev)
nll_p = stats.pareto.nnlf((b_pareto, loc_pareto, scale_pareto), y_sev)

print(f"Gamma NLL:     {nll_g:.2f}")
print(f"Lognormal NLL: {nll_ln:.2f}")
print(f"Pareto NLL:    {nll_p:.2f}")

print("\nSummary of Fit: The lower the NLL, the better the distribution handles the extreme $45M+ outliers!")
print("="*50)
