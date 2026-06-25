import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

print("==================================================")
print("HAWKES PROCESS OPTIMIZATION & SIMULATION")
print("==================================================")

# 1. Load Data
try:
    df_claims = pd.read_csv('data/02_claims.csv')
    df_claims['loss_date'] = pd.to_datetime(df_claims['loss_date'])
    df_claims = df_claims.sort_values('loss_date').reset_index(drop=True)
except Exception as e:
    print(f"Error loading claims data: {e}")
    exit()

# 2. Extract Event Times (in days relative to the first event)
t_events = (df_claims['loss_date'] - df_claims['loss_date'].min()).dt.days.values
T_max = t_events[-1] + 1  # Total time window in days

# 3. Define Hawkes Negative Log-Likelihood Function
def hawkes_nll(params, t):
    mu, alpha, beta = params
    
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return np.inf

    n = len(t)
    integral_term = mu * T_max + (alpha / beta) * np.sum(1 - np.exp(-beta * (T_max - t)))
    
    log_intensity_sum = 0
    R = 0 
    for i in range(n):
        if i > 0:
            R = np.exp(-beta * (t[i] - t[i-1])) * (1 + R)
        lam_i = mu + alpha * R
        log_intensity_sum += np.log(lam_i)
        
    nll = integral_term - log_intensity_sum
    return nll

print("Fitting Hawkes Parameters via Maximum Likelihood Estimation...")
init_params = [len(t_events)/T_max, 0.05, 0.1]
bnds = ((0.01, 5.0), (0.001, 0.9), (0.01, 2.0))
res = minimize(hawkes_nll, init_params, args=(t_events,), method='L-BFGS-B', bounds=bnds)

mu_opt, alpha_opt, beta_opt = res.x
print(f"Optimization Successful: {res.success}")
print(f"-> Baseline (mu):      {mu_opt:.4f} attacks/day")
print(f"-> Excitation (alpha): {alpha_opt:.4f}")
print(f"-> Decay (beta):       {beta_opt:.4f}")

# 4. Simulate 50,000 Portfolio Years
print("\nRunning 50,000 Year Stochastic Contagion Simulation...")

sev_data = df_claims['gross_incurred_usd'].dropna().values
shape_g, loc_g, scale_g = stats.gamma.fit(sev_data, floc=0)

num_sims = 50000
annual_losses_hawkes = np.zeros(num_sims)
annual_losses_poisson = np.zeros(num_sims)

poisson_rate = mu_opt / (1 - (alpha_opt / beta_opt))
branching_ratio = alpha_opt / beta_opt
expected_events_yr = 365 * poisson_rate

for i in range(num_sims):
    # Poisson (Independent) Simulation
    n_poisson = np.random.poisson(expected_events_yr)
    if n_poisson > 0:
        annual_losses_poisson[i] = np.sum(np.random.gamma(shape_g, scale_g, n_poisson))
        
    # Hawkes (Contagious) Simulation via Branching Approximation
    n_immigrants = np.random.poisson(365 * mu_opt)
    
    var_cluster = branching_ratio / ((1 - branching_ratio)**3)
    p = (1 / (1 - branching_ratio)) / var_cluster
    r = ((1 / (1 - branching_ratio))**2) / (var_cluster - (1 / (1 - branching_ratio)))
    if r <= 0:
        n_hawkes = n_immigrants
    else:
        n_hawkes = np.sum(np.random.negative_binomial(r, p, n_immigrants) + 1)
        
    if n_hawkes > 0:
        annual_losses_hawkes[i] = np.sum(np.random.gamma(shape_g, scale_g, int(n_hawkes)))

# 5. Calculate TVaR
p99_hawkes = np.percentile(annual_losses_hawkes, 99)
tvar_hawkes = np.mean(annual_losses_hawkes[annual_losses_hawkes >= p99_hawkes])

p99_poisson = np.percentile(annual_losses_poisson, 99)
tvar_poisson = np.mean(annual_losses_poisson[annual_losses_poisson >= p99_poisson])

print(f"\n[Poisson TVaR 99%]: ${tvar_poisson:,.2f}")
print(f"[Hawkes TVaR 99%]:  ${tvar_hawkes:,.2f}")
print(f"-> Contagion Risk Premium: +${tvar_hawkes - tvar_poisson:,.2f}")

hawkes_results = {
    "mu": mu_opt,
    "alpha": alpha_opt,
    "beta": beta_opt,
    "branching_ratio": branching_ratio,
    "tvar_poisson": tvar_poisson,
    "tvar_hawkes": tvar_hawkes,
    "contagion_premium": tvar_hawkes - tvar_poisson
}

with open('outputs/model_outputs/hawkes_results.json', 'w') as f:
    json.dump(hawkes_results, f)

print("\nResults saved to outputs/model_outputs/hawkes_results.json")
