# Advanced Cyber Model Comparison Report

We successfully trained and tested the advanced Cyber distributions against our baseline Actuarial GLMs. Below is the detailed breakdown of how each distribution performed on the exact cyber dataset.

## 1. Frequency Modeling (Poisson vs. Negative Binomial)

We trained a Negative Binomial (NB) model by tuning the dispersion parameter ($\alpha$) to see if it could outperform our existing Poisson GLM in predicting the likelihood of a cyber attack.

| Model | AIC (Lower is Better) | Deviance | Conclusion |
| :--- | :--- | :--- | :--- |
| **Poisson GLM (Baseline)** | **1352.58** | 838.58 | **Winner** |
| **Negative Binomial ($\alpha=0.10$)** | 1357.54 | 819.33 | Worse Fit |

> [!NOTE]  
> **Why did Poisson win?** The Negative Binomial distribution is theoretically superior when data is massively "over-dispersed" (Variance > Mean). However, in this specific modeled dataset, the variance (0.1390) is actually slightly lower than the mean (0.1667). Because the data is not over-dispersed, the extra complexity of the Negative Binomial model is penalized by the AIC score. **Therefore, Poisson remains the mathematically correct choice for this specific portfolio.**

---

## 2. Severity Modeling (Gamma vs. Lognormal)

We trained an L2-regularized Lognormal Regression model (by log-transforming the target) to test if it handles extreme cyber claims better than our standard Actuarial Gamma GLM.

| Model | Adjusted AIC (Lower is Better) | Conclusion |
| :--- | :--- | :--- |
| **Gamma GLM (Baseline)** | 8181.89 | Worse Fit |
| **Lognormal Regression** | **7853.41** | **Winner** |

> [!IMPORTANT]  
> **Why did Lognormal win?** The Lognormal regression completely outperformed the Gamma GLM (a massive 300+ point drop in AIC). Cyber risk severity has an exceptionally "fat right tail" (e.g., standard claims are $50k, but severe claims hit $45M+). The Lognormal distribution is naturally "thicker" than the Gamma distribution, allowing it to mathematically account for these extreme cyber catastrophes much more accurately without treating them as statistical anomalies.

---

## 3. Extreme Tail Risk (Raw Distribution Fitting)

To prove *why* Lognormal won, we stripped away the regression features and ran a pure statistical Negative Log-Likelihood (NLL) test on the raw severity data using Gamma, Lognormal, and the extreme **Pareto** distribution.

| Distribution | Negative Log-Likelihood (NLL) | 
| :--- | :--- | 
| **Lognormal** | **3935.06 (Best Fit)** |
| **Gamma** | 3985.05 |
| **Pareto** | 4084.77 |

> [!TIP]  
> **Why didn't Pareto win?** While Pareto is famous for modeling the absolute worst 1-in-1,000 year catastrophes, it is extremely heavy-tailed. In this dataset, there are a lot of small-to-medium claims alongside the massive ones. The Pareto distribution struggles to model the smaller claims effectively, leading to a worse overall NLL score. Lognormal provides the perfect "Goldilocks" fit—handling both the standard claims and the extreme $45M tail.

---

## Final Business Recommendation

1. **Keep the Poisson GLM** for Frequency pricing. The data simply does not require the complexity of a Negative Binomial model.
2. **We *could* replace the Gamma GLM with a Lognormal Regression** for Severity pricing if you want to optimize for the absolute best mathematical fit regarding extreme tail risk. 

*Let me know if you would like me to modify the `app.py` Streamlit Dashboard to permanently use Lognormal Regression instead of Gamma!*
