# Maestro Cyber Pricing: Project Summary

This document summarizes the end-to-end data science and actuarial pipeline developed for the Maestro Cyber Pricing engine, encompassing two primary use cases.

## Streamlit Dashboards

You can interact with the outputs and run live simulations using the following Streamlit applications. To launch them, open your terminal in the project root directory and run the commands below.

- **Use Case 1 (Frequency-Severity Pricing & EDA):**
  - **Command:** `streamlit run app.py`
  - **Link:** [http://localhost:8501](http://localhost:8501) (Once running)
- **Use Case 2 (Advanced BI Ransomware Simulator):**
  - **Command:** `streamlit run app_2.py`
  - **Link:** [http://localhost:8502](http://localhost:8502) (If running concurrently with app.py)

---

## The End-to-End Pricing Journey

### 1. Feature Engineering
The foundation of the pricing engine starts with transforming raw, disparate data into a cohesive modeling dataset.
- **Data Merging:** We combined policyholder details, security control scores (e.g., NIST), third-party vendor risk data, and historical claim events.
- **Categorical & Numeric Transformations:** Categorical variables like `sub_sector` and `primary_regulator` were one-hot encoded to allow machine learning algorithms to interpret them.
- **NLP Severity Extraction (DistilBERT):** To handle unstructured text from regulatory findings, we implemented a deep-learning `DistilBERT` pipeline. This model reads raw textual findings, converts them into high-dimensional vector embeddings, and uses a Random Forest classifier to output the probability of a "High Severity" finding. This probability (`high_sev_prob`) is injected back into the tabular data as a powerful engineered feature.

### 2. Frequency & Severity Modeling (Use Case 1)
With the engineered dataset ready, we predict the foundational components of insurance loss:
- **Frequency (Probability of Claim):** Using algorithmic class weighting to combat the heavily imbalanced dataset (claims are rare), we trained AI-Powered models (Random Forest) to predict the likelihood of an insured suffering a cyber event in a given year. The Random Forest significantly outperformed traditional Generalized Linear Models (GLMs).
- **Severity (Cost of Claim):** We mapped historical loss data against our features to determine the expected average severity if a claim does occur.

### 3. Pure Premium Calculation
The **Pure Premium** represents the pure mathematical expectation of loss.
- **In Use Case 1:** It is simply `Predicted Frequency × Expected Severity`.
- **In Use Case 2 (BI Simulation):** It is the **Mean** of $50,000$ Monte Carlo simulated years. The simulator models correlated ransomware attacks hitting multiple systems, applying bimodal Gaussian Mixture distributions to determine exact downtime and revenue loss.

### 4. Advanced Actuarial Risk Margin
Because the Pure Premium only covers the *average* expectation, an insurance company must charge more to protect against volatility and catastrophic tail risks (like a systemic ransomware outbreak). 
- We isolate the worst 1% of simulated years using **Tail Value at Risk (TVaR 99%)**.
- The required capital base to survive these catastrophic years is `TVaR_99 - Expected Loss`.
- We apply a **10% Cost of Capital Charge** to this required capital base, yielding our dynamic **Risk Load**. 
- Highly secure companies receive a tiny risk load, while highly volatile profiles receive massive risk loads.

### 5. Technical Premium Derivation
The final **Technical Premium** is the actual price required to confidently underwrite the policy while paying for all operational expenses.
- **Formula:** `(Pure Premium + Risk Load) / (1 - Expense Ratio)`
- **Components:**
  - *Pure Premium:* Pays the expected claims.
  - *Risk Load:* Pays the shareholders/investors for risking their capital on catastrophic tail events.
  - *Expense Ratio (25%):* Pays the broker commissions, underwriters, and company overhead.

The result is a hyper-accurate, AI-driven, risk-adjusted technical premium that correctly prices modern cyber risk!
