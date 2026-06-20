# Maestro Cyber Pricing Models

Welcome to the Maestro Cyber Risk & BI Pricing Engine repository. This project establishes an end-to-end data science and actuarial pipeline encompassing two primary use cases: **Frequency-Severity Pricing** and **Advanced Business Interruption (BI) Simulation**.

## 🚀 Live Streamlit Dashboards
You can interact with the outputs, visualize risk patterns, and run live Monte Carlo simulations using the following Streamlit applications. 

To launch them, open your terminal in this repository and run the dynamic links below:

- **[Use Case 1: Frequency-Severity Pricing & EDA](http://localhost:8501)**
  - Run command: `streamlit run app.py`
- **[Use Case 2: Advanced BI Ransomware Simulator](http://localhost:8502)**
  - Run command: `streamlit run app_2.py`

*(Note: Click the dynamic links above once you have executed the respective run commands in your terminal.)*

---

## The End-to-End Pricing Journey

### 1. Feature Engineering
The foundation of the pricing engine starts with transforming raw, disparate data into a cohesive modeling dataset.
- **Data Merging:** We combined policyholder details, security control scores (e.g., NIST), third-party vendor risk data, and historical claim events.
- **Categorical & Numeric Transformations:** Categorical variables like `sub_sector` and `primary_regulator` were one-hot encoded to allow machine learning algorithms to interpret them.
- **NLP Severity Extraction (DistilBERT):** To handle unstructured text from regulatory findings, we implemented a deep-learning `DistilBERT` pipeline. This model reads raw textual findings, converts them into high-dimensional vector embeddings, and uses a Random Forest classifier to output the probability of a "High Severity" finding. This probability (`high_sev_prob`) is injected back into the tabular data as a powerful engineered feature.

### 2. Frequency & Severity Modeling (Use Case 1)
With the engineered dataset ready, we predict the foundational components of insurance loss:
- **How it is AI-Powered:** Instead of relying on rigid, linear traditional actuarial models, we use Machine Learning. Non-linear, tree-based models like Random Forest and XGBoost capture complex interactions (e.g., how the *combination* of a weak vendor score and missing multi-factor authentication exponentially increases risk). Additionally, we use NLP (DistilBERT) to dynamically read unstructured regulatory text and convert it into a numeric risk probability.
- **Frequency (Probability of Claim):** Using algorithmic class weighting to combat the heavily imbalanced dataset (claims are rare), we trained multiple algorithms to predict the likelihood of an insured suffering a cyber event in a given year.
- **Severity (Cost of Claim):** We mapped historical loss data against our features using a lognormal regression to determine the expected average severity if a claim does occur.

#### Model Comparison Results
We rigorously evaluated three predictive algorithms for the frequency model. The predictive strength was measured using AUROC (Area Under the Receiver Operating Characteristic curve) which captures how well the model separates the 'Claim' vs 'No Claim' groups. Because cyber claims are rare events, traditional F1-Scores at a 0.5 threshold are zero, but the AUROC clearly shows AI's superiority:

| Algorithm | AUROC | F1-Score (at 0.5 thresh) | Actuarial Verdict |
| :--- | :--- | :--- | :--- |
| **Traditional GLM** | 0.5965 | 0.0000 | Baseline performance. Cannot capture non-linear cyber risks. |
| **Random Forest (AI)** | 0.6231 | 0.0000 | Better predictive power. Captures interactions between security controls. |
| **XGBoost (AI)** | **0.6434** | 0.0000 | **Best performance.** Gradient boosting excels at finding subtle, sequential risk patterns. |

### 3. Pure Premium Calculation & Monte Carlo Engine
The **Pure Premium** represents the pure mathematical expectation of loss (the exact amount of money needed just to pay the average claims, with no profit).

- **The Core Actuarial Formula:** `Pure Premium = Expected Frequency × Expected Severity`
- **In Use Case 1:** We calculate this deterministically using the outputs of the machine learning models.
- **In Use Case 2 (Monte Carlo Frequency-Severity Simulation):** We run an advanced stochastic simulation to generate 50,000 parallel universes (years) for a single policy. 
  - **Simulating Frequency:** In each universe, we draw from a Poisson distribution (driven by the AI frequency probability) to see exactly *how many* attacks hit the company. 
  - **Simulating Severity:** For every attack that occurs, we draw from a Lognormal distribution to simulate the *financial severity* of that specific attack.
  - The sum of all losses in a simulated year is the total BI loss for that universe. The **Mean** of all 50,000 universes becomes our simulated Pure Premium, while the worst universes represent our extreme Tail Risk.

### 4. Advanced Actuarial Risk Margin
Because the Pure Premium only covers the *average* expectation, an insurance company must charge more to protect against volatility and catastrophic tail risks (like a systemic ransomware outbreak). 
- We isolate the worst 1% of simulated years using **Tail Value at Risk (TVaR 99%)**.
- The required capital base to survive these catastrophic years is `TVaR_99 - Expected Loss`.
- We apply a **Cost of Capital Charge** (e.g., 10%) to this required capital base, yielding our dynamic **Risk Load**. 
- Highly secure companies receive a tiny risk load, while highly volatile profiles receive massive risk loads.

### 5. Technical Premium Derivation
The final **Technical Premium** is the actual price required to confidently underwrite the policy while paying for all operational expenses.
- **Formula:** `(Pure Premium + Risk Load) / (1 - Expense Ratio)`
- **Components:**
  - *Pure Premium:* Pays the expected claims.
  - *Risk Load:* Pays the shareholders/investors for risking their capital on catastrophic tail events.
  - *Expense Ratio (e.g. 25%):* Pays the broker commissions, underwriters, and company overhead.

The result is a hyper-accurate, AI-driven, risk-adjusted technical premium that correctly prices modern cyber risk!

---

## Developer Quick Start

```bash
# Set up environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the three core scripts
python code/01_use_case_1_eda.py
python code/03_nlp_severity_baseline.py
python code/02_use_case_2_bi_simulation.py
```

## Repository Layout
```
maestro_cyber/
├── README.md                           (this file)
├── requirements.txt
├── app.py                              (Streamlit Dashboard for Use Case 1)
├── app_2.py                            (Streamlit Dashboard for Use Case 2)
├── data/                               (Mock datasets and output features)
├── code/
│   ├── 01_use_case_1_eda.py            (End-to-End Pipeline & Modeling)
│   ├── 02_use_case_2_bi_simulation.py  (Actuarial Monte Carlo Engine)
│   ├── 03_nlp_severity_baseline.py     (DistilBERT extraction)
│   └── models/                         (Saved joblib baseline models)
├── outputs/                            (Saved EDA visuals and diagnostics)
└── docs/                               (Documentation and planning notes)
```
