# Maestro Cyber Pricing Engine

Welcome to the Maestro Cyber Risk Dashboard repository. This project establishes an end-to-end data science and actuarial pipeline designed to dynamically price cyber insurance policies. 

By combining traditional Actuarial Science with modern Machine Learning (XGBoost), Natural Language Processing (DistilBERT), and Generative AI (Google Gemini ADK), this engine provides highly accurate, transparent, and explainable pricing for complex cyber risks.

## 🚀 Live Streamlit Dashboard
You can interact with the outputs, visualize risk patterns, dynamically alter security controls, and chat with the AI Agent using the Streamlit application. 

To launch the dashboard, open your terminal in this repository and run:
```bash
streamlit run app.py
```

---

## The End-to-End Pricing Journey: Step-by-Step

### Step 1: Feature Engineering & NLP Extraction
The foundation of the pricing engine starts with transforming raw, unstructured data into a cohesive modeling dataset. We moved beyond simple checkboxes to create sophisticated **Risk Indices**.
- **Mathematical Merging:** We combined multiple security metrics (NIST scores, MFA deployment, EDR, and 24/7 SOC) into a single, cohesive **Cyber Control Score**. We similarly engineered a **Vendor Risk Pressure** index by calculating the ratio of third-party vendors to a company's internal security maturity.
- **NLP Severity Extraction (DistilBERT):** To handle unstructured text from regulatory audits, we implemented a deep-learning Transformer pipeline (`DistilBERT`). This model reads raw textual findings, converts them into high-dimensional vector embeddings, and outputs the probability of a "High Severity" regulatory finding. This probability is injected back into the tabular data as a powerful predictive feature.

### Step 2: The Machine Learning Benchmark (XGBoost)
Before building our actuarial models, we deployed an optimized **XGBoost** algorithm to rigorously analyze historical loss patterns. 
- Machine learning models like XGBoost are exceptional at finding non-linear, compounding risks (e.g., discovering how the *combination* of a weak vendor score and missing multi-factor authentication exponentially increases risk). 
- We use XGBoost to benchmark the maximum possible predictive power of our dataset.

### Step 3: The Actuarial GLM Engine (Poisson & Gamma)
While XGBoost is highly predictive, it acts as a "black box," which is difficult to justify to insurance regulators. Therefore, for our core pricing engine, we transition to industry-standard **Actuarial Generalized Linear Models (GLMs)** to ensure absolute transparency.
- **Frequency (Probability of Claim):** We trained a mathematically robust **Poisson Distribution GLM** to predict the exact likelihood of an insured suffering a cyber event in a given year.
- **Severity (Cost of Claim):** We paired this with a **Gamma Distribution GLM** to model the expected financial loss (severity) if a claim does occur.

### Step 4: The Core Pricing Equation
With our Actuarial GLMs trained, we can calculate the final price of the insurance policy.
1. **Pure Premium:** The exact amount of money needed just to pay the expected claims.
   - `Pure Premium = Poisson Frequency × Gamma Severity`
2. **Technical Premium:** Pure premium alone cannot sustain an insurance business. We must add risk margins and expense loads to calculate the final price charged to the customer.
   - `Technical Premium = Pure Premium / (1 - Expense Ratio - Risk Margin)`

**Dynamic Interactivity:** As you adjust security controls in the dashboard (like increasing MFA), the Poisson GLM instantly registers the lower risk, recalculates the probability, and the Technical Premium drops live on the screen!

### Step 5: Tail Risk & Systemic Catastrophes
Deterministic GLMs calculate the "average" expected loss, but they fail to capture the extreme volatility of modern cyber threats (like a global cloud outage). To solve this, we evaluate the portfolio's historical total losses to identify **Tail Risk**.
- We calculate the **95% Value at Risk (VaR)** and the **99% Tail Value at Risk (TVaR)**.
- This ensures the portfolio holds enough capital reserves to survive systemic, catastrophic events.

### Step 6: Google Gemini ADK Agent Explainability
Finally, we don't just calculate the price; we explain it. We integrated a **Google Gemini ADK Agent** directly into the dashboard interface.
- Instead of underwriters guessing why a premium spiked, the Agent reads the exact mathematical GLM coefficients in the background.
- It instantly translates the complex math into clear, plain-English pricing logic, empowering underwriters to confidently explain and defend the premium to brokers and clients.

---

## Developer Quick Start

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the foundational scripts to generate features and GLM coefficients
python code/01_use_case_1_eda.py
python code/03_nlp_severity_baseline.py

# 3. Launch the Application
streamlit run app.py
```

## Repository Layout
```
maestro_cyber/
├── README.md                           (this file)
├── requirements.txt
├── app.py                              (Streamlit Dashboard - Main Application)
├── data/                               (Mock datasets and output features)
├── code/
│   ├── 01_use_case_1_eda.py            (End-to-End Pipeline & GLM Modeling)
│   ├── 03_nlp_severity_baseline.py     (DistilBERT extraction)
│   └── models/                         (Saved models and scalers)
├── outputs/                            (Saved diagnostics and GLM coefficients)
└── docs/                               (Documentation and planning notes)
```
