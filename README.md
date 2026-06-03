# Intern Package: Cyber Risk & BI Pricing Models

Start here. Read in this order:

1. `docs/Intern_Onboarding_Package.docx` — Full onboarding document. Scope, technical components, workplan, week 1 tasks.
2. `data/DATA_DICTIONARY.md` — Description of every dataset and field.
3. `starter_code/01_use_case_1_eda.py` — Exploratory data analysis starter
4. `starter_code/03_nlp_severity_baseline.py` — NLP severity classifier baseline
5. `starter_code/02_use_case_2_bi_simulation.py` — BI Monte Carlo skeleton

## Quick Start

```bash
# Set up environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Generate mock data (already pre-generated in /data, but verifies your setup)
python generate_mock_data.py

# Run the three starter scripts
python starter_code/01_use_case_1_eda.py
python starter_code/03_nlp_severity_baseline.py
python starter_code/02_use_case_2_bi_simulation.py
```

If all three scripts run without errors and produce printed summaries, your environment is correctly set up.

## Repository Layout

```
intern_package/
├── README.md                          (this file)
├── requirements.txt
├── generate_mock_data.py              (reproduces the mock datasets)
├── build_docx.js                      (rebuilds the onboarding doc; ignore unless updating)
├── data/
│   ├── DATA_DICTIONARY.md
│   ├── 01_policies.csv
│   ├── 02_claims.csv
│   ├── 03_regulatory_findings.csv
│   ├── 04_questionnaire_responses.csv
│   ├── 05_system_recovery_profiles.csv
│   ├── 06_outage_events.csv
│   ├── 07_modeling_dataset.csv         (produced by 01_use_case_1_eda.py)
│   └── 08_bi_pricing_output_sample.csv (produced by 02_use_case_2_bi_simulation.py)
├── starter_code/
│   ├── 01_use_case_1_eda.py
│   ├── 02_use_case_2_bi_simulation.py
│   ├── 03_nlp_severity_baseline.py
│   └── models/                         (saved baseline models)
└── docs/
    └── Intern_Onboarding_Package.docx
```

## Important Reminders

- All data in `data/` is synthetic — no real institution is represented.
- The mock data is cleaner than real production data. Build defensively.
- Some data quality issues are intentional. Finding and handling them is part of the work.
- Commit early and often. Daily commits to a shared repo are expected.
- Reach out before you spend two days on the wrong thing. The cost of a question is low.
