"""
NLP Severity Classifier — Starter Baseline
============================================
Trains a TF-IDF + logistic regression baseline on regulatory finding text
to predict severity (Low / Medium / High). This is the FIRST model to build —
once the team validates the pipeline end-to-end, they should swap in a
fine-tuned transformer (DistilBERT, FinBERT) and compare.

Outputs:
  - Train/val/test metrics
  - Confusion matrix
  - A reusable predict() function
  - The same model applied to questionnaire responses for quality scoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. Load NLP corpus
# ============================================================
findings = pd.read_csv(DATA_DIR / "03_regulatory_findings.csv")
print(f"Loaded {len(findings)} regulatory findings")
print(f"Label distribution:\n{findings['severity_label'].value_counts()}\n")

# ============================================================
# 2. Train/val/test split
# ============================================================
X = findings["finding_text"].values
y = findings["severity_label"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)
print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

# ============================================================
# 3. Baseline: TF-IDF + Logistic Regression
# ============================================================
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        stop_words="english",
    )),
    ("clf", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=1.0,
    )),
])

pipe.fit(X_train, y_train)

# ============================================================
# 4. Evaluate
# ============================================================
print("=" * 60)
print("VALIDATION METRICS")
print("=" * 60)
val_preds = pipe.predict(X_val)
print(classification_report(y_val, val_preds, digits=3))

print("=" * 60)
print("TEST METRICS")
print("=" * 60)
test_preds = pipe.predict(X_test)
print(classification_report(y_test, test_preds, digits=3))

print("Confusion matrix (rows: true, cols: predicted):")
labels = sorted(np.unique(y))
cm = confusion_matrix(y_test, test_preds, labels=labels)
print("       " + "  ".join(f"{l:>7s}" for l in labels))
for label, row in zip(labels, cm):
    print(f"{label:>7s}  " + "  ".join(f"{v:>7d}" for v in row))

# ============================================================
# 5. Persist model
# ============================================================
joblib.dump(pipe, MODEL_DIR / "severity_classifier_baseline.joblib")
print(f"\nSaved baseline model: {MODEL_DIR / 'severity_classifier_baseline.joblib'}")

# ============================================================
# 6. Inference helper for use in pricing pipeline
# ============================================================
def predict_severity(texts):
    """Predict severity labels and class probabilities for a list of finding texts."""
    labels = pipe.predict(texts)
    probs = pipe.predict_proba(texts)
    return labels, probs

# Test it on a single example
example = ["The institution has failed to implement basic access controls. MRIA issued."]
lbl, prb = predict_severity(example)
print(f"\nExample prediction: {lbl[0]} (probs: {dict(zip(pipe.classes_, prb[0].round(3)))})")

# ============================================================
# 7. Apply to questionnaire responses (re-use for quality scoring)
# ============================================================
print("\n" + "=" * 60)
print("APPLYING TO QUESTIONNAIRE RESPONSES")
print("=" * 60)
print("Note: this re-trains on the questionnaire labels — different model.")

responses = pd.read_csv(DATA_DIR / "04_questionnaire_responses.csv")
Xq = responses["response_text"].values
yq = responses["response_quality_label"].values

Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(Xq, yq, test_size=0.20, stratify=yq, random_state=42)

pipe_q = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
pipe_q.fit(Xq_tr, yq_tr)
print("\nQuestionnaire quality classifier — test metrics:")
print(classification_report(yq_te, pipe_q.predict(Xq_te), digits=3))

joblib.dump(pipe_q, MODEL_DIR / "questionnaire_quality_classifier.joblib")

# ============================================================
# 8. Next steps
# ============================================================
print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("""
1. Compare this TF-IDF baseline against:
   - DistilBERT fine-tuned on the same labels (HuggingFace Trainer)
   - A sentence-transformer + linear probe (cheaper, often surprisingly close)
2. Use the model's class probabilities, not just the hard label, as features
   in the downstream pricing model — preserves uncertainty information
3. Build an aggregation function: per policy_id, summarize the findings
   into a feature vector (count by severity, weighted severity score, days
   since most recent high-severity finding, etc.)
4. Look for spurious correlations: if the model learns the template language
   (e.g. "MRIA" always = High), it will overfit. Test by paraphrasing a few
   findings manually and seeing if predictions change appropriately.
5. Wire the predicted scores into 07_modeling_dataset.csv as new features.
""")
