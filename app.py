# Day 5 — Streamlit app: transaction fraud probability, threshold slider, SHAP waterfall plot.
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import precision_score, recall_score
import joblib

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------
# 1. Load model and data (cached so this only runs once per session)
# --------------------------------

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "models" / "xgb_model.pkl")


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_data():
    X_test = joblib.load(BASE_DIR / "models" / "X_test.pkl")
    fraud_predictions = pd.read_csv(BASE_DIR / "reports" / "fraud_predictions.csv", index_col=0)
    return X_test, fraud_predictions


xgb_model = load_model()
explainer = load_explainer(xgb_model)
X_test, fraud_predictions = load_data()

y_test = fraud_predictions["actual"]
y_prob = fraud_predictions["predicted_proba"]


# --------------------------------
# 2. UI
# --------------------------------

st.title("Fraud Detection Dashboard")


# Case type filter — jump straight to a caught fraud, a missed fraud, or a false alarm
case_type = st.selectbox(
    "Case Type",
    ["All", "Fraud (caught) [TP]", "Fraud (missed) [FN]", "False alarm [FP]", "Legit (correct) [TN]"]
)

if case_type == "Fraud (caught) [TP]":
    filtered = fraud_predictions[(fraud_predictions["actual"] == 1) & (fraud_predictions["predicted"] == 1)]
elif case_type == "Fraud (missed) [FN]":
    filtered = fraud_predictions[(fraud_predictions["actual"] == 1) & (fraud_predictions["predicted"] == 0)]
elif case_type == "False alarm [FP]":
    filtered = fraud_predictions[(fraud_predictions["actual"] == 0) & (fraud_predictions["predicted"] == 1)]
elif case_type == "Legit (correct) [TN]":
    filtered = fraud_predictions[(fraud_predictions["actual"] == 0) & (fraud_predictions["predicted"] == 0)]
else:
    filtered = fraud_predictions

transaction_id = st.selectbox(
    "Select Transaction",
    filtered.index
)


# Get selected transaction
selected_X = X_test.loc[[transaction_id]]


# --------------------------------
# 3. Fraud probability
# --------------------------------

probability = xgb_model.predict_proba(
    selected_X
)[0, 1]

st.metric(
    "Fraud Probability",
    f"{probability:.2%}"
)


# --------------------------------
# 4. Threshold
# --------------------------------

threshold = st.slider(
    "Fraud Threshold",
    0.0,
    1.0,
    0.5,
    0.01
)


# --------------------------------
# 5. Precision / Recall
# --------------------------------

y_pred = (y_prob >= threshold).astype(int)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)


col1, col2 = st.columns(2)

col1.metric("Precision", f"{precision:.2%}")
col2.metric("Recall", f"{recall:.2%}")


# --------------------------------
# 6. Selected transaction result
# --------------------------------

if probability >= threshold:
    st.error("FRAUD")
else:
    st.success("NOT FRAUD")


# --------------------------------
# 7. SHAP
# --------------------------------

st.subheader("Why did the model make this prediction?")

shap_values = explainer(selected_X)

fig = plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
plt.close(fig)
