# Day 5 — Streamlit app: transaction fraud probability, threshold slider, SHAP force plot.
from pathlib import Path

import streamlit as st
import pandas as pd
import shap

from sklearn.metrics import precision_score, recall_score
import streamlit.components.v1 as components
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
    data = pd.read_csv(BASE_DIR / "data" / "processed" / "creditcard_dedup.csv", index_col=0)
    X_test = joblib.load(BASE_DIR / "models" / "X_test.pkl")
    fraud_predictions = pd.read_csv(BASE_DIR / "reports" / "fraud_predictions.csv", index_col=0)
    return data, X_test, fraud_predictions


xgb_model = load_model()
explainer = load_explainer(xgb_model)
data, X_test, fraud_predictions = load_data()

y_test = fraud_predictions["actual"]
y_prob = fraud_predictions["predicted_proba"]


# --------------------------------
# 2. UI
# --------------------------------

st.title("Fraud Detection Dashboard")


# Transaction selection — restrict to test-set transactions (the ones we have ground truth for)
test_data = data.loc[X_test.index]

transaction_id = st.selectbox(
    "Select Transaction",
    test_data.index
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

shap_plot = shap.force_plot(
    explainer.expected_value,
    shap_values.values[0],
    selected_X.iloc[0]
)

components.html(
    shap.getjs() + shap_plot.html(),
    height=300
)
