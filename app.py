# Streamlit app: side-by-side comparison of XGBoost (main) vs FLAML-tuned XGBoost (automl-ceiling)

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)

st.set_page_config(page_title="Fraud Detection: Model Comparison", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

LEFT = "XGBoost (main)"
RIGHT = "FLAML-tuned XGBoost"

# --------------------------------
# 1. Loaders (cached)
# --------------------------------
@st.cache_resource
def load_main_model():
    return joblib.load(BASE_DIR / "models" / "xgb_model.pkl")


@st.cache_resource
def load_automl_bundle():
    # dict with keys: model, feature_names, threshold, best_config, ...
    return joblib.load(BASE_DIR / "models" / "flaml_bundle.joblib")


@st.cache_resource
def load_explainer(_model, key):
    return shap.TreeExplainer(_model)


@st.cache_resource
def load_data():
    X_test = joblib.load(BASE_DIR / "models" / "X_test.pkl")
    preds = pd.read_csv(BASE_DIR / "reports" / "fraud_predictions.csv", index_col=0)
    return X_test, preds


@st.cache_resource
def predict_all(_model, key, _X):
    return pd.Series(_model.predict_proba(_X)[:, 1], index=_X.index)


@st.cache_data
def compute_metrics(name, thr, _proba, _y):
    y_pred = (_proba >= thr).astype(int)
    tn, fp_, fn, tp = confusion_matrix(_y, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(_y, y_pred, zero_division=0),
        "recall": recall_score(_y, y_pred, zero_division=0),
        "f1": f1_score(_y, y_pred, zero_division=0),
        "pr_auc": average_precision_score(_y, _proba),
        "tn": tn, "fp": fp_, "fn": fn, "tp": tp,
    }


with st.spinner("Loading models and data (first run takes a bit)..."):
    main_model = load_main_model()
    bundle = load_automl_bundle()
    automl_model = bundle["model"]
    automl_features = bundle["feature_names"]
    automl_threshold = float(bundle["threshold"])

    X_test, fraud_predictions = load_data()
    y_test = fraud_predictions["actual"]

    # Each model gets the feature columns it was trained on, in the right order
    X_for = {
        LEFT: X_test,
        RIGHT: X_test[automl_features],
    }
    models = {LEFT: main_model, RIGHT: automl_model}
    explainers = {name: load_explainer(m, name) for name, m in models.items()}
    probas = {
        name: predict_all(models[name], name, X_for[name]).reindex(y_test.index)
        for name in models
    }

# --------------------------------
# 2. Shared controls
# --------------------------------
st.title("Fraud Detection Dashboard: Model Comparison")

c1, c2 = st.columns(2)

with c1:
    case_type = st.selectbox(
        "Case Type (based on main model's predictions)",
        ["All", "Fraud (caught) [TP]", "Fraud (missed) [FN]",
         "False alarm [FP]", "Legit (correct) [TN]"],
    )

fp = fraud_predictions
if case_type == "Fraud (caught) [TP]":
    filtered = fp[(fp["actual"] == 1) & (fp["predicted"] == 1)]
elif case_type == "Fraud (missed) [FN]":
    filtered = fp[(fp["actual"] == 1) & (fp["predicted"] == 0)]
elif case_type == "False alarm [FP]":
    filtered = fp[(fp["actual"] == 0) & (fp["predicted"] == 1)]
elif case_type == "Legit (correct) [TN]":
    filtered = fp[(fp["actual"] == 0) & (fp["predicted"] == 0)]
else:
    filtered = fp

with c2:
    transaction_id = st.selectbox("Select Transaction", filtered.index)

threshold_mode = st.radio(
    "Threshold mode",
    ["Same threshold for both (slider)", "Each model's own tuned threshold"],
    horizontal=True,
)

if threshold_mode.startswith("Same"):
    t = st.slider("Risk Score Threshold", 0.0, 1.0, 0.5, 0.01)
    thresholds = {LEFT: t, RIGHT: t}
else:
    thresholds = {LEFT: 0.5, RIGHT: automl_threshold}
    st.caption(
        f"{LEFT}: 0.50  |  {RIGHT}: {automl_threshold:.4f} "
        "(the F1-optimal threshold from the AutoML notebook)"
    )

actual_label = int(y_test.loc[transaction_id])
st.caption(f"Actual label for this transaction: **{'FRAUD' if actual_label == 1 else 'LEGIT'}**")

st.divider()

# --------------------------------
# 3. Side-by-side panels
# --------------------------------
def render_panel(name):
    thr = thresholds[name]
    score = float(probas[name].loc[transaction_id])

    st.subheader(name)
    st.metric("Risk Score", f"{score:.3f}")
    if prob >= thr:
        st.error(f"FRAUD (threshold {thr:.2f})")
    else:
        st.success(f"NOT FRAUD (threshold {thr:.2f})")

    mt = compute_metrics(name, thr, probas[name], y_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{mt['precision']:.2%}")
    m2.metric("Recall", f"{mt['recall']:.2%}")
    m3.metric("F1", f"{mt['f1']:.2%}")
    m4.metric("PR-AUC", f"{mt['pr_auc']:.3f}")

    st.caption(f"TP: {mt['tp']}  |  FP: {mt['fp']}  |  FN: {mt['fn']}  |  TN: {mt['tn']}")

    st.markdown("**Why did the model make this prediction?**")
    try:
        shap_values = explainers[name](X_for[name].loc[[transaction_id]])
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.info(f"SHAP couldn't be computed for this model: {e}")


left_col, right_col = st.columns(2, gap="large")
with left_col:
    render_panel(LEFT)
with right_col:
    render_panel(RIGHT)

# --------------------------------
# 4. Agreement summary
# --------------------------------
st.divider()
pred_l = probas[LEFT] >= thresholds[LEFT]
pred_r = probas[RIGHT] >= thresholds[RIGHT]
agree = (pred_l == pred_r).mean()
disagree_fraud = ((pred_l != pred_r) & (y_test == 1)).sum()
st.caption(
    f"The two models agree on {agree:.2%} of test transactions. "
    f"They disagree on {int(disagree_fraud)} actual fraud cases."
)