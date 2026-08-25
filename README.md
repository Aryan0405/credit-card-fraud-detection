# Fraud Detection MVP

Model + SHAP + Streamlit + Failure Analysis + Benchmark Table, built on the
[Kaggle ULB Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.

## Setup

```bash
fraud-detection-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Structure

- `data/raw/` — original dataset (gitignored)
- `data/processed/` — generated/cleaned data (gitignored)
- `notebooks/` — day-by-day build notebooks
- `src/` — reusable data loading, training, and evaluation code
- `models/` — saved model artifacts (gitignored)
- `reports/figures/` — SHAP plots, confusion matrix images
- `reports/failure_analysis.md` — written failure analysis (Day 4)
- `app/streamlit_app.py` — Streamlit demo app (Day 5)
