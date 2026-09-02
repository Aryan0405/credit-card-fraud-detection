# Fraud Detection MVP

Model + SHAP + Streamlit + Failure Analysis + Benchmark Table, built on the
[Kaggle ULB Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.

## Setup

```bash
conda create -n fraud-detection python=3.11
conda activate fraud-detection
pip install -r requirements.txt
```

`numpy`/`scipy` are version-pinned in `requirements.txt` — newer releases have a broken LAPACK backend on Windows that crashes on `import shap`.

## Structure

- `data/raw/` — original dataset (gitignored)
- `data/processed/` — generated/cleaned data (gitignored)
- `notebooks/` — day-by-day build notebooks
- `models/` — saved model artifacts (gitignored)
- `reports/benchmark_table.csv` — baseline vs. main model metrics
- `reports/fraud_predictions.csv` — test-set predictions used by the notebooks and app
- `reports/failure_analysis.md` — written failure analysis (Day 4)
- `app.py` — Streamlit demo app (Day 5)
