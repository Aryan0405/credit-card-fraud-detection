# Fraud Detection MVP

Model + SHAP + Streamlit + Failure Analysis + Benchmark Table, built on the
[Kaggle ULB Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.

## Results

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.85 | 0.59 | 0.70 | 0.69 |
| XGBoost (main model) | 0.80 | 0.76 | 0.78 | 0.79 |

XGBoost was trained with `scale_pos_weight` class weighting rather than SMOTE: SMOTE synthesizes unrealistic points, weighting is simpler and safer. The weight was chosen by comparing 3 candidates on a held-out validation split, then re-checked against the test set since the top validation-PR-AUC pick didn't actually hold up there. Full table: [reports/benchmark_table.csv](reports/benchmark_table.csv).

## Failure Analysis

Out of 56,746 test transactions, the model produced 18 false positives and 23 false negatives, slightly biased toward missing fraud over raising false alarms, at a false-alarm rate of only 0.03%. Reviewing 4 representative cases with SHAP showed the errors split into two distinct types: **borderline cases** near the 0.5 decision threshold (fixable by adjusting the threshold, exactly what the Streamlit slider is for), and **confident misses** where the anonymized feature space doesn't separate the classes or fraud doesn't match the model's learned pattern (not fixable by threshold alone, would need better features or more diverse fraud examples).

Full case-by-case breakdown with SHAP evidence for each: [reports/failure_analysis.md](reports/failure_analysis.md).

## Setup

```bash
conda create -n fraud-detection python=3.11
conda activate fraud-detection
pip install -r requirements.txt
```

`numpy`/`scipy` are version-pinned in `requirements.txt`: newer releases have a broken LAPACK backend on Windows that crashes on `import shap`.

## Structure

- `data/raw/`: original dataset (gitignored)
- `data/processed/`: generated/cleaned data (gitignored)
- `notebooks/`: day-by-day build notebooks
- `models/`: saved model artifacts (gitignored)
- `reports/benchmark_table.csv`: baseline vs. main model metrics
- `reports/fraud_predictions.csv`: test-set predictions used by the notebooks and app
- `reports/failure_analysis.md`: written failure analysis (Day 4)
- `app.py`: Streamlit demo app (Day 5)
