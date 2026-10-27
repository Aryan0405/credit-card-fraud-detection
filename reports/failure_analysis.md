# Failure Analysis

This analysis covers the main XGBoost model at the default 0.5 threshold (18 false positives, 23 false negatives on the 56,746-transaction test set). The FLAML-tuned model is not analysed here.

All features (`V1`-`V28`) are anonymized PCA components with no disclosed real-world meaning, so "what does this feature mean" isn't answerable. What SHAP does let us answer is which anonymized components drove each decision, in which direction, and whether that matches the pattern the model learned from real fraud (Day 3: strongly negative `V14` is the dominant, consistent fraud signal).

> **Data note.** Fraud probabilities and `f(x)` (log-odds) values below match the current `reports/fraud_predictions.csv`. The per-feature SHAP contributions (e.g. `V14` +4.71) were read from waterfalls generated before the model was retrained with cross-validated weight selection; probabilities moved by about 0.2 percentage points, so the contributions may differ slightly. Re-run `notebooks/04_failure_analysis.ipynb` to refresh them. The case indices and conclusions are unchanged.

## False Positives

### Case 1 (index 172818, scored 0.815 by the main model, actually legit)

- **What features pushed the model toward its (wrong) decision?** `V14 = -6.754` dominates (+4.71 SHAP), followed by `V4` (+1.02), `V10` (+0.81), and `V12` (+0.81). This is the same set of features, in the same direction, that drove every genuine fraud case reviewed in Day 3.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** It makes sense given what the model learned: this legit transaction's PCA profile statistically overlaps with real fraud. It's not a malfunction, it's a hard example where the anonymized feature space doesn't fully separate the two classes.
- **What would you change?** Cross-check against the other models: the FLAML model scores this transaction 0.198 (below its 0.2222 cut-off) and logistic regression 0.254, so it is not inherently inseparable, and part of this false positive comes from the class-weighted main model. Whether new features (merchant category, device signals, transaction velocity) would help cannot be tested with this dataset.

### Case 2 (index 152102, scored 0.548 by the main model, actually legit)

- **What features pushed the model toward its (wrong) decision?** `V4 = 4.382` dominates (+2.46), not `V14`; `V14` only contributes +0.35 here, far below its usual weight. `V12` (+0.92) and `V19` (+0.39) also contribute.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** This is a weak, borderline call: `f(x) = 0.193` sits barely above the decision boundary (54.8%, barely over 50%). The model isn't confidently wrong, it's uncertain, and the usual strongest signal (`V14`) barely shows up. This looks like model uncertainty, not a real pattern-based error.
- **What would you change?** This is exactly what a threshold adjustment fixes: raising the decision threshold slightly above 0.55 flips this specific prediction to legit without needing a different model or new features. The Streamlit threshold slider lets you test this directly.

## False Negatives

### Case 1 (index 240222, scored 0.311 by the main model, actually fraud)

- **What features pushed the model toward its (wrong) decision?** `V14 = -4.252` pushes toward fraud strongly (+3.8), consistent with the usual pattern. But `V11 = -0.688` pulls back hard (-1.25), along with smaller negative contributions from `V10`, `V12`, `V21`, and `V16`. The net score (`f(x) = -0.796`, ≈31% predicted fraud probability) lands just below the 0.5 threshold.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** This is a near-miss, not a fundamental error. The fraud signal (`V14`) is present and correctly recognized; it's just outweighed by conflicting signal from `V11` and a handful of smaller features. 31% is close enough to the boundary that this reads as model uncertainty, not confident wrongness.
- **What would you change?** Lowering the decision threshold to 0.31 or below would catch this case (and similar near-misses), at the cost of more false positives elsewhere: the core precision/recall trade-off the Streamlit slider makes visible.

### Case 2 (index 249239, scored 0.0006 by the main model, actually fraud)

- **What features pushed the model toward its (wrong) decision?** `V14 = 0.985` pushes *away* from fraud (-1.5) here, the opposite sign from every other case reviewed (Day 3's true positives, both FP cases above, and FN Case 1 all had strongly negative `V14` driving toward fraud). `V12` (-0.63) and `V4` (-0.37) also push away from fraud, more weakly.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** No, this genuinely exposes a gap. The model has effectively learned "fraud looks like strongly negative `V14`," and this fraud case doesn't fit that pattern at all. It's confidently wrong (predicted fraud probability ≈0.06%, `f(x) = -7.37`) because it doesn't resemble the fraud examples the model was trained on.
- **What would you change?** This isn't a threshold problem: no reasonable threshold shift flips a <0.1% prediction. It would need either more training examples of this atypical fraud pattern, or additional features capturing whatever makes this transaction fraudulent beyond what `V1`-`V28` encode.

## Overall Trade-off

Out of 56,746 test transactions (95 fraud, 56,651 legit), the model produced 18 false positives and 23 false negatives (confusion matrix: TP 72, FP 18, FN 23, TN 56,633). In raw count the model is biased slightly toward false negatives, missing more fraud (23) than it wrongly flags (18), while still catching 76% of all fraud (recall 0.76) at a false-alarm rate of only 0.03% of legit transactions (18/56,651). Of the 90 transactions flagged as fraud, 72 (80%) were real fraud.

The 4 cases above split into two different failure types, not one: the borderline cases (FP Case 2, FN Case 1) are threshold problems: both sit close to the 0.5 boundary and a different cutoff would flip them, trading precision for recall or vice versa. The confident cases (FP Case 1, FN Case 2) are not threshold problems: they're places where the anonymized feature space itself doesn't separate the classes (FP Case 1) or where fraud shows up in a pattern the model hasn't learned to recognize (FN Case 2). No threshold adjustment fixes those; they'd need better features or more diverse fraud examples. These are four examples, not a categorized breakdown of all 41 errors.

How moving the threshold changes the error counts on this test set (computed from `reports/fraud_predictions.csv`; illustrative, since the thresholds are evaluated on the same data):

| Threshold | Precision | Recall | False positives | False negatives |
|---|---|---|---|---|
| 0.2 | 0.62 | 0.81 | 47 | 18 |
| 0.3 | 0.72 | 0.80 | 29 | 19 |
| 0.5 | 0.80 | 0.76 | 18 | 23 |
| 0.7 | 0.83 | 0.76 | 15 | 23 |

Missed fraud (false negatives) costs money directly while false positives mainly cost customer friction, and the current false-alarm rate is already very low (0.03%). That suggests room to lower the threshold, trading some precision for a meaningful recall gain: moving from 0.5 to 0.3 catches 4 more frauds for 11 more false alarms. This is reasoning, not a measured result: the project has no cost figures, so the right threshold depends on the real cost of each error type, and a threshold should be chosen on validation data rather than on the test set. The Streamlit threshold slider lets a user explore this trade-off directly rather than committing to one fixed threshold.

With only 95 fraud cases in the test set, each missed or caught fraud moves recall by about one percentage point, so small differences between thresholds are within noise.

## Cross-model check (added after independent review)

- **Scores are not probabilities.** Because the main model is trained with `scale_pos_weight=24`, its scores are inflated: the 13 test transactions scored between 0.5 and 0.9 were fraud only once (1 of 13), and the average score is 2.2 times the true fraud rate. Read "0.815" as a risk score, not an 81.5% chance of fraud.
- **Most main-model false positives are model-specific.** FLAML flags only 3 legit transactions, all 3 also flagged by the main model, so 15 of the main model's 18 false positives are not flagged by FLAML. FP Case 1 and Case 2 are both among them.
- **The structural failures are shared.** 22 of the 95 test frauds are missed by the main model, FLAML and logistic regression alike. In 20 of those 22, the nearest training transaction in `V1`-`V28` space is a legit one rather than a fraud (7 of 73 for the caught frauds); their median `V14` is -0.68 versus -7.13 for caught frauds, and their median `Amount` is 2.0 versus 33.6. FN Case 2 is one example of this group, FN Case 1 is not (the main model scores it 0.311 but FLAML 0.0007).