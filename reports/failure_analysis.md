# Failure Analysis

All features (`V1`-`V28`) are anonymized PCA components with no disclosed real-world meaning, so "what does this feature mean" isn't answerable. What SHAP does let us answer is which anonymized components drove each decision, in which direction, and whether that matches the pattern the model learned from real fraud (Day 3: strongly negative `V14` is the dominant, consistent fraud signal).

## False Positives

### Case 1 (index 172818, predicted fraud at 81.7% confidence, actually legit)

- **What features pushed the model toward its (wrong) decision?** `V14 = -6.754` dominates (+4.71 SHAP), followed by `V4` (+1.02), `V10` (+0.81), and `V12` (+0.81). This is the same set of features, in the same direction, that drove every genuine fraud case reviewed in Day 3.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** It makes sense given what the model learned: this legit transaction's PCA profile statistically overlaps with real fraud. It's not a malfunction, it's a hard example where the anonymized feature space doesn't fully separate the two classes.
- **What would you change?** Nothing within the current feature set: the model weighted `V14` correctly, this is just where the classes overlap. Additional features not present in this dataset (merchant category, device signals, transaction velocity) would be needed to separate this specific legit transaction from real fraud.

### Case 2 (index 152102, predicted fraud at 54.6% confidence, actually legit)

- **What features pushed the model toward its (wrong) decision?** `V4 = 4.382` dominates (+2.46), not `V14`; `V14` only contributes +0.35 here, far below its usual weight. `V12` (+0.92) and `V19` (+0.39) also contribute.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** This is a weak, borderline call: `f(x) = 0.183` sits barely above the decision boundary (54.6%, barely over 50%). The model isn't confidently wrong, it's uncertain, and the usual strongest signal (`V14`) barely shows up. This looks like model uncertainty, not a real pattern-based error.
- **What would you change?** This is exactly what a threshold adjustment fixes: raising the decision threshold slightly above 0.5 flips this specific prediction to legit without needing a different model or new features. Worth testing with Day 5's threshold slider.

## False Negatives

### Case 1 (index 240222, predicted legit at ~31% fraud probability, actually fraud)

- **What features pushed the model toward its (wrong) decision?** `V14 = -4.252` pushes toward fraud strongly (+3.8), consistent with the usual pattern. But `V11 = -0.688` pulls back hard (-1.25), along with smaller negative contributions from `V10`, `V12`, `V21`, and `V16`. The net score (`f(x) = -0.806`, ≈31% predicted fraud probability) lands just below the 0.5 threshold.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** This is a near-miss, not a fundamental error. The fraud signal (`V14`) is present and correctly recognized; it's just outweighed by conflicting signal from `V11` and a handful of smaller features. 31% is close enough to the boundary that this reads as model uncertainty, not confident wrongness.
- **What would you change?** Lowering the decision threshold below 0.5 would catch this case (and similar near-misses), at the cost of more false positives elsewhere: the core precision/recall trade-off the Streamlit slider should make visible.

### Case 2 (index 249239, predicted legit at <0.1% fraud probability, actually fraud)

- **What features pushed the model toward its (wrong) decision?** `V14 = 0.985` pushes *away* from fraud (-1.5) here, the opposite sign from every other case reviewed (Day 3's true positives, both FP cases above, and FN Case 1 all had strongly negative `V14` driving toward fraud). `V12` (-0.63) and `V4` (-0.37) also push away from fraud, more weakly.
- **Does the pattern make sense, or does it expose something the model doesn't understand?** No, this genuinely exposes a gap. The model has effectively learned "fraud looks like strongly negative `V14`," and this fraud case doesn't fit that pattern at all. It's confidently wrong (predicted fraud probability <0.1%) because it doesn't resemble the fraud examples the model was trained on.
- **What would you change?** This isn't a threshold problem: no reasonable threshold shift flips a <0.1% prediction. It would need either more training examples of this atypical fraud pattern, or additional features capturing whatever makes this transaction fraudulent beyond what `V1`-`V28` encode.

## Overall Trade-off

Out of 56,746 test transactions, the model produced 18 false positives and 23 false negatives (Day 2 confusion matrix). In raw count the model is biased slightly toward false negatives, missing more fraud (23) than it wrongly flags (18), while still catching 76% of all fraud (recall 0.76) at a false-alarm rate of only 0.03% of legit transactions (18/56,651).

The 4 cases above split into two different failure types, not one: the borderline cases (FP Case 2, FN Case 1) are threshold problems: both sit close to the 0.5 boundary and a different cutoff would flip them, trading precision for recall or vice versa. The confident cases (FP Case 1, FN Case 2) are not threshold problems: they're places where the anonymized feature space itself doesn't separate the classes (FP Case 1) or where fraud shows up in a pattern the model hasn't learned to recognize (FN Case 2). No threshold adjustment fixes those; they'd need better features or more diverse fraud examples.

Given that missed fraud (false negatives) costs money directly while false positives mainly cost customer friction, and the current false-alarm rate is already very low (0.03%), there's room to push the decision threshold down to trade a modest precision loss for a meaningful recall gain: this is exactly the trade-off Day 5's threshold slider should let a user explore directly, rather than committing to one fixed threshold now.
