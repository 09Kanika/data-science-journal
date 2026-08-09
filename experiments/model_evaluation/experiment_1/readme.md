# Why Accuracy Lies and When F1 Saves You

Choosing the wrong evaluation metric can make a model look excellent while silently failing the business objective. A model that predicts every transaction as "not fraud" in a 99:1 dataset hits 99% accuracy and detects zero fraud. The business loses millions. The metric looks fine.

This experiment builds the intuition for *why* different metrics exist, *when* each one matters, and *how* threshold choices reshape what a model appears capable of.

---

## Question

**Can the same trained model look "perfect" under one metric and completely broken under another — and what does that mean for real decisions?**

---

## The real cost of getting metrics wrong

Different mistakes carry different business consequences. A false negative in fraud detection means a fraudulent transaction went through. A false negative in disease screening means a sick patient was sent home. A false positive in loan approval means a good customer got rejected. A false positive in spam filtering means an important email was buried.

The metric you choose decides *which* mistakes you are willing to tolerate.

| Mistake | What Happened | Who Suffers | Cost |
|---|---|---|---|
| False Negative in Fraud | Fraud passed through | Business | Financial loss |
| False Negative in Diagnosis | Sick patient sent home | Patient | Health risk |
| False Positive in Loan | Good customer rejected | Customer + Bank | Lost revenue |
| False Positive in Spam | Legitimate email hidden | User | Missed communication |

---

## Why accuracy fails on imbalanced data

Consider a dataset with 990 non-fraud and 10 fraud cases. A model that predicts everything as "not fraud" achieves:

**Accuracy = 990/1000 = 99%**

That looks excellent. But Recall on the positive class = 0. Zero frauds caught. The metric lied because it never asked *which* class the model got right, just *how many* in total.

Accuracy is only trustworthy when:
- Classes are roughly balanced
- The cost of a false positive is approximately equal to the cost of a false negative
- No class carries disproportionate business weight

It works perfectly for digit recognition, cats vs dogs, or balanced multi-class problems. It silently fails everywhere else.

---

## Everything comes from the confusion matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually positive** | TP — caught the thing | FN — missed it |
| **Actually negative** | FP — false alarm | TN — correctly ignored |

**Precision** = TP / (TP + FP) — Of everything flagged, how much was real?
**Recall** = TP / (TP + FN) — Of everything real, how much did we catch?
**F1** = harmonic mean of Precision and Recall — Are we catching positives without too many false alarms?

F1 exists because Accuracy ignores the minority class entirely. Precision and Recall each optimize one thing at the cost of the other. F1 asks for both at once.

---

## F1 also has limits

F1 treats Precision and Recall as equally important. That is not always true.

In medical diagnosis, missing a sick patient (low Recall) can be catastrophic even if it means more false alarms (lower Precision). You may deliberately sacrifice Precision to maximize Recall. F1 will not push you there on its own.

Similarly, in fraud detection you might set the threshold very low — flagging more cases, accepting more false positives — because catching every real fraud matters more than annoying some legitimate customers.

---

## Threshold dependence

A classifier does not output class labels — it outputs probabilities. The threshold decides where "positive" begins.

| Threshold | Effect |
|---|---|
| High (0.7+) | Fewer positives predicted — Precision up, Recall down |
| Low (0.3-) | More positives predicted — Recall up, Precision down |

The same trained model produces completely different Precision, Recall, F1, and Accuracy at different thresholds. Choosing 0.5 by default is an arbitrary decision, not a calibrated one.

---

## Macro vs weighted F1

In multiclass problems, F1 must be aggregated across classes.

- **Macro F1**: Averages F1 across all classes equally. Cares about rare classes just as much as common ones.
- **Weighted F1**: Weights each class F1 by its sample count. Gives more credit for getting the majority class right.

Use macro when every class matters equally (e.g., rare disease detection). Use weighted when class frequency should influence the score.

---

## Dataset and experiment design

A progressive single experiment rather than disconnected examples:

1. **Balanced dataset** (50:50) — baseline Logistic Regression, all metrics
2. **Imbalanced dataset** (95:5) — same model, compare metric behavior
3. **Threshold sweep** (0.2 to 0.8) — same model, watch Precision/Recall/F1 trade off
4. **Random Forest on imbalanced** — see how a stronger model interacts with the same metrics

---

## Results summary

| Dataset | Model | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Balanced | Logistic Regression | 0.5 | ~90% | ~0.90 | ~0.90 | ~0.90 |
| Imbalanced (95:5) | Logistic Regression | 0.5 | ~95% | — | ~0.00 | ~0.00 |
| Imbalanced (95:5) | Logistic Regression | 0.3 | ~88% | lower | higher | improves |
| Imbalanced (95:5) | Random Forest | 0.3 | ~91% | balanced | high | significantly higher |

---

## Observations

- On balanced data, Accuracy, Precision, Recall, and F1 all tell the same story. There is no meaningful difference in what metric you report.
- On imbalanced data, Accuracy jumps to 95% while F1 collapses to near zero. The model learned nothing about the minority class. Accuracy rewarded it anyway.
- Lowering the threshold on the same imbalanced model immediately improves Recall and F1 without retraining. The model's knowledge did not change; only what counts as a "positive" changed.
- Random Forest on imbalanced data with a tuned threshold significantly outperforms the baseline Logistic Regression — demonstrating that both model choice and threshold choice are decisions, not defaults.

---

## Practical guidelines

| Scenario | Preferred Metric | Reason |
|---|---|---|
| Balanced Classification | Accuracy | Equal class distribution, equal error costs |
| Fraud Detection | Recall / F1 | Missing fraud is expensive |
| Disease Detection | Recall | Missing a sick patient is critical |
| Spam Detection | Precision | Do not bury legitimate emails |
| Customer Churn | F1 | Balance between catching churners and avoiding unnecessary interventions |
| Multiclass (all classes matter) | Macro F1 | Rare classes should get equal weight |
| Multiclass (frequency matters) | Weighted F1 | Majority class performance should dominate |

---

## What I learned

Metrics do not evaluate models — they evaluate whether a model aligns with the business objective. The best metric is the one that reflects the real-world cost of making mistakes.

Accuracy is not wrong. It is just answering a different question than you often need answered. F1 is not always right either — it treats Precision and Recall as equally important when the business often does not.

The threshold is a decision. Not a default. Most beginners set it to 0.5 and move on. In practice, the threshold is where you encode your tolerance for each type of mistake, and it deserves as much thought as model selection.

---
