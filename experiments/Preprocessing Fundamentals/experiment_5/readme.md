# Is removing outliers always a good idea?

Most tutorials say "remove outliers" as if it's a mandatory preprocessing step. This experiment tests whether blindly removing all outliers actually improves model performance — or whether it depends on the type of outlier and the model.

---

## Problem Statement

Outlier removal is one of the most misunderstood steps in data preprocessing. The common advice — *"just remove anything beyond 1.5× IQR"* — ignores a critical distinction:

- **Erroneous outliers**: Data entry mistakes, sensor failures, measurement errors. These should be removed.
- **Legitimate outliers**: Real, valid extreme values that reflect genuine variation in the data. Removing these loses information.

Blindly removing all outliers can hurt your model just as much as keeping corrupted ones.

---

## Why This Experiment Exists

Every data science course covers outlier detection but rarely answers the follow-up question: **then what?**

I wanted to see with real numbers:
1. How much does outlier removal actually change model performance?
2. Does the effect depend on the algorithm?
3. What happens when you only remove the erroneous ones?

---

## Hypothesis

- **Linear Regression** will be the most sensitive to outliers because it minimises squared error, which disproportionately penalises extreme residuals.
- **Random Forest** will be largely unaffected because bagging and averaging naturally dilute the influence of any single extreme point.
- **XGBoost** will show a moderate improvement as gradient boosting is iterative and can overfit to outliers in early boosting rounds, but its regularisation provides some protection.
- Removing only **erroneous** outliers will improve performance more cleanly than removing all extreme values indiscriminately.

---

## Dataset

**California Housing** (`fetch_california_housing` from scikit-learn)

| Property | Value |
|---|---|
| Samples | 20,640 |
| Features | 8 numerical |
| Target | Median house value (in $100,000s) |
| Source | sklearn built-in, no download needed |

Chosen because it has a continuous target, is available without any download, and naturally contains both legitimate extreme values (very expensive coastal properties) and some suspicious entries (income or occupancy values that seem implausibly large).

---

## Results

| Model | Baseline MAE | Clean MAE | Δ MAE | Baseline R² | Clean R² | Δ R² |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.5332 | 0.5122 | −0.0210 | 0.5758 | 0.5025 | −0.0733 |
| Random Forest | 0.3275 | 0.3290 | +0.0015 | 0.8051 | 0.8034 | −0.0017 |
| XGBoost | 0.3026 | 0.3013 | −0.0013 | 0.8358 | 0.8396 | +0.0038 |


---

## Discussion

**Linear Regression changed the most.**
It minimises Mean Squared Error, which squares each residual. A single prediction 3× off the true value contributes 9× the loss of a normal residual. Erroneous outliers in the target variable directly inflate this, so removing them has an outsized effect.

**Random Forest barely moved.**
Each tree in the forest sees a random bootstrap sample, and the final prediction is an average across hundreds of trees. No single outlier can dominate all trees simultaneously, so the ensemble naturally dilutes their impact.

**XGBoost changed slightly.**
Gradient boosting builds trees sequentially and can focus its later iterations on hard-to-predict points, which outliers often are. Its regularisation (`reg_alpha`, `reg_lambda`) provides some protection, but it's not fully immune. The improvement after cleaning is real but modest.

---

## What I Learned

The insight isn't just "remove outliers" — it's **understand your outliers first**.

The IQR method flags anyone with an extreme value, but extreme doesn't mean wrong. In the California Housing dataset, a house in Palo Alto with a very high value is a legitimate data point, and removing it removes information. A house with `AveOccup = 1024` (average of 1,024 occupants per household) is almost certainly a data entry error, and removing it removes noise.

The split between these two categories matters enormously. And it requires domain knowledge, not just statistics.

---

## Industry Takeaways

- In production ML, outlier handling is part of the data validation pipeline, not a one-time preprocessing step.
- Robust loss functions (`Huber`, `MAE`) are often preferred over MSE in real-world regression precisely because they don't over penalise outliers.
- Tree based models are used in industry settings with messy, outlier-heavy data (fraud detection, sensor logs) partly because of this robustness.
- Always ask: *where did this value come from?* before removing it.

---
