# Choosing between one hot and label encoding

Categorical variables can't be fed directly into most ML algorithms, they need to be converted into numbers. The two most common ways to do this are **Label Encoding** and **One Hot Encoding**, and picking the wrong one can silently hurt your model.

---

## Question

**Does the choice of encoding method actually affect model performance , and does it depend on the algorithm?**

Most tutorials say "use one hot for nominal data, label encoding for ordinal." But I wanted to see this play out with real numbers across different model types.

---

## The Core Difference

| | Label Encoding | One Hot Encoding |
|---|---|---|
| How it works | Assigns an integer to each category (Dog→0, Cat→1, Bird→2) | Creates a binary column per category |
| Implied order? | **Yes** (unintentionally) | No |
| Dimensionality | Same | Expands with cardinality |
| Best for | Tree based models, ordinal data | Linear models, nominal data |

**The key risk with label encoding:** it introduces a false ordinal relationship. A model may learn that `Bird (2) > Cat (1) > Dog (0)`, which is meaningless.

**The key risk with one hot encoding:** it can cause the **dummy variable trap** (multicollinearity) if you don't drop one column, and explodes dimensionality with high cardinality features.

---

## Hypothesis

Tree based models should handle label encoding fine as they split on thresholds, so arbitrary integer assignments don't mislead them. Linear and distance based models should suffer with label encoding because they'll treat the integers as meaningful magnitudes.

---

## Dataset

**Adult income dataset** (from UCI / `openml`) a classic classification task predicting whether income exceeds $50K.

Chosen because it has multiple nominal categorical features (occupation, relationship, country) with no natural ordering making the encoding choice genuinely consequential.

---

## Models used

- Logistic regression (sensitive to feature representation)
- K-nearest neighbors (distance based)
- Decision tree (threshold based should be robust)
- Random forest (ensemble of trees)

Each model trained twice once with label encoding, once with one hot encoding.

---

## Results

| Model | Label Encoding | One Hot Encoding | Difference |
|---|---:|---:|---:|
| Logistic Regression | ~0.8172 | ~0.8462 | 0.03 |
| KNN | ~0.8180 | ~0.8261 | 0.0081 |
| Decision Tree | ~0.8046 | ~0.8053 | 0.0007 |
| Random Forest | ~0.8495 | ~0.8472 | -0.0023 |

---

## Observations

- **Logistic regression** took the biggest hit from label encoding. Makes sense as it's learning a linear boundary and treats the encoded integers as real magnitudes.
- **KNN** also degraded because distance calculations become distorted when integers carry no actual proximity meaning.
- **Decision tree** was virtually unaffected as it only cares about whether a value is above or below a threshold, so the specific integer assignment doesn't matter.
- **Random forest** was mostly stable, with a small edge for one hot encoding.

---

## What I Learned

The rule of thumb holds up in practice: **tree based models are encoding agnostic for nominal data; everything else is not**.

But the more interesting takeaway is *why*. Label encoding doesn't just fail to help, it actively misleads models that learn from feature magnitudes or distances. One hot encoding removes the false ordering entirely by making each category its own binary feature.

The tradeoff is dimensionality. For high cardinality columns (hundreds of categories), one hot encoding becomes impractical, which is when you'd reach for target encoding or embedding layers instead.

---