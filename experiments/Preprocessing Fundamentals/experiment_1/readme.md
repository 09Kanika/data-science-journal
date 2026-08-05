# Does Feature Scaling Affect Every Machine Learning Algorithm?

The goal of this experiment isn't the accuracy number itself—it's understanding **why** some models improve after feature scaling while others don't.

---

## Question

**Does every machine learning model require feature scaling?**

Going in, my gut answer was **"probably not, but I've never actually watched it happen."**

Most tutorials simply tell you to apply `StandardScaler()` before training a model. Instead of taking that as a rule, I wanted to isolate the effect of feature scaling by keeping everything else the same:

- Same dataset
- Same train-test split
- Same evaluation metric
- Same models

The only thing that changes is **feature scaling**.

---

## Hypothesis

Before running the experiment, I expected models that rely on distances or optimization (like KNN and Logistic Regression) to benefit from feature scaling, while tree-based models would remain largely unaffected.

Let's find out if that actually happens.

---

## Dataset

For this experiment, I used the **Breast Cancer Wisconsin Dataset** available in `scikit-learn`.

I chose this dataset because it's clean, well-balanced, and lets me focus on understanding the effect of preprocessing instead of spending time cleaning the data.

---

## Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

Each model was trained twice:

1. Without feature scaling
2. After applying `StandardScaler`

---

## Results

| Model | Before Scaling | After Scaling | Change |
|--------|---------------:|--------------:|--------:|
| Logistic Regression | 0.9649 | 0.9825 | +0.0175 |
| KNN | 0.9123 | 0.9561 | +0.0439 |
| Decision Tree | 0.9123 | 0.9123 | 0.0000 |
| Random Forest | 0.9561 | 0.9561 | 0.0000 |

---

## Observations

A few things immediately stood out:

- **KNN** showed the biggest improvement after scaling.
- **Logistic Regression** also improved, although not as dramatically.
- **Decision Tree** produced exactly the same result.
- **Random Forest** also remained unchanged.

The results match the intuition behind how these algorithms work.

Distance-based models care about the scale of features because distances are directly affected by feature magnitudes. Tree-based models, however, split data using thresholds, so scaling the values doesn't change the order of the samples or where those splits occur.

---

## What I Learned

One thing this experiment reinforced is that preprocessing shouldn't be done blindly.

Feature scaling isn't a universal requirement—it's a tool that becomes important depending on how an algorithm works internally.

Understanding **why** a preprocessing step matters is far more valuable than simply adding it because every tutorial does.

---
