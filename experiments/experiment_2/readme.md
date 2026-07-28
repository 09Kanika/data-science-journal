# Does PCA Always Improve Machine Learning Performance?

**Experiment 002**

Most people's mental model is: *"PCA reduces dimensions, so it must improve the
model."* This experiment tries to break that assumption on purpose by measuring
not just accuracy, but training time and feature count, across four levels of
compression.

---

## Question

Does PCA always improve machine learning performance?

I went in assuming the answer was "no," but I wanted to see *where exactly* it stops
helping, is it a hard cliff once you compress too far, or a slow, steady decline?
And does it affect every model the same way, or does it depend on how the model
actually uses its input features?

---

## Hypothesis

Reducing dimensions can remove noise and speed up training, but it may also discard
useful information, so I expected:

- **Mild PCA (95% variance)** to be roughly neutral or slightly positive: enough
  compression to cut some noise/redundancy without losing real signal.
- **Aggressive PCA (10 components)** to start showing cracks, especially for models
  that rely on subtler feature interactions.
- **Extreme PCA (2 components)** to visibly hurt accuracy for at least one model,
  since compressing 30 features down to 2 is a huge amount of information to
  throw away.
- **KNN** to benefit the most from moderate compression, since distance based
  methods suffer from the curse of dimensionality, fewer, denser dimensions should
  make its distance calculations more meaningful.
- **Random forest** to be the most resistant to *needing* PCA, since it already does
  its own implicit feature selection at each split, so I expected PCA to be neutral
  at best, and possibly harmful, since it removes information the trees could have
  used directly.
- **Training time** to drop for all models as features shrink, most obviously for
  KNN (distance calculations scale with dimensionality).

---

## Dataset

`load_breast_cancer` from `sklearn.datasets` — 569 samples, 30 numeric features,
binary target (malignant vs. benign).
---

## Methodology

1. Single `train_test_split` (80/20, stratified, `random_state=42`), reused for
   every condition.
2. Fit `StandardScaler` on training data (PCA is variance-based, so unscaled
   features would bias the components toward whichever feature has the largest
   raw numbers).
3. Fit a full `PCA` (all components) once, just to plot cumulative explained
   variance and find how many components hit the 95% threshold.
4. Run three models — **Logistic Regression, KNN, Random Forest** — under four
   conditions:
   - **No PCA** (all 30 scaled features)
   - **PCA 95% variance** (however many components that takes)
   - **PCA 10 components** (fixed)
   - **PCA 2 components** (fixed, also used for the scatter plot)
5. For every run, record: **accuracy**, **training time** (`time.perf_counter()`
   around `.fit()` only), and **number of features** going into the model.

---

## Results

*(numbers from `pca_experiment.ipynb`, `random_state=42`)*

It turned out 95% variance and 10 components landed on the exact same number of
components (10) for this dataset, so those two columns are identical.

### Accuracy

| Model | No PCA | PCA 95% (10 comp) | PCA 10 | PCA 2 |
|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9737 | 0.9737 | 0.9474 |
| KNN | 0.9561 | 0.9561 | 0.9561 | 0.9211 |
| Random Forest | 0.9561 | 0.9211 | 0.9211 | 0.9211 |

### Training Time (seconds)

| Model | No PCA | PCA 95% | PCA 10 | PCA 2 |
|---|---|---|---|---|
| Logistic Regression | 0.00540 | 0.00381 | 0.00399 | 0.00274 |
| KNN | 0.00087 | 0.00098 | 0.00103 | 0.00078 |
| Random Forest | 0.21504 | 0.19809 | 0.20103 | 0.18930 |

### Feature Count

| Condition | Features |
|---|---|
| No PCA | 30 |
| PCA 95% | 10 |
| PCA 10 | 10 |
| PCA 2 | 2 |

---

## Observations

- **PCA never helped accuracy here, not once, at any compression level, for any
  model.** Every single PCA condition matched or underperformed its own model's
  "No PCA" baseline. That's a stronger result than my hypothesis predicted, I
  expected at least mild PCA to be roughly neutral-to-positive somewhere.
- **Random Forest was hurt the most, and earliest.** It dropped from 0.9561 to
  0.9211 the moment *any* PCA was applied (even at 95% variance / 10 components),
  and didn't drop further at 2 components. This matches my hypothesis that trees
  do their own feature selection, compressing the input before the forest sees it
  removed information the trees would have used, and it hit that ceiling
  immediately rather than degrading gradually.
- **KNN was the most stable across moderate compression.** It held exactly at
  0.9561 through both 95% variance and 10 components, only dropping at the extreme
  2-component setting. That's the opposite of what I expected, I thought KNN
  would show the clearest *positive* case for PCA, but instead it just tolerated
  compression without benefiting from it.
- **Logistic Regression degraded the most gradually**, a small drop at 10
  components, a bigger drop at 2. It's the only model where each additional level
  of compression cost something.
- **Training time barely moved for any model**, and definitely didn't scale the
  way I expected. Random Forest's training time is dominated by growing many trees
  (100 by default), not by the number of input features, so cutting 30 features
  down to 2 barely nudged its time (0.215s → 0.189s). KNN's "training" is really
  just storing the data, so its time was tiny and near-flat regardless of PCA.
  Only Logistic Regression showed a training-time benefit that felt proportional
  to the compression (0.0054s → 0.0027s, roughly half).
- **The 2-component scatter plot showed real, visible overlap between malignant
  and benign points**, it wasn't a clean two-cluster separation. That overlap is
  the visual explanation for why every model's accuracy dropped at that setting,
  2 components genuinely aren't enough to cleanly separate the classes anymore.

---

## Key Takeaways

- **"PCA reduces dimensions, so it must improve the model" is false, at least on
  this dataset.** In every single case tested, PCA either matched or hurt
  accuracy, it never beat the full-feature baseline.
- **The cost of PCA depends entirely on how "wasteful" the original feature set
  was.** This dataset's 30 features apparently didn't have much redundant noise to
  strip out, cumulative explained variance needed 10 of 30 components just to
  hit 95%, meaning the "unnecessary" 20 features weren't purely noise, they were
  carrying real, if smaller, signal.
- **Tree-based models can lose more from PCA than they gain**, because PCA
  competes with (and can override) the model's own built-in feature selection.
  This is a genuine reason to think twice before defaulting to PCA in front of a
  Random Forest or Gradient Boosted Trees model.
- **Training time savings from PCA aren't automatic or uniform.** They depend on
  which part of the algorithm actually scales with feature count. For models where
  the bottleneck is elsewhere (like the number of trees in a forest), fewer
  features won't meaningfully speed things up.
- **The real value of PCA 2 components here wasn't accuracy, it was
  visualization.** Being able to see class overlap directly explains model
  behavior in a way a single accuracy number never could.

---

## Real-world Applications

- **Don't reach for PCA by default before tree-based models** (random forest,
  gradient boosted trees, XGBoost/LightGBM) in a production pipeline, this
  experiment suggests it's more likely to cost accuracy than save meaningful
  training time, since trees already handle irrelevant/redundant features
  reasonably well on their own.
- **PCA is most defensible when the actual bottleneck is dimensionality itself**,
  e.g., very high dimensional data (thousands of features, like genomic or
  text-embedding data) where distance-based models like KNN would otherwise
  struggle with the curse of dimensionality, or where storage/memory (not just
  training time) is the real constraint.
- **Use low-dimensional PCA (2–3 components) for exploration and communication,
  not for the final production model**, the scatter plot here was genuinely
  useful for *understanding* the data's separability, even though the same
  2-component transform made every model worse at actually predicting.
- **In a regulated or high-stakes context** (like this medical dataset), losing
  even ~3-6% accuracy for a marginal or non-existent speed gain is a bad trade, 
  this experiment is a concrete argument for validating PCA's actual effect
  on your specific model and dataset, rather than applying it as a default
  preprocessing step.

