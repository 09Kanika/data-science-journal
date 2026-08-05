# Does mean imputation always beat median imputation?

The common rule of thumb is simple: use the **mean** for symmetric data and the **median** for skewed data. Rather than accepting that advice, this experiment evaluates whether it actually improves imputation quality and downstream model performance.

## Hypothesis

- Median should perform better on highly skewed features.
- Mean and median should perform similarly on near normal features.
- Any difference in model accuracy should be much smaller than the difference in imputation quality.

## Dataset

`load_breast_cancer` (569 samples, 30 features). Since the dataset contains no missing values, 15% of values were removed completely at random (MCAR) so the original values could be used as ground truth.

## Methodology

1. Measure feature skewness.
2. Inject 15% MCAR missing values.
3. Train separate mean and median imputation pipelines.
4. Compare imputation quality using MAE against the hidden true values.
5. Compare logistic regression, KNN and random forest accuracy after imputation.

## Results

- Median produced lower MAE on **21/30** features.
- Mean performed better on **9/30** features.
- Relative improvement showed a moderate positive relationship with skewness, while raw MAE was heavily affected by feature scale.
- Model accuracy changed very little:
  - logistic regression: Tie
  - random forest: Tie
  - KNN: Mean slightly higher

## Observations

- Median generally produced better imputations for skewed features, but not universally.
- Raw MAE alone was misleading because feature scales differed substantially.
- Better imputation quality did **not** consistently translate into better model accuracy.
- Skewness is a useful guideline, not a guarantee.

## Key Takeaways

- Choose an imputation strategy based on the feature distribution rather than habit.
- Evaluate imputation quality and downstream model performance separately.
- Mean and median often have only a small effect on final model accuracy when missingness is modest.

## Real-world Applications

- Use median for strongly skewed numerical features.
- Use mean for approximately symmetric distributions.
- Consider hybrid, column wise imputation strategies instead of one global rule.
