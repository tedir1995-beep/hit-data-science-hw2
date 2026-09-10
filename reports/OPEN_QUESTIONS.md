# Homework 2 - Written Answers

**Student:** Tedi Revelis<br>
**Student ID:** 313261919<br>
**Dataset:** Hugging Face Models Trending

## Method and cross-validation

The complete dataset contains 156,000 rows and 13 original columns. Modeling uses the latest snapshot (1,000 unique models) so repeated daily observations do not violate independence. I selected 5-fold cross-validation: each regression fold contains about 200 validation models, and stratification leaves approximately 8-9 gated examples in each classification validation fold. Preprocessing is fitted inside every fold, and every reported prediction is out-of-fold. Fixed model settings were chosen before evaluation rather than tuned on these scores.

## Regression model comparison

The required Linear Regression, Decision Tree, and Random Forest models use the same feature set. KNN is included for bonus comparison. All models learn a log1p-transformed target and return predictions in original download units.

| Model | MAE | MSE | RMSE | R2 | Fold RMSE SD |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 1,873,999.9833 | 94,809,189,863,379.2812 | 9,737,001.0713 | 0.0755 | 7,263,313.5518 |
| Decision Tree | 2,057,339.2277 | 97,132,917,570,062.0312 | 9,855,603.3590 | 0.0529 | 7,236,446.9667 |
| KNN (bonus) | 1,960,761.2634 | 101,894,913,683,865.6250 | 10,094,301.0498 | 0.0064 | 7,463,382.6748 |
| Linear Regression | 2,567,832.7874 | 388,645,134,205,866.7500 | 19,714,084.6657 | -2.7897 | 15,891,607.5161 |

The preferred model is **Random Forest** with MAE 1,874,000, RMSE 9,737,001, and R2 0.076. Nonlinear models can capture thresholds and interactions; the linear model is easier to interpret but has higher structural bias. A single tree has higher variance, while Random Forest reduces variance through averaging. KNN is sensitive to sparse local neighborhoods.

## Regression error analysis

- Mean residual: 1,222,514
- Residual standard deviation: 9,664,784
- MAE: 1,874,000
- Skewness: 18.032
- Excess kurtosis: 408.629
- Top-5% absolute-error cutoff: 6,201,936 (50 observations)

Residuals are not homoskedastic: the cloud widens with predicted popularity, and exceptional blockbuster models are systematically under-predicted. Among pipeline sectors with at least 10 observations, `text-to-speech` has the largest observed rate of top-5% errors (2/13, 15.4%). All 50 extreme observations are analyzed individually in the notebook and exported to `reports/EXTREME_ERROR_ANALYSIS.csv` with error direction, scale, local sample size, and likely-source notes.

## Classification model comparison

| Model | ROC-AUC | Average Precision | Precision@0.5 | Recall@0.5 | F1@0.5 | MCC@0.5 | Balanced Accuracy@0.5 | Fold AUC SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8849 | 0.5938 | 0.7143 | 0.4545 | 0.5556 | 0.5547 | 0.7231 | 0.0719 |
| Decision Tree | 0.7577 | 0.3685 | 0.1884 | 0.5909 | 0.2857 | 0.2817 | 0.7369 | 0.0432 |
| Logistic Regression | 0.8320 | 0.2644 | 0.1921 | 0.7727 | 0.3077 | 0.3349 | 0.8116 | 0.0898 |

The preferred classifier is **Random Forest**, with out-of-fold ROC-AUC 0.885 and average precision 0.594. Average precision is emphasized because only 44 of 1000 models are gated. Downloads are excluded from the predictors because gating may affect download activity; the task is an error-analysis benchmark rather than a replacement for authoritative metadata.

## Confusion matrix and error implications

Nested threshold selection gives TN=919, FP=37, FN=15, TP=29. Each outer fold applies a threshold selected only from inner training folds: [0.3, 0.3, 0.4, 0.2, 0.4]. False negatives are most critical because they identify a restricted model as unrestricted, potentially breaking automated access or deployment workflows. False positives create extra review but do not incorrectly promise access.

## Probability, feature, and threshold analysis

There are 11 incorrect predictions with extreme class probability of at least 0.80. Because the probabilities are not calibrated, these are investigation candidates rather than literal confidence guarantees. Error rates vary by pipeline, library, and likes tier, especially in sparse subpopulations.

| Threshold | Precision | Recall | F1 | F2 | MCC | Predicted positives |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.1750 | 0.7955 | 0.2869 | 0.4654 | 0.3194 | 200.0000 |
| 0.2000 | 0.3056 | 0.7500 | 0.4342 | 0.5810 | 0.4437 | 108.0000 |
| 0.3000 | 0.4516 | 0.6364 | 0.5283 | 0.5882 | 0.5110 | 62.0000 |
| 0.4000 | 0.5952 | 0.5682 | 0.5814 | 0.5734 | 0.5628 | 42.0000 |
| 0.5000 | 0.7143 | 0.4545 | 0.5556 | 0.4902 | 0.5547 | 28.0000 |
| 0.6000 | 0.8000 | 0.3636 | 0.5000 | 0.4082 | 0.5266 | 20.0000 |
| 0.7000 | 0.9286 | 0.2955 | 0.4483 | 0.3421 | 0.5139 | 14.0000 |
| 0.8000 | 1.0000 | 0.1818 | 0.3077 | 0.2174 | 0.4186 | 8.0000 |
| 0.9000 | 1.0000 | 0.1364 | 0.2400 | 0.1648 | 0.3621 | 6.0000 |

The fixed-threshold table is exploratory. Its global F2 optimum is 0.3, but the unbiased threshold-policy estimate comes from nested selection. That policy has precision 0.439, recall 0.659, F1 0.527, F2 0.599, and MCC 0.512. Comparatively stable adjacent intervals are ['0.8-0.9', '0.5-0.6']; however, the high-threshold stable interval has low recall and is not operationally attractive. The largest joint metric changes occur in ['0.1-0.2', '0.2-0.3'].

## Final reflection

1. Regression fails most on blockbuster models and sectors with elevated top-5% error rates; classification fails in underrepresented pipeline/library and popularity regions.
2. Failures arise from missing or omitted data, model assumptions, and formulation choices, including a trending-only sample and an ordinarily observable classification target.
3. Improvements include leakage-safe longitudinal features, authoritative metadata repair, nested tuning, probability calibration, explicit error costs, and more gated examples.
4. Aggregate scores are insufficient. Error structure, minority-class behavior, fold stability, and leakage-safe threshold evaluation determine whether a model is actually reliable.
