# Homework 2 - Written Answers

**Student:** טדי רבליס<br>
**Student ID:** 313261919<br>
**Dataset:** Hugging Face Models Trending

## Method and cross-validation

The complete dataset contains 156,000 rows and 13 original columns. Modeling uses the latest snapshot (1,000 unique models) so repeated daily observations do not violate independence. I selected 5-fold cross-validation: each regression fold contains about 200 validation models, and stratification leaves approximately 8-9 gated examples in each classification validation fold. Preprocessing is fitted inside every fold, and every reported prediction is out-of-fold.

## Regression model comparison

The required Linear Regression, Decision Tree, and Random Forest models use the same feature set. KNN is included for bonus comparison. All models learn a log1p-transformed target and return predictions in original download units.

| Model | MAE | MSE | RMSE | R2 | Fold RMSE SD |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 1,873,999.9833 | 94,809,189,863,379.2656 | 9,737,001.0713 | 0.0755 | 7,263,313.5518 |
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
- Top-5% absolute-error cutoff: 6,137,024 (50 observations)

Residuals are not homoskedastic: the cloud widens with predicted popularity, and exceptional blockbuster models are systematically under-predicted. The most error-prone sufficiently populated pipeline sector is `text-ranking`. Extreme errors combine rare cases, omitted features, missing technical metadata, and model limitations. The strongly positive skewness and excess kurtosis show a heavy right tail.

## Classification model comparison

| Model | ROC-AUC | Average Precision | Precision@0.5 | Recall@0.5 | F1@0.5 | MCC@0.5 | Balanced Accuracy@0.5 | Fold AUC SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8913 | 0.5574 | 0.6923 | 0.4091 | 0.5143 | 0.5165 | 0.7004 | 0.0640 |
| Decision Tree | 0.7241 | 0.3564 | 0.1742 | 0.5227 | 0.2614 | 0.2476 | 0.7044 | 0.0641 |
| Logistic Regression | 0.8322 | 0.2593 | 0.1976 | 0.7500 | 0.3128 | 0.3353 | 0.8049 | 0.0972 |

The preferred classifier is **Random Forest**, with out-of-fold ROC-AUC 0.891 and average precision 0.557. Average precision is emphasized because only 44 of 1000 models are gated.

## Confusion matrix and error implications

At the F2-selected threshold of 0.2, the confusion matrix contains TN=889, FP=67, FN=12, TP=32. False negatives are most critical because they identify a restricted model as unrestricted, potentially breaking automated access or deployment workflows. False positives create extra review but do not incorrectly promise access.

## Probability, feature, and threshold analysis

There are 12 high-confidence errors at confidence >= 0.80. Their existence shows that some failures are systematic rather than simple threshold ambiguity. Error rates vary by pipeline and download tier, especially in sparse subpopulations.

| Threshold | Precision | Recall | F1 | F2 | MCC | Predicted positives |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 0.1777 | 0.7955 | 0.2905 | 0.4692 | 0.3228 | 197.0000 |
| 0.2000 | 0.3232 | 0.7273 | 0.4476 | 0.5818 | 0.4513 | 99.0000 |
| 0.3000 | 0.4333 | 0.5909 | 0.5000 | 0.5508 | 0.4796 | 60.0000 |
| 0.4000 | 0.5641 | 0.5000 | 0.5301 | 0.5116 | 0.5109 | 39.0000 |
| 0.5000 | 0.6923 | 0.4091 | 0.5143 | 0.4455 | 0.5165 | 26.0000 |
| 0.6000 | 0.8125 | 0.2955 | 0.4333 | 0.3385 | 0.4778 | 16.0000 |
| 0.7000 | 0.9000 | 0.2045 | 0.3333 | 0.2419 | 0.4195 | 10.0000 |
| 0.8000 | 1.0000 | 0.1136 | 0.2041 | 0.1381 | 0.3304 | 5.0000 |
| 0.9000 | 1.0000 | 0.0455 | 0.0870 | 0.0562 | 0.2087 | 2.0000 |

Lower thresholds improve recall but increase false positives; higher thresholds improve precision but increase false negatives. The selected operating point has precision 0.323, recall 0.727, F1 0.448, F2 0.582, and MCC 0.451. F-beta changes with beta because beta encodes the relative importance of recall.

## Final reflection

1. The regression model fails most on blockbuster models and rare pipeline sectors; the classifier fails in underrepresented task/popularity regions.
2. Failures arise from all three sources: missing or omitted data, model assumptions, and the trending-only/binary-target formulation.
3. Improvements include leakage-safe longitudinal features, authoritative metadata repair, nested tuning, probability calibration, explicit error costs, and more gated examples.
4. The central insight is that aggregate scores are insufficient. Error structure, minority-class behavior, fold stability, and threshold choice determine whether a model is actually reliable.
