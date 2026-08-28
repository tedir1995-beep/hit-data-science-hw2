# Quality Assurance Report

**Student:** טדי רבליס<br>
**Student ID:** 313261919

## Automated verification

| Check | Result |
|---|---|
| Dataset SHA-256 matches Homework 1 | Pass |
| Full dataset shape is `156,000 x 13` | Pass |
| Latest modeling snapshot contains 1,000 unique IDs | Pass |
| Notebook structure is valid `nbformat` | Pass |
| Code cells executed | `25 / 25` |
| Notebook error outputs | `0` |
| Notebook standard-error warning streams | `0` |
| Exported figures | `14` |
| Missing submission placeholders | `0` |
| Written report generated from notebook results | Pass |
| Fixed seed used for shuffled folds and stochastic models | `42` |

## Statistical verification

- Regression and classification preprocessing are fitted independently inside every cross-validation fold.
- Regression uses shuffled 5-fold `KFold` and original-unit out-of-fold residuals.
- Classification uses shuffled 5-fold `StratifiedKFold`, preserving approximately 8-9 gated examples per validation fold.
- All regression models share one declared feature set; all classification models share one declared feature set.
- The top 5% regression error group contains 50 of 1,000 observations.
- Threshold sensitivity covers every required threshold from `0.1` through `0.9`.
- Precision, recall, F1, F2, MCC, ROC-AUC, average precision, and the ROC curve are derived from out-of-fold predictions.
- The selected classification threshold is based on F2 because false negatives have the higher stated practical cost.

## Visual verification

All 14 exported PNG figures were opened and inspected after the final execution. The review found:

- no clipped titles, labels, legends, or annotations;
- no overlapping panels;
- legible axis labels and category names;
- a readable confusion matrix with values matching the written report;
- threshold, F-beta, and ROC curves consistent with the reported metrics;
- regression residual and extreme-error plots that preserve the heavy-tail behavior rather than hiding it through truncation.

## Re-execution record

The final notebook was executed from the repository root after the last code update. Every result and figure was regenerated, the generated Markdown report was refreshed, and the notebook was saved with all outputs embedded.
