# Review Guide

The complete submission is in `notebooks/huggingface_models_error_analysis.ipynb`.

## Suggested review order

1. Read the title, dataset framing, and target definitions in Sections 1-2.
2. Check the four-model regression comparison and preferred-model reasoning in Section 3.
3. Review residual patterns, the top 5% errors, and error statistics in Section 4.
4. Check the three-model classification comparison in Section 5.
5. Review the confusion matrix, probability errors, feature regions, thresholds, F-beta, MCC, and ROC analysis in Section 6.
6. Read the integrated discussion and final reflection in Sections 7-8.
7. Use `ASSIGNMENT_CHECKLIST.md` to confirm every PDF requirement is covered.
8. Use `QA_REPORT.md` for the final execution and visual-verification record.

## Reproducibility checks

- Run the notebook from the repository root.
- Confirm every code cell completes without an error output.
- Confirm the dataset checksum matches the value in `README.md`.
- Confirm the exported figures exist in `reports/figures/`.
- Confirm the generated numerical answers in `OPEN_QUESTIONS.md` match the notebook outputs.

No merge into `main` is required for review; the complete work is intended to remain in a Draft Pull Request until approved.
