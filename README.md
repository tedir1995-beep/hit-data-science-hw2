# Data Science Homework 2 - Regression and Classification Error Analysis

**Student:** tedi revelis<br>
**Student ID:** 313261919<br>
**Course:** Introduction to Data Science<br>
**Assignment:** Homework 2 - Regression and Classification: Error Analysis

This repository contains a reproducible regression and classification study using the frozen **Hugging Face Models Trending** dataset.

## Research questions

1. **Regression:** How accurately can model downloads be predicted from likes, age, update recency, tag count, task type, library, and access status?
2. **Classification:** Can we identify whether a model is gated, and where do classifiers make false-positive and false-negative errors?

The modeling frame uses the latest snapshot (1,000 unique models) to avoid treating repeated daily records of the same model as independent observations. The complete 156,000-row dataset remains in `data/raw/` and is verified by SHA-256.

## Repository contents

- `notebooks/huggingface_models_error_analysis.ipynb` - complete executable analysis, explanations, outputs, and charts.
- `reports/HOMEWORK_2_REPORT.md` - complete submission in organized Markdown form with all figures.
- `reports/OPEN_QUESTIONS.md` - concise written answers generated from the verified results.
- `reports/figures/` - exported figures used by the notebook and report.
- `data/raw/hf_models_snapshot.csv` - the frozen source dataset.
- `requirements.txt` - reproducible Python environment.

## Reproduce the analysis

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook notebooks\huggingface_models_error_analysis.ipynb
```

Open the notebook and choose **Run All** from the repository root. All random operations use seed `42`.

## Dataset integrity

- Rows: `156,000`
- Original columns: `13`
- Latest-snapshot modeling rows: `1,000`
- SHA-256: `A74C74F76E7FDF3095861AFA3F52484ACE9C1B6EDD83B24B66F87BAA9DDD2966`
- Kaggle source: [Hugging Face Models Trending](https://www.kaggle.com/datasets/zoupet/hugging-face-models-trending)

The raw CSV is preserved without modification so its checksum and value formatting remain reproducible.
