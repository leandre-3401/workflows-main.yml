# MLOps Lab — ML Pipeline (Data → Preprocessing → Training → Evaluation → CI/CD)

This repository contains my submission for the **MLOps lab**: building a complete and reproducible pipeline for **tweet sentiment classification** (positive / negative), including:
- modular Python scripts (data loading, preprocessing, training, evaluation),
- experiment tracking with **MLflow**,
- automation with **GitHub Actions** (CI/CT: Continuous Training),
- saving MLflow results as **artifacts** during CI runs.

---

## Project goals

- Structure an ML project clearly (separation of concerns).
- Build a robust preprocessing pipeline tailored to tweets (URLs, mentions, stopwords, lemmatization, etc.).
- Train and compare **two models**:
  - Logistic Regression
  - Naive Bayes
- Track experiments (parameters, metrics, artifacts) with **MLflow**.
- Automate pipeline execution on every `push` using **GitHub Actions**.

---

## Tech stack

- Python 3.9+
- pandas, scikit-learn
- nltk (NLP preprocessing)
- mlflow (experiment tracking)
- joblib (model serialization)
- GitHub Actions (CI/CD)




