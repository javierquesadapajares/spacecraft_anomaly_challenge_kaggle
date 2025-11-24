# Spacecraft Anomaly Detection – Kaggle Challenge

Solution for the **Kaggle Spacecraft Anomaly Detection Challenge**.
The goal is to detect anomalies in multivariate spacecraft telemetry time series using a combination of deep learning autoencoders and tabular ML models.

Final result: **19th / 84 teams in the final leaderboard** (with a peak position of **#9** during the competition.

---

## Table of Contents

- [Overview](#overview)
- [Dataset & Problem](#dataset--problem)
- [Approach](#approach)
  - [Deep Learning Models](#deep-learning-models)
  - [Tabular Models](#tabular-models)
- [Repository Structure](#repository-structure)
- [How to Run the Notebooks](#how-to-run-the-notebooks)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project explores different approaches for **anomaly detection in spacecraft telemetry**.
The work is structured around Jupyter notebooks where several models are implemented, tuned and compared:

- Autoencoder-based models (dense, convolutional, LSTM).
- Gradient boosting models (LightGBM, XGBoost).
- A TabPFN-based approach for tabular predictions.
- Generation of final Kaggle submissions based on the best-performing models.

The **convolutional autoencoder** achieved the best performance and was used in the final submission.

---

## Dataset & Problem

- Time series telemetry data from a spacecraft.
- Objective: detect anomalies in the telemetry, framed as an **anomaly detection / binary classification** problem at the time-step or window level (depending on the model).
- Evaluation metric: the one defined by the Kaggle competition (using the official public/private leaderboard).

The notebooks handle:

- Loading and preprocessing the telemetry.
- Creating appropriate train/validation/test splits.
- Training and evaluating several model families.
- Generating CSV submissions in the format required by Kaggle.

---

## Approach

### Deep Learning Models

Located mainly in the `notebooks/` folder:

- **`autoencoder.ipynb`**
  Baseline **dense autoencoder** for anomaly detection on telemetry features.

- **`convolucional.ipynb`**
  **Convolutional autoencoder** over the telemetry sequences.
  This model obtained the **best leaderboard performance** and was used for the final submission.

- **`lstm.ipynb`**
  **LSTM-based autoencoder**, modeling temporal dependencies explicitly through recurrent layers.

These notebooks include:

- Data preprocessing and normalisation.
- Model definition and training.
- Reconstruction-error-based anomaly scoring.
- Evaluation on validation data and submission generation.

### Tabular Models

Additional tabular experiments:

- **`notebook-lightgbm.ipynb`**
  LightGBM model on engineered tabular features as a strong gradient boosting baseline.

- **`notebook-xgboost.ipynb`**
  XGBoost model exploring tree-based ensembles for anomaly prediction.

- **`notebook-tabpfn.ipynb`**
  Experiments with **TabPFN** for tabular predictions on the processed telemetry features.

These notebooks focus on:

- Feature engineering / aggregation from telemetry.
- Training and tuning of the corresponding tabular model.
- Comparison of validation scores and submissions.

---

## Repository Structure

```text
/
├── main.ipynb
├── notebooks
│   ├── autoencoder.ipynb
│   ├── convolucional.ipynb
│   ├── lstm.ipynb
│   ├── notebook-lightgbm.ipynb
│   ├── notebook-tabpfn.ipynb
│   └── notebook-xgboost.ipynb
└── submissions
    ├── submission-lightgbm.csv
    ├── submission_conv.csv
    └── submission_lstm.csv
