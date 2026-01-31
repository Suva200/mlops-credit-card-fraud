# MLOps pipeline for credit card fraud detection

## Project Description

This project implements an end-to-end **MLOps pipeline for credit card fraud detection**, The focus is on building a **reproducible, modular, and production-oriented workflow**, covering data loading, preprocessing, model training, and experiment structure.

The problem is particularly challenging due to **extreme class imbalance**, where fraudulent transactions represent only **0.17%** of all records. The project emphasizes good engineering practices, leakage-free preprocessing, and appropriate evaluation metrics for imbalanced classification.

---

## Task Definition

Binary classification task to predict whether a credit card transaction is **fraudulent (1)** or **legitimate (0)**.

Key challenges addressed:

* Severe class imbalance
* High cost of false negatives (missed fraud)
* Tabular data with anonymized features

The baseline implementation focuses on correctness, reproducibility, and clarity rather than maximizing model performance.

---

## Dataset Source

* **Dataset:** Credit Card Fraud Detection
* **Source:** Kaggle (ULB Credit Card Dataset)

The dataset contains **284,807 transactions**, with **492 fraud cases (0.17%)** and **30 numerical features**.

---

## Team Roles & Responsibilities

Each team member is responsible for implementing a key component of the MLOps pipeline.

 * **Member 1 & 2 – Git Repository, Data & Preprocessing**

  * GitHub setup, project structure, dependency management using **UV**
  * *Code:* `pyproject.toml`, `uv.lock`, base structure
  * Data loading, cleaning, preprocessing, and train/test split
  * *Code:* `src/data.py`, `src/features.py`

 * **Member 3 & 4 – Model Training, Testing & Documentation**

  * Model implementation, training pipeline, evaluation
  * *Code:* `src/model.py`, `src/train.py`
  * Project documentation and unit tests
  * *Code:* `README.md`, `tests/`


