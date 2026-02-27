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

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the file here:
data/creditcard.csv

Coverage results: 
<img width="553" height="383" alt="image" src="https://github.com/user-attachments/assets/9af34019-66a5-4cab-9f7c-bda31e4bb9a7" />

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

## Status

Initial project setup and repository structure completed.

## Model Serving (Checkpoint 3)


### Run with Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

### API Endpoints

- `GET /health` → Returns service status  
- `POST /predict` → Returns fraud prediction  

### Example Request

```json
{
  "features": [0.0, -1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36,
  0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40,
  0.25, -0.01, 0.27, -0.11, 0.06, -0.14, -0.06, -0.06, 0.12, 149.62]
}
```

### Example Response

```json
{
  "fraud_prediction": 0
}
```