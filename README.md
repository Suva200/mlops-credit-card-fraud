# MLOps pipeline for credit card fraud detection

## Project Overview
 
This project demonstrates an end-to-end **MLOps pipeline for credit card fraud detection**. The goal is to build a reproducible machine learning system that includes **data preprocessing, model training, evaluation, experiment tracking, containerized deployment, and API-based inference**.
 
The project follows modern MLOps practices including modular code structure, automated testing, containerization with Docker, and serving predictions through a REST API built with FastAPI.
 
The final system allows users to send transaction features to an API endpoint and receive a prediction indicating whether the transaction is **fraudulent or legitimate**.
 
---
 
# Demo Video
 
🎥 **Project Demo Video**
 
The full demonstration of the system including **CI pipeline, Docker build, application startup, and API usage** is available here:
 
**Video Link:**

 
The demo video shows:
 
* GitHub repository overview
* Continuous Integration pipeline
* Docker container build and deployment
* FastAPI application startup
* API endpoints
* Example prediction request and response
 
---
 
# Problem Definition & Data
 
Credit card fraud detection is a **binary classification problem** where the objective is to detect fraudulent transactions among legitimate ones.
 
Financial institutions must detect fraud quickly to prevent financial losses and protect customers.
 
### Dataset
 
This project uses the **Credit Card Fraud Detection dataset**, which contains anonymized transaction features.
 
Key characteristics:
 
* Highly **imbalanced dataset**
* Fraud transactions represent a **very small percentage**
* Requires evaluation metrics suited for imbalance such as **F1-score**
 
### Target Variable
 
| Value | Meaning                |
| ----- | ---------------------- |
| 0     | Legitimate transaction |
| 1     | Fraudulent transaction |
 
---
 
# System Architecture
 
The system is designed as a modular machine learning pipeline.
 
### Pipeline Components
 
| Component       | Description                                   |
| --------------- | --------------------------------------------- |
| `data.py`       | Data loading and preprocessing                |
| `features.py`   | Feature engineering pipeline                  |
| `model.py`      | Machine learning model definitions            |
| `evaluation.py` | Model evaluation metrics                      |
| `train.py`      | Training pipeline and MLflow tracking         |
| `inference.py`  | Model loading for prediction                  |
| `api.py`        | FastAPI service exposing prediction endpoints |
 
### Workflow
 
```
Dataset
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Best Model Saved (artifacts/)
   ↓
FastAPI Service
   ↓
Prediction API
```
 
The trained model is saved as an artifact and later used for inference through the API.
 
---
 
# MLOps Practices
 
This project applies several key MLOps principles.
 
### Modular Code Structure
 
The machine learning pipeline is divided into independent modules to improve maintainability and scalability.
 
### Experiment Tracking
 
MLflow is used to track:
 
* Model parameters
* Evaluation metrics
* Trained models
 
This enables reproducible experiments and comparison between models.
 
### Model Artifact Management
 
The best performing model is saved locally as an artifact.
 
```
artifacts/best_model.pkl
```
 
This artifact is later used by the API for inference.
 
### Containerization
 
Docker is used to package the application along with all dependencies. This ensures the system runs consistently across environments.
 
### Automated Testing
 
Unit tests are included to validate key components such as data loading and model evaluation.
 
---
 
# Monitoring & Reliability
 
Basic monitoring mechanisms are implemented to ensure the reliability of the application.
 
### Health Check Endpoint
 
The API includes a `/health` endpoint that allows monitoring tools to verify whether the service is running.
 
Example response:
 
```json
{
  "status": "ok"
}
```
 
### Logging
 
The application logs important events including:
 
* Model loading
* Prediction requests
* Prediction outputs
* Errors during inference
 
Logging helps identify issues and track system behavior.
 
### Error Handling
 
The system checks whether the trained model artifact exists before loading it. If the model is missing, the application raises an exception to prevent incorrect pred
