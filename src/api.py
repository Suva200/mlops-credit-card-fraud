from fastapi import FastAPI
from src.schemas import PredictionRequest, PredictionResponse
import joblib
from src.inference import ModelLoader



app = FastAPI(title="Credit Card Fraud Detection API")

## model = joblib.load("artifacts/best_model.pkl")
model = ModelLoader.load_model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {"fraud_prediction": int(prediction[0])}