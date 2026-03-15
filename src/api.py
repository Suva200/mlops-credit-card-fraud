from fastapi import FastAPI
from src.schemas import PredictionRequest, PredictionResponse
from src.inference import ModelLoader
import logging
import joblib

#  Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

#  App & model initialization 
app = FastAPI(title="Credit Card Fraud Detection API")

logger.info("Loading fraud detection model...")
model = ModelLoader.load_model()
logger.info("Model loaded successfully.")


#  Routes 
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    logger.info("Prediction request received")
    prediction = model.predict([request.features])
    logger.info(f"Prediction result: {prediction[0]}")
    return {"fraud_prediction": int(prediction[0])}