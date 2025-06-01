from fastapi import FastAPI
from pydantic import BaseModel, conlist
import joblib
import pandas as pd
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
Instrumentator().instrument(app).expose(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/api.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load model and encoder
try:
    model = joblib.load("api/output/model.pkl")
    col_transf = joblib.load("api/output/label_encoder.joblib")
    logger.info("Model and label encoder loaded successfully.")
    logger.info("Label classes: %s", col_transf.classes_)
except Exception as e:
    logger.error("Failed to load model or label encoder: %s", str(e))
    raise e

# Mapping gestures to game actions
gesture_to_action = {
    "like": "move_up",
    "dislike": "move_down",
    "peace": "move_left",
    "peace_inverted": "move_right",
    "stop": "pause"
}

# Request schema
class HandGestureRequest(BaseModel):
    landmarks: conlist(float, min_length=63, max_length=63)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Hand Gesture Model API is running."}

@app.get("/health")
def read_status():
    logger.info("Health check endpoint accessed")
    return {"status": "ok"}

@app.post("/predict")
def predict(request: HandGestureRequest):
    logger.info("Prediction endpoint accessed")
    try:
        # Prepare input
        input_df = pd.DataFrame([request.landmarks])
        logger.info("Received data for prediction: %s", input_df.values.tolist())

        # Prediction
        pred = model.predict(input_df)
        proba = model.predict_proba(input_df)  # Get probabilities

        # Decode label
        gesture = col_transf.inverse_transform(pred)[0]
        action = gesture_to_action.get(gesture, "unknown")

        # Logging
        logger.info("Prediction result: Gesture='%s' → Action='%s'", gesture, action)
        logger.info("Prediction probabilities: %s", proba.tolist())

        return {
            "gesture": gesture,
            "action": action,
            "probabilities": proba.tolist(),
            "classes": col_transf.classes_.tolist()
        }
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": "An error occurred during prediction."}
