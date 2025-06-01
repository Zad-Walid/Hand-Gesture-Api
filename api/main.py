
from fastapi import FastAPI
from pydantic import BaseModel, conlist
import joblib
import pandas as pd
import logging
import os
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI()
Instrumentator().instrument(app).expose(app)

model = joblib.load("api/output/model.pkl")
col_transf = joblib.load("api/output/label_encoder.joblib")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/api.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

gesture_to_action = {
    "like": "move_up",
    "dislike": "move_down",
    "peace": "move_left",
    "peace_inverted": "move_right",
    "stop": "pause"
}

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

        # Model prediction
        pred = model.predict(input_df)
        gesture = col_transf.inverse_transform(pred)[0]
        action = gesture_to_action.get(gesture, "unknown")

        logger.info("Prediction result: Gesture='%s' → Action='%s'", gesture, action)

        return {
            "gesture": gesture,
            "action": action
        }
    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        return {"error": "An error occurred during prediction."}
