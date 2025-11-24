# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import math
from data.preprocessing import preprocess_data
import logging
import os
import uvicorn

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XGBoost Traffic Prediction API")

MODEL_PATH = "data/xgboost_model.pkl"
model = None

# --- 1. LOAD MODEL ---
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# --- 2. DEFINE KNOWN JUNCTIONS (Coordinate Lookup) ---
# This maps the EXACT names your model knows to real-world coordinates.
# NOTE: These are approximate coordinates for Bengaluru. 
# You can refine them if you have exact training data coordinates.
KNOWN_JUNCTIONS = {
    "Intersection_100 Feet Road":       (12.9081, 77.6476),
    "Intersection_Anil Kumble Circle":  (12.9756, 77.6066),
    "Intersection_Ballari Road":        (13.0068, 77.5813),
    "Intersection_CMH Road":            (12.9790, 77.6408),
    "Intersection_Hebbal Flyover":      (13.0354, 77.5971),
    "Intersection_Hosur Road":          (12.9345, 77.6101),
    "Intersection_ITPL Main Road":      (12.9893, 77.7282),
    "Intersection_Jayanagar 4th Block": (12.9295, 77.5794),
    "Intersection_Marathahalli Bridge": (12.9552, 77.6984),
    "Intersection_Sarjapur Road":       (12.9245, 77.6493),
    "Intersection_Silk Board Junction": (12.9172, 77.6228),
    "Intersection_Sony World Junction": (12.9363, 77.6265),
    "Intersection_South End Circle":    (12.9368, 77.5757),
    "Intersection_Trinity Circle":      (12.9729, 77.6147),
    "Intersection_Tumkur Road":         (13.0292, 77.5399),
    "Intersection_Yeshwanthpur Circle": (13.0263, 77.5507)
}

def get_nearest_junction(lat, lng):
    """
    Finds the closest known junction to the input coordinates.
    """
    closest_name = None
    min_distance = float('inf')

    for name, (j_lat, j_lng) in KNOWN_JUNCTIONS.items():
        # Euclidean distance approximation (sufficient for local selection)
        # Calculate squared distance to avoid expensive sqrt()
        dist = (lat - j_lat)**2 + (lng - j_lng)**2
        
        if dist < min_distance:
            min_distance = dist
            closest_name = name
            
    return closest_name

# --- 3. INPUT MODELS ---
class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictionInput(BaseModel):
    model: str
    coordinates: Coordinates
    predictionTime: str
    event: str | None = None

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict/")
async def make_prediction(input_data: PredictionInput):
    logger.info(f"Received prediction request: {input_data.dict()}")

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not available")

    try:
        # A. Find the nearest mapped junction based on user click
        mapped_junction = get_nearest_junction(
            input_data.coordinates.lat, 
            input_data.coordinates.lng
        )
        
        logger.info(f"Mapped coordinates {input_data.coordinates} to '{mapped_junction}'")

        # B. Prepare DataFrame for preprocessing
        data = {
            'DateTime': [pd.Timestamp.now()], # Placeholder timestamp
            'latitude': [input_data.coordinates.lat],
            'longitude': [input_data.coordinates.lng],
            # Use the dynamically found junction name
            'JunctionName': [mapped_junction] 
        }
        input_df = pd.DataFrame(data)

        # C. Preprocess
        processed_df = preprocess_data(input_df)
        logger.info(f"Preprocessed data columns: {processed_df.columns.tolist()}")

        # D. Predict
        prediction_raw = model.predict(processed_df)
        predicted_vehicles = prediction_raw[0]
        logger.info(f"Raw prediction: {predicted_vehicles}")

        # E. Translate to Congestion/Speed
        # (Logic can be refined based on your specific traffic data ranges)
        congestion_level = 0.0
        congestion_label = "Low"
        avg_speed = 60.0

        if predicted_vehicles > 50:
            congestion_level = 0.5
            congestion_label = "Medium"
            avg_speed = 35.0
        if predicted_vehicles > 120:
            congestion_level = 0.9
            congestion_label = "Heavy"
            avg_speed = 12.0

        # Safety bounds
        predicted_vehicles = max(0, int(predicted_vehicles))
        congestion_level = max(0.0, min(1.0, congestion_level))

        # F. Response
        response_data = {
            "predictions": {
                "congestion": {
                    "level": round(congestion_level, 2), 
                    "label": congestion_label
                },
                "avgSpeed": round(avg_speed)
            },
            "alternativeRoute": None,
            "mappedLocation": mapped_junction # Optional: Let frontend know which location was used
        }
        
        return response_data

    except ValueError as ve:
        logger.error(f"Data preprocessing error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {ve}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        if "feature_names mismatch" in str(e):
             logger.error(f"Feature mismatch detail: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/")
def read_root():
    return {"message": "XGBoost Traffic Prediction API is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)