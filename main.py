# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from data.preprocessing import preprocess_data # Import updated function
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XGBoost Traffic Prediction API")
 
MODEL_PATH = "data/xgboost_model.pkl"
model = None
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")

# --- UPDATED INPUT MODEL ---
# Temporarily accept JunctionName instead of coordinates
# Expect coordinates now, matching the frontend
class Coordinates(BaseModel):
    lat: float
    lng: float

class PredictionInput(BaseModel):
    model: str
    coordinates: Coordinates # Expects the coordinates object
    predictionTime: str
    # Add other optional fields if needed, like event for CatBoost
    event: str | None = None # Optional event field

# ... (the rest of the main.py code remains the same for now) ...

@app.post("/predict/")
async def make_prediction(input_data: PredictionInput):
    logger.info(f"Received prediction request: {input_data.dict()}")

    if model is None:
        raise HTTPException(status_code=500, detail="Model is not available")

    try:
        # --- a. Convert input to DataFrame (Using coordinates) ---
        # We still need to figure out how to map coordinates to Junction features
        # for the current model. TEMPORARY DUMMY JUNCTION:
        data = {
            'DateTime': [pd.Timestamp.now()], # Placeholder
            'latitude': [input_data.coordinates.lat],
            'longitude': [input_data.coordinates.lng],
            # !! IMPORTANT TEMPORARY STEP !!
            # The model expects one-hot encoded Junctions. We MUST map lat/lon
            # back to a JunctionName OR retrain the model with lat/lon.
            # For now, we'll hardcode one to prevent errors, but the prediction
            # will likely be nonsensical until this mapping is done.
            'JunctionName': ['Intersection_Trinity Circle'] # HARDCODED - NEEDS FIXING
        }
        input_df = pd.DataFrame(data)

        # --- b. Preprocess the input data ---
        processed_df = preprocess_data(input_df)
        logger.info(f"Preprocessed data for prediction: \n{processed_df}")
        logger.info(f"Columns sent to model: {processed_df.columns.tolist()}") # Log columns for debugging

        # --- c. Make prediction ---
        prediction_raw = model.predict(processed_df)
        predicted_vehicles = prediction_raw[0]
        logger.info(f"Raw model prediction (vehicles): {predicted_vehicles}")

        # --- d. Translate prediction (Remains the same) ---
        congestion_level = 0.0
        congestion_label = "Low"
        avg_speed = 60
        if predicted_vehicles > 50:
            congestion_level = 0.5; congestion_label = "Medium"; avg_speed = 30
        if predicted_vehicles > 100:
            congestion_level = 0.85; congestion_label = "Heavy"; avg_speed = 15
        predicted_vehicles = max(0, predicted_vehicles)
        congestion_level = max(0.0, min(1.0, congestion_level))
        avg_speed = max(0, avg_speed)

        # --- e. Format the response (Remains the same) ---
        response_data = {
            "predictions": {
                "congestion": {"level": round(congestion_level, 2), "label": congestion_label},
                "avgSpeed": round(avg_speed)
            },
            "alternativeRoute": None
        }
        logger.info(f"Sending response: {response_data}")
        return response_data

    except ValueError as ve: # Catch errors from preprocessing
        logger.error(f"Data preprocessing error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input data: {ve}")
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Log the feature mismatch error specifically if it happens again
        if "feature_names mismatch" in str(e):
             logger.error(f"Feature mismatch detail: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.get("/")
def read_root():
    return {"message": "XGBoost Traffic Prediction API is running!"}

if __name__ == "__main__":
    import os, uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

