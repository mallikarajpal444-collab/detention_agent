import os
import joblib
import pandas as pd
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# -------------------------
# Load model locally
# -------------------------

MODEL_PATH = "detention_prediction_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("detention_prediction_model.pkl not found in project directory")

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# -------------------------
# Feature columns
# -------------------------

FEATURE_COLUMNS = [
    "dock_utilization",
    "arrival_hour",
    "arrival_day",
    "dock_pressure",
    "day_of_week_Monday",
    "day_of_week_Saturday",
    "day_of_week_Sunday",
    "day_of_week_Thursday",
    "day_of_week_Tuesday",
    "day_of_week_Wednesday",
    "congestion_level_Low",
    "congestion_level_Medium"
]

# -------------------------
# Routes
# -------------------------

@app.get("/")
def home():
    return {"message": "Detention Prediction API running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Add missing columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct order
    df = df[FEATURE_COLUMNS]

    probs = model.predict_proba(df)[:, 1]
    prediction = (probs > 0.35).astype(int)

    return {
        "detention_risk_probability": float(probs[0]),
        "prediction": int(prediction[0])
    }

# -------------------------
# Run server
# -------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)