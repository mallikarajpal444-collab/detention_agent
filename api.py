import os
import requests
import joblib
import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Model location
MODEL_URL = "https://drive.google.com/uc?export=download&id=1ua2ubhE8bzCumTEd3v0hhiKATeYEiFjV"
MODEL_PATH = "detention_model.pkl"

# Download model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = joblib.load(MODEL_PATH)

# Feature columns expected by the model
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

    # Ensure correct column order
    df = df[FEATURE_COLUMNS]

    probs = model.predict_proba(df)[:, 1]
    prediction = (probs > 0.35).astype(int)

    return {
        "detention_risk_probability": float(probs[0]),
        "prediction": int(prediction[0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)