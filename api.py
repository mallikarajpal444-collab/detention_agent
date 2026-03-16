from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("detention_model.pkl")

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

    # add missing columns
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # enforce correct order
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