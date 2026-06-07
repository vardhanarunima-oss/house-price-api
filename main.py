from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Load the saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create the FastAPI app
app = FastAPI(title="House Price Predictor API")

# Define what input the API expects
class HouseFeatures(BaseModel):
    MedInc: float        # Median income
    HouseAge: float      # House age
    AveRooms: float      # Average rooms
    AveBedrms: float     # Average bedrooms
    Population: float    # Population
    AveOccup: float      # Average occupancy
    Latitude: float      # Latitude
    Longitude: float     # Longitude

# Root endpoint — just to test if API is running
@app.get("/")
def home():
    return {"message": "House Price Predictor API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to array for model
    data = np.array([[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude
    ]])
    
    prediction = model.predict(data)
    price = round(prediction[0] * 100000, 2)  # convert to actual dollars
    
    return {
        "predicted_price": f"${price:,.2f}",
        "note": "Based on California housing data"
    }