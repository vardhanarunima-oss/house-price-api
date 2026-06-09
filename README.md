# 🏠 House Price Prediction API

A machine learning API that predicts California house prices based on housing features.

## 🌐 Live Demo
[Try the API](https://house-price-api-0fke.onrender.com/docs)

## 🔧 Tech Stack
- Python
- Scikit-learn (Random Forest Regressor)
- FastAPI
- Uvicorn
- Pandas

## 📊 Model Performance
- Algorithm: Random Forest Regressor
- R² Score: 0.80+
- Mean Absolute Error: ~0.33

## 🚀 How to Run Locally

1. Clone the repo
   git clone https://github.com/vardhanarunima-oss/house-price-api.git
   cd house-price-api

2. Install dependencies
   pip install pandas scikit-learn fastapi uvicorn matplotlib seaborn

3. Train the model
   python model_v2.py

4. Run the API
   python -m uvicorn main:app --reload

5. Open in browser
   http://127.0.0.1:8000/docs

## 📬 API Usage

Send a POST request to `/predict` with:

{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.98,
  "AveBedrms": 1.02,
  "Population": 322.0,
  "AveOccup": 2.55,
  "Latitude": 37.88,
  "Longitude": -122.23
}

Response:
{
  "predicted_price": "$428,000.00",
  "note": "Based on California housing data"
}

## 💡 Features
- Trained on 20,640 real California housing records
- Random Forest model with 100 estimators
- Auto-generated interactive API docs via FastAPI
- Input validation with Pydantic