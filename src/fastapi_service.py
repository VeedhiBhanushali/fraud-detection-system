from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from lightgbm import LGBMClassifier  # Ensure LightGBM is imported
from pydantic import BaseModel

app = FastAPI()

# Load trained model
model = joblib.load('src/fraud_detection_model.pkl')  # Ensure correct path

# Define request body schema
class Transaction(BaseModel):
    amount: float
    hour: int
    dayofweek: int
    txns_last_24h: float
    amount_last_24h: float
    risk_score: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.dict()])

        # Ensure feature names match the model's training data
        df.rename(columns={"amount": "Amount"}, inplace=True)

        # Predict fraud using the trained model
        is_fraud = model.predict(df)[0]

        return {"is_fraud": int(is_fraud)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print(" ✅  Fraud Detection API is running.")