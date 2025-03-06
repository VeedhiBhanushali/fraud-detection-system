from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import xgboost as xgb  # Ensure XGBoost is imported
from pydantic import BaseModel

app = FastAPI()

# Load trained model
xgb_model = joblib.load('src/xgb_model.pkl')  # Ensure correct path

# Define request body schema
class Transaction(BaseModel):
    amount: float
    hour: int
    risk_score: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    try:
        df = pd.DataFrame([transaction.dict()])

        # Ensure feature names match training data
        df.rename(columns={"amount": "Amount"}, inplace=True)

        # Convert input data to DMatrix
        dmatrix = xgb.DMatrix(df[['Amount', 'hour', 'risk_score']])

        # Predict fraud probability
        pred = xgb_model.predict(dmatrix)
        is_fraud = int(pred[0] > 0.5)

        return {"is_fraud": is_fraud}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

print(" Fraud Detection API is running.")
