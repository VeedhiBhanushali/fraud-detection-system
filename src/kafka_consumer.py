from kafka import KafkaConsumer
import joblib
import pandas as pd
import json
import logging
import xgboost as xgb

logging.basicConfig(level=logging.INFO)

# Load trained model
xgb_model = joblib.load('/Users/veedhibhanushali/fraud-detection-system/src/xgb_model.pkl')

consumer = KafkaConsumer(
    'fraud_transactions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    try:
        transaction = message.value
        df = pd.DataFrame([transaction])
        df.rename(columns={'amount': 'Amount'}, inplace=True)
        dmatrix = xgb.DMatrix(df[['Amount', 'hour', 'risk_score']]) 

        pred = xgb_model.predict(dmatrix)
        is_fraud = int(pred[0] > 0.5)
        logging.info(f"Transaction {transaction} classified as fraud: {is_fraud}")
    except Exception as e:
        logging.error(f"Error processing transaction {transaction}: {str(e)}")
