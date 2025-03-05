from kafka import KafkaProducer
import pandas as pd
import json
import time
import logging

logging.basicConfig(level=logging.INFO)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

df = pd.read_csv('data/test_processed_transactions.csv')

for _, row in df.iterrows():
    transaction = {
        'amount': row['Amount'],
        'hour': row['hour'],
        'risk_score': row['risk_score']
    }
    producer.send('fraud_transactions', value=transaction)
    logging.info(f"Sent transaction: {transaction}")
    time.sleep(1)  # Simulating real-time transactions
