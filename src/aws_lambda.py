import joblib
import json
import boto3
import pandas as pd
import os
import xgboost as xgb

xgb_model = joblib.load('../src/xgb_model.pkl')

SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")

def lambda_handler(event, context):
    try:
        data = json.loads(event['body'])
        df = pd.DataFrame([data])
        dmatrix = xgb.DMatrix(df[['amount', 'hour', 'risk_score']])
        pred = xgb_model.predict(dmatrix)
        is_fraud = int(pred[0] > 0.5)

        if is_fraud:
            sns = boto3.client('sns')
            sns.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message="🚨 Fraudulent transaction detected!",
                Subject="Fraud Alert"
            )
        
        return {"statusCode": 200, "body": json.dumps({"is_fraud": is_fraud})}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
