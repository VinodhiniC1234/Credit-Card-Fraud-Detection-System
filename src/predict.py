import joblib
import numpy as np

def load_model():
    model = joblib.load('models/fraud_model.pkl')
    return model

def predict_transaction(model, data):
    prediction = model.predict(data)
    return prediction