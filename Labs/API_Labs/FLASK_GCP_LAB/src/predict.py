import numpy as np
import joblib
import os
import pandas as pd
from train import run_training

def load_model():
    """Load the trained model and feature names"""
    model_path = "../model/model_svm.pkl"
    try:
        model_info = joblib.load(model_path)
        return model_info['model'], model_info['feature_names']
    except FileNotFoundError:
        print("Model not found, training new model...")
        run_training()
        model_info = joblib.load(model_path)
        return model_info['model'], model_info['feature_names']

# Load model and feature names
model, feature_names = load_model()

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    """Make prediction using loaded model"""
    # Create input data as pandas DataFrame with feature names
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], 
        columns=feature_names
    )
    prediction = model.predict(input_data)
    return prediction[0]


if __name__ == "__main__":
    # Test the prediction
    try:
        test_pred = predict_iris(5.1, 3.5, 1.4, 0.2)
        print(f"Test prediction: {test_pred}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")