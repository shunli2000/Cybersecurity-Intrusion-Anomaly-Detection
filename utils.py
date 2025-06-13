import sys
sys.path.append('./models')

import pandas as pd
import numpy as np
import pickle
import torch
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 定義訓練時使用的欄位
FEATURES = [
    'timestamp', 'processId', 'threadId', 'parentProcessId', 'userId',
    'mountNamespace', 'processName', 'hostName', 'eventId', 'eventName',
    'stackAddresses', 'argsNum', 'returnValue', 'args'
]

def process_input_data(filepath):
    """
    Process the input CSV file to prepare it for model prediction
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # 只保留需要的欄位
        df = df[FEATURES]
        
        # Basic data cleaning
        df = df.fillna(0)  # Fill missing values with 0
        
        # Handle categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Convert to numpy array
        X = df.values
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
        
    except Exception as e:
        print(f"Error processing input data: {str(e)}")
        raise

def load_model(model_type):
    """
    Load the specified model from pickle file
    """
    try:
        model_path = os.path.join('models', f'{model_type}_model.pkl')
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            print(f"Model loaded successfully: {type(model)}")
            return model
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def predict_with_model(data, model_type):
    """
    Make predictions using the specified model
    """
    try:
        # Load the model
        model = load_model(model_type)
        
        # Make predictions
        print(f"Making predictions with model type: {model_type}")
        predictions = model.predict(data)
        print(f"Predictions shape: {predictions.shape}")
        
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        raise 