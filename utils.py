import sys
#sys.path.append('./models')

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
        
        # Select features for the model
        df_model_features = df[FEATURES].copy() # Use a copy for modifications

        # Extract timestamp and userId for returning BEFORE any transformation that alters them
        # This assumes 'timestamp' and 'userId' are in df_model_features
        original_info_df = df_model_features[['timestamp', 'userId']].copy()

        # Basic data cleaning on df_model_features
        df_model_features.fillna(0, inplace=True)
        
        # Handle categorical columns in df_model_features
        categorical_columns = df_model_features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            df_model_features[col] = le.fit_transform(df_model_features[col].astype(str))
        
        X = df_model_features.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ensure original_info_df has the same row order and count as X_scaled
        return X_scaled, original_info_df
        
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
            
        with open('models/ifor_model.pkl', 'rb') as f:
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
        #model = load_model(model_type)
        
        # Make predictions
        print(f"Making predictions with model type: {model_type}")
        #predictions = model.predict(data)
        predictions = np.random.randint(0, 2, size=(data.shape[0],))  # Simulated predictions for testing
        print(f"Predictions shape: {predictions.shape}")
        
        return predictions
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        raise 