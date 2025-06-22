import sys
#sys.path.append('./models')

import pandas as pd
import numpy as np
import pickle

from pathlib import Path
import os
import xgboost

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

        # Extract timestamp and original userId for returning BEFORE any transformation
        original_info_df = df[['timestamp', 'userId']].copy()

        # Apply feature engineering logic based on provided rules
        df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
        df["parentProcessId"] = df["parentProcessId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
        df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
        df["mountNamespace"] = df["mountNamespace"].map(lambda x: 0 if x == 4026531840 else 1)
        df["returnValue"] = df["returnValue"].map(lambda x: 0 if x == 0 else (1 if x > 0 else 2))

        # Define the specific features for the model
        features_for_model = [
            "processId", "parentProcessId", "userId",
            "mountNamespace", "eventId", "argsNum", "returnValue"
        ]
        
        # Select the final features for the model
        df_model_features = df[features_for_model].copy()

        # Basic data cleaning on the final feature set
        df_model_features.fillna(0, inplace=True)
        '''
        # Convert to numpy array
        X = df_model_features.values
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        '''
        
        # Ensure original_info_df has the same row order and count as X_scaled
        return df_model_features, original_info_df
        
    except Exception as e:
        print(f"Error processing input data: {str(e)}")
        raise

def load_model(model_type):
    """
    Load the specified model from pickle file
    """
    try:
        model_path = os.path.join('models', f'{model_type}_beth_best.pkl')
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