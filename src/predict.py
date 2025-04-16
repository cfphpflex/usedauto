import os
import json
import pickle
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import joblib

def load_model():
    """Load the trained model and artifacts"""
    try:
        # Load model
        model_path = os.path.join('models', 'model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)

        # Load scaler
        scaler_path = os.path.join('models', 'scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Load feature info from JSON
        feature_info_path = os.path.join('models', 'feature_info.json')
        if not os.path.exists(feature_info_path):
            raise FileNotFoundError(f"Feature info file not found at {feature_info_path}")
        
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)

        print("Model and artifacts loaded successfully")
        return model, scaler, feature_info

    except Exception as e:
        print(f"Error loading model or artifacts: {str(e)}")
        return None, None, None

def clean_data(df, is_training=True):
    """Clean the dataset"""
    # Drop rows with missing prices during training
    if is_training:
        df = df.dropna(subset=['price'])
        # Filter out unreasonable prices
        df = df[(df['price'] >= 100) & (df['price'] <= 100000)]
    
    # Fill missing values
    df['odometer'] = df['odometer'].fillna(df['odometer'].mean() if is_training else 50000)
    df['year'] = df['year'].fillna(df['year'].median() if is_training else 2020)
    
    # Fill missing categorical values with 'unknown'
    categorical_cols = ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 
                       'transmission', 'drive', 'type', 'paint_color']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    return df

def engineer_features(df, is_training=True):
    """Engineer additional features"""
    # Convert categorical columns to lowercase
    for col in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']:
        if col in df.columns:
            df[col] = df[col].str.lower()
    
    # Calculate age
    df['age'] = 2024 - df['year']
    
    # Calculate mileage per year
    df['miles_per_year'] = df['odometer'] / (df['age'] + 1)
    
    # Only calculate price-related features during training
    if is_training:
        df['price_per_mile'] = df['price'] / (df['odometer'] + 1)
        df['price_per_year'] = df['price'] / (df['age'] + 1)
    
    # Create interaction features
    df['age_mileage'] = df['age'] * df['odometer']
    
    return df

def prepare_features(data):
    """
    Prepare features for prediction by converting categorical variables to dummy variables
    and engineering numerical features.
    """
    # Convert input to DataFrame if it's a dictionary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    print(f"Input data shape: {data.shape}")
    print(f"Input columns: {data.columns.tolist()}")
    
    # Load feature info from JSON
    feature_info_path = os.path.join('models', 'feature_info.json')
    if not os.path.exists(feature_info_path):
        raise FileNotFoundError(f"Feature info file not found at {feature_info_path}")
    
    with open(feature_info_path, 'r') as f:
        feature_info = json.load(f)
    
    # Get feature information
    feature_columns = feature_info.get('feature_columns', [])
    numeric_features = feature_info.get('numeric_features', [])
    categorical_features = feature_info.get('categorical_features', [])
    
    print(f"Expected features: {feature_columns}")
    
    # Convert categorical columns to lowercase
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].str.lower()
            data[col] = data[col].fillna('unknown')
    
    # Calculate derived features
    current_year = datetime.now().year
    data['age'] = current_year - data['year']
    data['miles_per_year'] = data['odometer'] / data['age'].clip(lower=0.1)
    data['age_mileage'] = data['age'] * data['odometer']
    
    # Create dummy variables for categorical features
    dummies = pd.get_dummies(data[categorical_features], prefix=categorical_features)
    print(f"Created dummy variables with shape: {dummies.shape}")
    
    # Combine numeric and dummy features
    features = pd.concat([data[numeric_features], dummies], axis=1)
    print(f"Combined features shape: {features.shape}")
    
    # Add missing columns with zeros
    for col in feature_columns:
        if col not in features.columns:
            print(f"Adding missing column: {col}")
            features[col] = 0
    
    # Select only the expected features in the correct order
    features = features[feature_columns]
    print(f"Final features shape: {features.shape}")
    
    return features

def predict_price(input_data, model=None, scaler=None, feature_info=None):
    """
    Predict the price of a vehicle based on its features.
    
    Args:
        input_data (dict or pd.DataFrame): Input data containing vehicle features
        model: Pre-loaded model object (optional)
        scaler: Pre-loaded scaler object (optional)
        feature_info: Pre-loaded feature information (optional)
    
    Returns:
        float: Predicted price or None if prediction fails
    """
    try:
        # Load model and artifacts if not provided
        if model is None or scaler is None or feature_info is None:
            model, scaler, feature_info = load_model()
            if model is None or scaler is None or feature_info is None:
                print("Error: Failed to load model and artifacts")
                return None
        
        # Prepare features
        X = prepare_features(input_data)
        if X is None or X.empty:
            print("Error: Failed to prepare features")
            return None
            
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction and ensure it's JSON serializable
        prediction = float(model.predict(X_scaled)[0])
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def main():
    """Main function to test the prediction"""
    # Check if input file is provided
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        try:
            # Load input data
            with open(input_file, 'r') as f:
                input_data = json.load(f)
            
            # Convert to DataFrame
            df = pd.DataFrame([input_data])
            
            # Load model and artifacts
            model, scaler, feature_info = load_model()
            
            # Preprocess features
            X = prepare_features(df)
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            
            # Print prediction
            print(f"Predicted price: ${prediction:,.2f}")
            
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default test vehicle if no input file
        test_vehicle = {
            'year': 2020,
            'manufacturer': 'tesla',
            'model': 'model 3',
            'condition': 'excellent',
            'odometer': 25000,
            'fuel': 'electric',
            'title_status': 'clean',
            'transmission': 'automatic',
            'drive': 'rwd',
            'type': 'sedan',
            'paint_color': 'white'
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([test_vehicle])
        
        # Load model and artifacts
        model, scaler, feature_info = load_model()
        
        # Preprocess features
        X = prepare_features(df)
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        # Print prediction
        print(f"Predicted price: ${prediction:,.2f}")

def preprocess_data(df, is_training=True):
    """Preprocess the data for model training or prediction"""
    # Convert categorical columns to lowercase
    categorical_cols = ['manufacturer', 'model', 'condition', 'fuel', 'title_status',
                       'transmission', 'drive', 'type', 'paint_color']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.lower()

    # Calculate interaction features
    df['age_mileage'] = df['age'] * df['odometer']

    # Get all possible categories for each categorical column
    category_mappings = {}
    for col in categorical_cols:
        if col in df.columns:
            category_mappings[col] = sorted(df[col].unique().tolist())

    # Create dummy variables
    dummies = pd.get_dummies(df[categorical_cols])

    # Combine with numeric features
    numeric_features = ['year', 'odometer', 'age', 'miles_per_year', 'age_mileage']
    X = pd.concat([df[numeric_features], dummies], axis=1)

    # Save feature names
    feature_names = list(X.columns)

    # During training, return features and target
    if is_training:
        y = df['price']
        return X, y

    # During prediction, ensure we have all the same features as during training
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Select only the columns that were present during training
    X = X[feature_names]

    return X

def predict_from_gui(vehicle_data):
    """
    Handle GUI payload and return prediction.
    Args:
        vehicle_data: dict containing vehicle information from GUI
    Returns:
        dict: {'prediction': float, 'error': str or None}
    """
    try:
        # Load model and artifacts
        model, scaler, feature_info = load_model()
        if model is None or scaler is None or feature_info is None:
            return {'prediction': None, 'error': 'Failed to load model and artifacts'}

        # Prepare features using the exact same function that works in improved_model.py
        X_pred = prepare_features(vehicle_data, is_training=False)
        X_pred_scaled = scaler.transform(X_pred)
        prediction = model.predict(X_pred_scaled)[0]

        return {'prediction': prediction, 'error': None}

    except Exception as e:
        return {'prediction': None, 'error': str(e)}

if __name__ == "__main__":
    main() 