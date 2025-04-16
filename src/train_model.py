import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib
import json
from datetime import datetime
import os

def load_and_prepare_data(nrows=None):
    """Load and prepare the vehicle data"""
    print("Loading data...")
    df = pd.read_csv('data/vehicles.csv', nrows=nrows)
    print(f"Loaded {len(df)} records")
    
    # Basic cleaning
    df = df.dropna(subset=['price', 'year', 'odometer'])
    df = df[df['price'].between(100, 1000000)]  # Remove unrealistic prices
    df = df[df['year'] >= 1990]  # Remove very old vehicles
    df = df[df['odometer'] <= 500000]  # Remove vehicles with unrealistic mileage
    
    # Convert categorical variables to lowercase
    categorical_cols = ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 
                       'transmission', 'drive', 'type', 'paint_color']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.lower()
    
    # Calculate features
    current_year = datetime.now().year
    df['age'] = current_year - df['year']
    df['mileage_per_year'] = df['odometer'] / (df['age'] + 1)
    df['price_per_mile'] = df['price'] / df['odometer']
    
    # Define feature set
    numeric_features = ['year', 'odometer', 'age', 'mileage_per_year']
    categorical_features = ['manufacturer', 'model', 'condition', 'fuel', 'title_status',
                          'transmission', 'drive', 'type', 'paint_color']
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_features)
    
    # Get all feature columns
    dummy_columns = [col for col in df.columns if any(col.startswith(cat + '_') for cat in categorical_features)]
    features = numeric_features + dummy_columns
    
    # Split data
    X = df[features]
    y = df['price']
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, scaler, features

def train_model(X_train, X_val, y_train, y_val):
    """Train the LightGBM model"""
    print("Training model...")
    
    # Define model parameters
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model with callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=100)
    ]
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=callbacks
    )
    
    return model

def save_model_and_artifacts(model, scaler, features):
    """Save the model and related artifacts"""
    print("Saving model and artifacts...")
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save model
    model.save_model('../models/model.txt')
    
    # Save scaler
    joblib.dump(scaler, '../models/scaler.joblib')
    
    # Define feature categories
    numeric_features = ['year', 'odometer', 'age', 'mileage_per_year']
    categorical_features = [
        'manufacturer', 'model', 'condition', 'fuel', 'title_status',
        'transmission', 'drive', 'type', 'paint_color'
    ]
    
    # Save feature information
    feature_info = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': features
    }
    
    with open('../models/feature_info.json', 'w') as f:
        json.dump(feature_info, f)
    
    print("Model and artifacts saved successfully!")

def main():
    """Main function to train the model"""
    # Load and prepare data
    X_train, X_val, y_train, y_val, scaler, features = load_and_prepare_data(nrows=200000)
    
    # Train model
    model = train_model(X_train, X_val, y_train, y_val)
    
    # Save model and artifacts
    save_model_and_artifacts(model, scaler, features)
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main() 