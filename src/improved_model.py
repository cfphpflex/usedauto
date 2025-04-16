import warnings
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json
import os
from joblib import dump as joblib_dump

warnings.filterwarnings('ignore')

class VehiclePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_info = None
        self.feature_names = None
        self.category_mappings = {}
        
    def clean_data(self, df, is_training=True):
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

    def engineer_features(self, df, is_training=True):
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

    def preprocess_data(self, df, is_training=True):
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
        self.category_mappings = {}
        for col in categorical_cols:
            if col in df.columns:
                self.category_mappings[col] = sorted(df[col].unique().tolist())

        # Create dummy variables
        dummies = pd.get_dummies(df[categorical_cols])

        # Combine with numeric features
        numeric_features = ['year', 'odometer', 'age', 'miles_per_year', 'age_mileage']
        X = pd.concat([df[numeric_features], dummies], axis=1)

        # Save feature names
        self.feature_names = list(X.columns)

        # During training, return features and target
        if is_training:
            y = df['price']
            return X, y

        # During prediction, ensure we have all the same features as during training
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        # Select only the columns that were present during training
        X = X[self.feature_names]

        return X

    def train(self, data):
        """Train the model and save artifacts"""
        try:
            print("Starting model training...")
            
            # Clean and engineer features
            data = self.clean_data(data, is_training=True)
            data = self.engineer_features(data, is_training=True)
            
            # Prepare features
            features, target = self.preprocess_data(data, is_training=True)
            print(f"Prepared features with shape: {features.shape}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            print("Split data into train and validation sets")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            print("Scaled features")
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
            print("Trained model")
            
            # Evaluate model
            val_score = np.mean(np.abs(self.model.predict(X_val_scaled) - y_val))
            print(f"Validation MAE: {val_score:.2f}")
            
            # Save feature information
            feature_info = {
                'feature_columns': features.columns.tolist(),
                'numeric_features': ['year', 'odometer', 'age', 'miles_per_year', 'age_mileage'],
                'categorical_features': [
                    'manufacturer', 'model', 'condition', 'fuel', 'title_status',
                    'transmission', 'drive', 'type', 'paint_color'
                ]
            }
            
            # Save artifacts
            os.makedirs('models', exist_ok=True)
            joblib_dump(self.model, os.path.join('models', 'model.joblib'))
            joblib_dump(self.scaler, os.path.join('models', 'scaler.joblib'))
            with open(os.path.join('models', 'feature_info.json'), 'w') as f:
                json.dump(feature_info, f, indent=4)
            print("Saved model artifacts")
            
            return val_score
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def predict(self, vehicle):
        """Make a price prediction for a single vehicle"""
        # Create DataFrame from vehicle data
        df = pd.DataFrame([vehicle])
        
        # Clean and engineer features
        df = self.clean_data(df, is_training=False)
        df = self.engineer_features(df, is_training=False)
        
        # Preprocess features
        X = self.preprocess_data(df, is_training=False)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X_scaled)[0]
        
        return prediction

    def save(self, directory='models'):
        """Save the model and artifacts"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Save model
            model_path = os.path.join(directory, 'model.joblib')
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")

            # Save scaler
            scaler_path = os.path.join(directory, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")

            # Save feature info for training
            feature_info_train = {
                'feature_columns': self.feature_names,
                'numeric_features': ['year', 'odometer', 'age', 'miles_per_year', 'age_mileage'],
                'categorical_features': ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 
                                       'transmission', 'drive', 'type', 'paint_color'],
                'category_mappings': self.category_mappings
            }
            feature_info_train_path = os.path.join(directory, 'feature_info_train.joblib')
            joblib.dump(feature_info_train, feature_info_train_path)
            print(f"Training feature info saved to {feature_info_train_path}")

            # Save feature info for prediction
            feature_info = {
                'feature_names': self.feature_names,
                'category_mappings': self.category_mappings
            }
            
            # Save as joblib
            feature_info_path = os.path.join(directory, 'feature_info.joblib')
            joblib.dump(feature_info, feature_info_path)
            print(f"Feature info saved to {feature_info_path}")

            # Save as JSON for readability
            feature_info_json_path = os.path.join(directory, 'feature_info.json')
            with open(feature_info_json_path, 'w') as f:
                json.dump(feature_info, f, indent=4)
            print(f"Feature info saved to {feature_info_json_path}")

            print("Model and artifacts saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving model and artifacts: {str(e)}")
            return False

    def load(self, directory='models'):
        """Load the model and artifacts"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Load model
            model_path = os.path.join(directory, 'model.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")

            # Load scaler
            scaler_path = os.path.join(directory, 'scaler.joblib')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")

            # Load feature info
            feature_info_path = os.path.join(directory, 'feature_info.json')
            if not os.path.exists(feature_info_path):
                raise FileNotFoundError(f"Feature info file not found at {feature_info_path}")
            with open(feature_info_path, 'r') as f:
                self.feature_info = json.load(f)
            print(f"Feature info loaded from {feature_info_path}")

            # Extract feature names and category mappings
            self.feature_names = self.feature_info['feature_columns']
            self.category_mappings = self.feature_info.get('category_mappings', {})

            print("Model and artifacts loaded successfully!")
            return self.model, self.scaler, self.feature_info
        except Exception as e:
            print(f"Error loading model and artifacts: {str(e)}")
            return None, None, None

    def prepare_features(self, data, is_training=False):
        """Prepare features for model training or prediction"""
        try:
            # Convert to DataFrame if dict
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            # Convert categorical columns to lowercase
            categorical_features = [
                'manufacturer', 'model', 'condition', 'fuel', 'title_status',
                'transmission', 'drive', 'type', 'paint_color'
            ]
            numeric_features = ['year', 'odometer']
            
            # Fill missing values
            for col in numeric_features:
                if col not in data.columns:
                    data[col] = 50000 if col == 'odometer' else 2020
                data[col] = data[col].fillna(50000 if col == 'odometer' else 2020)
            
            for col in categorical_features:
                if col in data.columns:
                    data[col] = data[col].str.lower() if data[col].dtype == 'object' else data[col]
                    data[col] = data[col].fillna('unknown')
                else:
                    data[col] = 'unknown'
            
            # Calculate derived features
            current_year = datetime.now().year
            data['age'] = current_year - data['year']
            data['miles_per_year'] = data['odometer'] / data['age'].replace(0, 1)  # Avoid division by zero
            data['age_mileage'] = data['age'] * data['miles_per_year']
            
            # Create dummy variables for categorical features
            dummy_features = pd.get_dummies(data[categorical_features], prefix=categorical_features)
            
            # Combine numeric and dummy features
            numeric_data = data[['year', 'odometer', 'age', 'miles_per_year', 'age_mileage']]
            features = pd.concat([numeric_data, dummy_features], axis=1)
            
            # Ensure all features are float type
            features = features.astype(float)
            
            print(f"Prepared features shape: {features.shape}")
            return features
        
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            raise

def load_and_explore_data():
    """Load and clean the dataset"""
    # Load data with a reasonable limit
    df = pd.read_csv('data/vehicles.csv', nrows=200000)
    
    # Keep only relevant columns
    columns_to_keep = ['year', 'manufacturer', 'model', 'condition', 'odometer', 
                      'fuel', 'title_status', 'transmission', 'drive', 'type', 
                      'paint_color', 'price']
    df = df[columns_to_keep]
    
    # Basic cleaning
    df = df.dropna(subset=['price'])  # Drop rows with missing prices
    df = df[(df['price'] >= 100) & (df['price'] <= 100000)]  # Filter price range
    
    # Convert categorical columns to lowercase
    categorical_cols = ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 
                       'transmission', 'drive', 'type', 'paint_color']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.lower()
    
    # Fill missing values
    df['odometer'] = df['odometer'].fillna(df['odometer'].mean())
    df['year'] = df['year'].fillna(df['year'].median())
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    return df

def prepare_features(data, is_training=True):
    """
    Prepare features for model training or prediction.
    Args:
        data: DataFrame for training or dict/Series for prediction
        is_training: bool, whether preparing for training or prediction
    """
    if is_training:
        df = data.copy()
        # Drop rows with missing prices for training data
        df = df.dropna(subset=['price'])
        
        # Fill missing values
        df['odometer'] = df['odometer'].fillna(df['odometer'].mean())
        df['year'] = df['year'].fillna(df['year'].median())
        
        numeric_features = ['year', 'odometer']
        categorical_features = ['manufacturer', 'model', 'condition', 'fuel', 
                              'title_status', 'transmission', 'drive', 'type', 
                              'paint_color']
        
        # Convert categorical columns to lowercase and fill missing values
        for col in categorical_features:
            df[col] = df[col].str.lower()
            df[col] = df[col].fillna('unknown')
        
        # Calculate derived features
        current_year = datetime.now().year
        df['age'] = current_year - df['year']
        df['miles_per_year'] = df['odometer'] / df['age'].clip(lower=0.1)
        df['age_mileage'] = df['age'] * df['odometer']
        
        numeric_features.extend(['age', 'miles_per_year', 'age_mileage'])
        
        # Create dummy variables for categorical features
        dummies = pd.get_dummies(df[categorical_features], prefix=categorical_features)
        
        # Store category mapping
        category_mapping = {}
        for col in categorical_features:
            category_mapping[col] = df[col].unique().tolist()
        
        # Combine features
        features = pd.concat([df[numeric_features], dummies], axis=1)
        
        # Store feature information
        feature_info = {
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'category_mapping': category_mapping,
            'all_features': features.columns.tolist()
        }
        
        return features, df['price'], feature_info
    
    else:
        # Handle single vehicle prediction
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame([data.to_dict()])
        
        # Load feature info
        with open('models/feature_info.json', 'r') as f:
            feature_info = json.load(f)
        
        numeric_features = feature_info['numeric_features']
        categorical_features = feature_info['categorical_features']
        category_mapping = feature_info['category_mapping']
        all_features = feature_info['all_features']
        
        # Convert categorical columns to lowercase
        for col in categorical_features:
            df[col] = df[col].str.lower()
            df[col] = df[col].fillna('unknown')
        
        # Calculate derived features
        current_year = datetime.now().year
        df['age'] = current_year - df['year']
        df['miles_per_year'] = df['odometer'] / df['age'].clip(lower=0.1)
        df['age_mileage'] = df['age'] * df['odometer']
        
        # Create dummy variables
        dummies = pd.get_dummies(df[categorical_features], prefix=categorical_features)
        
        # Ensure all training features are present
        for feature in all_features:
            if feature not in dummies.columns and feature not in numeric_features:
                dummies[feature] = 0
        
        # Combine features
        features = pd.concat([df[numeric_features], dummies[all_features[len(numeric_features):]]], axis=1)
        
        return features[all_features]

def train_model(X_train, y_train, X_val, y_val):
    """Train a RandomForest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    val_score = np.mean(np.abs(model.predict(X_val) - y_val))
    print(f"Validation MAE: {val_score:.2f}")
    
    return model

def save_model(model, scaler, feature_info):
    """Save the model, scaler, and feature information."""
    os.makedirs('models', exist_ok=True)
    
    # Save model using joblib
    joblib_dump(model, 'models/model.joblib')
    
    # Save scaler using joblib
    joblib_dump(scaler, 'models/scaler.joblib')
    
    # Save feature information
    with open('models/feature_info.json', 'w') as f:
        json.dump(feature_info, f)

def main():
    """Main function to train and save the model."""
    print("Loading and preparing data...")
    df = load_and_explore_data()
    
    print("Engineering features...")
    X, y, feature_info = prepare_features(df, is_training=True)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training model...")
    model = train_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("Saving model and artifacts...")
    save_model(model, scaler, feature_info)
    
    # Test prediction
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
        'paint_color': 'red'
    }
    
    X_pred = prepare_features(test_vehicle, is_training=False)
    X_pred_scaled = scaler.transform(X_pred)
    prediction = model.predict(X_pred_scaled)[0]
    
    print(f"\nPredicted price for test vehicle: ${prediction:,.2f}")

if __name__ == "__main__":
    main() 