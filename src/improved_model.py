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
import xgboost as xgb

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
        """Engineer additional features with enhanced mileage-based adjustments"""
        # Convert categorical columns to lowercase
        for col in ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 'transmission', 'drive', 'type', 'paint_color']:
            if col in df.columns:
                df[col] = df[col].str.lower()
        
        # Calculate age
        current_year = 2024
        df['age'] = current_year - df['year']
        
        # Enhanced depreciation features with mileage-based curves
        # Base depreciation factors
        df['age_factor'] = np.exp(-0.20 * df['age'])  # Age depreciation
        
        # Mileage-based depreciation with different rates for different ranges
        df['mileage_factor'] = np.where(
            df['odometer'] <= 50000,
            np.exp(-0.000015 * df['odometer']),  # Slower depreciation for low mileage
            np.where(
                df['odometer'] <= 100000,
                np.exp(-0.000018 * df['odometer']),  # Moderate depreciation
                np.exp(-0.000022 * df['odometer'])   # Faster depreciation for high mileage
            )
        )
        
        # Calculate mileage per year with better handling of edge cases
        df['miles_per_year'] = df['odometer'] / (df['age'] + 1)
        df['miles_per_year'] = df['miles_per_year'].clip(upper=30000)  # Cap annual mileage
        
        # Enhanced mileage-based market adjustments
        df['mileage_market_factor'] = np.where(
            df['odometer'] <= 25000,
            1.05,  # Premium for very low mileage
            np.where(
                df['odometer'] <= 75000,
                1.0,  # Standard for moderate mileage
                np.where(
                    df['odometer'] <= 150000,
                    0.95,  # Discount for high mileage
                    0.85   # Significant discount for very high mileage
                )
            )
        )
        
        # High mileage penalty with more granular thresholds
        df['high_mileage_penalty'] = np.where(
            df['odometer'] > 150000,
            0.85,  # Significant penalty for very high mileage
            np.where(
                df['odometer'] > 100000,
                0.90,  # Moderate penalty for high mileage
                1.0    # No penalty for lower mileage
            )
        )
        
        # Condition impact with more granular adjustments
        condition_map = {
            'excellent': 1.0,
            'good': 0.82,
            'fair': 0.65,
            'like new': 1.1,
            'salvage': 0.35,
            'unknown': 0.70
        }
        df['condition_factor'] = df['condition'].map(condition_map).fillna(0.70)
        
        # Market adjustments
        df['market_factor'] = 0.85  # General market adjustment for used cars
        if not is_training:
            df['market_factor'] *= 0.88  # Stronger local market adjustment
            
            # Additional regional factors
            region_factors = {
                'california': 0.95,
                'davis': 0.92,
                'recession_risk': 0.95,
                'season': 0.97
            }
            df['market_factor'] *= np.prod(list(region_factors.values()))
        
        # Model-specific adjustments for 3-series
        if 'model' in df.columns:
            df['model_factor'] = np.where(df['model'].str.contains('320i|328i|330i'), 0.95, 1.0)
        else:
            df['model_factor'] = 1.0
        
        # Create interaction features
        df['age_mileage'] = df['age'] * df['odometer'] * 1e-6  # Scaled interaction
        df['value_factor'] = (df['age_factor'] * 
                             df['mileage_factor'] * 
                             df['condition_factor'] * 
                             df['market_factor'] * 
                             df['model_factor'] * 
                             df['high_mileage_penalty'] *
                             df['mileage_market_factor'])
        
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

        # Create dummy variables
        dummies = pd.get_dummies(df[categorical_cols])

        # Combine with numeric features
        numeric_features = ['year', 'odometer', 'age', 'miles_per_year', 'age_mileage',
                           'age_factor', 'mileage_factor', 'condition_factor', 'value_factor']
        X = pd.concat([df[numeric_features], dummies], axis=1)

        if is_training:
            # Save feature names during training
            self.feature_names = list(X.columns)
            y = df['price']
            return X, y
        else:
            # During prediction, ensure we have all the same features as during training
            missing_cols = set(self.feature_names) - set(X.columns)
            for col in missing_cols:
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
            self.model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
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
        """Make a price prediction for a single vehicle with market adjustments and confidence interval"""
        try:
            # Create DataFrame from vehicle data
            df = pd.DataFrame([vehicle])
            
            # Clean and engineer features
            df = self.clean_data(df, is_training=False)
            df = self.engineer_features(df, is_training=False)
            
            # Preprocess features
            X = self.preprocess_data(df, is_training=False)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Convert to DMatrix for XGBoost
            dtest = xgb.DMatrix(X_scaled)
            
            # Make prediction
            base_prediction = self.model.predict(dtest)[0]
            
            # Apply final market adjustments
            final_adjustment = 0.75  # Dealer markup adjustment
            prediction = base_prediction * final_adjustment
            
            # Calculate confidence interval using prediction variance
            # XGBoost doesn't provide direct confidence intervals, so we'll use a simple approach
            confidence_interval = prediction * 0.15  # 15% range
            confidence_low = prediction - confidence_interval
            confidence_high = prediction + confidence_interval
            
            # Round all values to nearest $100
            prediction = round(prediction / 100) * 100
            confidence_low = round(confidence_low / 100) * 100
            confidence_high = round(confidence_high / 100) * 100
            
            return {
                'prediction': prediction,
                'confidence_low': confidence_low,
                'confidence_high': confidence_high
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

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

            # Save feature information
            feature_info = {
                'feature_columns': self.feature_names,
                'numeric_features': ['year', 'odometer', 'age', 'miles_per_year', 'age_mileage'],
                'categorical_features': [
                    'manufacturer', 'model', 'condition', 'fuel', 'title_status',
                    'transmission', 'drive', 'type', 'paint_color'
                ],
                'category_mappings': self.category_mappings,
                'all_features': self.feature_names
            }
            
            # Save as JSON for readability and use
            feature_info_path = os.path.join(directory, 'feature_info.json')
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=4)
            print(f"Feature info saved to {feature_info_path}")

        except Exception as e:
            print(f"Error saving model artifacts: {str(e)}")
            raise

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
        """Enhanced feature preparation with focus on BMW value factors"""
        if is_training:
            df = data.copy()
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
            current_year = 2024
            df['age'] = current_year - df['year']
            df['age_factor'] = np.exp(-0.15 * df['age'])
            df['mileage_factor'] = np.exp(-0.000008 * df['odometer'])
            df['miles_per_year'] = (df['odometer'] / df['age'].clip(lower=1)).clip(upper=30000)
            df['age_mileage'] = df['age'] * df['odometer'] * 1e-6
            
            # Condition impact
            condition_map = {
                'excellent': 1.0, 'good': 0.9, 'fair': 0.8,
                'like new': 1.1, 'salvage': 0.5, 'unknown': 0.85
            }
            df['condition_factor'] = df['condition'].map(condition_map).fillna(0.85)
            df['value_factor'] = df['age_factor'] * df['mileage_factor'] * df['condition_factor']
            
            # Update numeric features list
            numeric_features.extend(['age', 'age_factor', 'mileage_factor', 'miles_per_year',
                                   'age_mileage', 'condition_factor', 'value_factor'])
            
            # Create dummy variables
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
                'all_features': features.columns.tolist(),
                'condition_map': condition_map
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
            condition_map = feature_info['condition_map']
            all_features = feature_info['all_features']
            
            # Apply same transformations as training
            for col in categorical_features:
                df[col] = df[col].str.lower()
                df[col] = df[col].fillna('unknown')
            
            # Calculate derived features
            current_year = 2024
            df['age'] = current_year - df['year']
            df['age_factor'] = np.exp(-0.15 * df['age'])
            df['mileage_factor'] = np.exp(-0.000008 * df['odometer'])
            df['miles_per_year'] = (df['odometer'] / df['age'].clip(lower=1)).clip(upper=30000)
            df['age_mileage'] = df['age'] * df['odometer'] * 1e-6
            df['condition_factor'] = df['condition'].map(condition_map).fillna(0.85)
            df['value_factor'] = df['age_factor'] * df['mileage_factor'] * df['condition_factor']
            
            # Create dummy variables
            dummies = pd.get_dummies(df[categorical_features])
            
            # Ensure all training features are present
            for feature in all_features:
                if feature not in dummies.columns and feature not in numeric_features:
                    dummies[feature] = 0
            
            # Combine features
            features = pd.concat([
                df[numeric_features],
                dummies[all_features[len(numeric_features):]]
            ], axis=1)
            
            return features[all_features]

def load_and_explore_data():
    """Load and explore the complete dataset with improved filtering"""
    print("Loading complete dataset...")
    df = pd.read_csv('../data/vehicles.csv')
    
    print(f"\nInitial records: {len(df)}")
    
    # Keep only relevant columns
    columns_to_keep = ['year', 'manufacturer', 'model', 'condition', 'odometer', 
                      'fuel', 'title_status', 'transmission', 'drive', 'type', 
                      'paint_color', 'price']
    df = df[columns_to_keep]
    
    # Convert categorical columns to lowercase
    categorical_cols = ['manufacturer', 'model', 'condition', 'fuel', 'title_status', 
                       'transmission', 'drive', 'type', 'paint_color']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.lower()
    
    # Initial cleaning
    df = df.dropna(subset=['price', 'manufacturer', 'model', 'year'])
    print(f"Records after dropping missing values: {len(df)}")
    
    # Remove unrealistic prices
    df = df[
        (df['price'] > 5000) &  # Remove very low prices
        (df['price'] < 30000)  # Remove extremely high prices
    ]
    print(f"Records after price filtering: {len(df)}")
    
    # Remove outliers using IQR method for price
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[
        (df['price'] >= Q1 - 1.5 * IQR) &
        (df['price'] <= Q3 + 1.5 * IQR)
    ]
    print(f"Records after removing price outliers: {len(df)}")
    
    # Year filtering
    current_year = 2024
    df = df[
        (df['year'] >= 1990) &  # Remove very old vehicles
        (df['year'] <= current_year)  # Remove future years
    ]
    print(f"Records after year filtering: {len(df)}")
    
    # Mileage filtering
    df = df[
        (df['odometer'] >= 20000) &  # Remove negative mileage
        (df['odometer'] <= 150000)  # Remove extreme mileage
    ]
    print(f"Records after mileage filtering: {len(df)}")
    
    # Condition filtering
    valid_conditions = ['excellent', 'good', 'fair', 'like new']
    df = df[df['condition'].isin(valid_conditions)]
    print(f"Records after condition filtering: {len(df)}")
    
    # Title status filtering
    valid_titles = ['clean', 'rebuilt', 'lien']
    df = df[df['title_status'].isin(valid_titles)]
    print(f"Records after title filtering: {len(df)}")
    
    # Calculate price per mile and remove outliers
    df['price_per_mile'] = df['price'] / (df['odometer'] + 1)
    Q1_ppm = df['price_per_mile'].quantile(0.25)
    Q3_ppm = df['price_per_mile'].quantile(0.75)
    IQR_ppm = Q3_ppm - Q1_ppm
    df = df[
        (df['price_per_mile'] >= Q1_ppm - 1.5 * IQR_ppm) &
        (df['price_per_mile'] <= Q3_ppm + 1.5 * IQR_ppm)
    ]
    df = df.drop('price_per_mile', axis=1)
    print(f"Records after price/mile filtering: {len(df)}")
    
    # Fill missing values
    df['odometer'] = df['odometer'].fillna(df.groupby(['manufacturer', 'model', 'year'])['odometer'].transform('median'))
    df['odometer'] = df['odometer'].fillna(df.groupby(['manufacturer', 'year'])['odometer'].transform('median'))
    df['odometer'] = df['odometer'].fillna(df['odometer'].median())
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # Print final statistics
    print("\nFinal Dataset Statistics:")
    print("\nRecords by manufacturer (top 10):")
    print(df['manufacturer'].value_counts().head(10))
    print("\nPrice statistics by manufacturer (top 10):")
    print(df.groupby('manufacturer')['price'].agg(['count', 'mean', 'std', 'min', 'max']).head(10))
    
    # Save detailed statistics
    stats_file = 'dataset_statistics.txt'
    with open(stats_file, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=================\n\n")
        f.write(f"Total records: {len(df)}\n\n")
        
        f.write("Price Statistics by Manufacturer:\n")
        f.write(df.groupby('manufacturer')['price'].describe().to_string())
        
        f.write("\n\nPrice Statistics by Year:\n")
        f.write(df.groupby('year')['price'].describe().to_string())
        
        f.write("\n\nPrice Statistics by Condition:\n")
        f.write(df.groupby('condition')['price'].describe().to_string())
        
        f.write("\n\nMileage Statistics by Year:\n")
        f.write(df.groupby('year')['odometer'].describe().to_string())
    
    print(f"\nDetailed statistics saved to {stats_file}")
    
    return df

def train_model(X_train, y_train, X_val, y_val):
    """Train an improved XGBoost model with optimized cross-validation"""
    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'max_depth': 8,           # Reduced from 12 to prevent overfitting
        'learning_rate': 0.05,    # Slower learning rate for better generalization
        'n_estimators': 500,      # More trees for better accuracy
        'min_child_weight': 5,    # Increased to prevent overfitting
        'subsample': 0.8,         # Random subsampling of training data
        'colsample_bytree': 0.8,  # Random subsampling of features
        'reg_alpha': 0.1,         # L1 regularization
        'reg_lambda': 1.0,        # L2 regularization
        'random_state': 42
    }
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Perform cross-validation with reduced folds
    print("Performing cross-validation...")
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        nfold=3,  # Reduced from 5 to 3 folds
        metrics=['mae'],
        early_stopping_rounds=20,  # Reduced from 50 to 20
        verbose_eval=10,  # Print progress every 10 rounds
        seed=42
    )
    
    # Get best number of trees from CV
    best_n_estimators = cv_results.shape[0]
    params['n_estimators'] = best_n_estimators
    
    # Train final model
    print("Training XGBoost model...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=best_n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        verbose_eval=10  # Print progress every 10 rounds
    )
    
    # Calculate validation metrics
    val_pred = model.predict(dval)
    val_mae = np.mean(np.abs(val_pred - y_val))
    val_mape = np.mean(np.abs((val_pred - y_val) / y_val)) * 100
    
    print(f"Cross-validation MAE: ${cv_results['test-mae-mean'].iloc[-1]:,.2f}")
    print(f"Validation MAE: ${val_mae:,.2f}")
    print(f"Validation MAPE: {val_mape:.1f}%")
    
    # Calculate and print R² score
    r2_score = 1 - np.sum((y_val - val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)
    print(f"Validation R² Score: {r2_score:.3f}")
    
    # Calculate feature importance using XGBoost's native method
    try:
        # Get feature importance scores
        importance = model.get_score(importance_type='gain')
        
        # Create DataFrame with feature importance
        feature_importance = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10).to_string())
        
        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        print("\nFeature importance saved to feature_importance.csv")
    except Exception as e:
        print(f"\nCould not calculate feature importance: {str(e)}")
    
    return model

def analyze_manufacturer_accuracy(model, X_val, y_val, df_val):
    """Analyze prediction accuracy by manufacturer"""
    # Get predictions using XGBoost's predict method
    dval = xgb.DMatrix(X_val)
    predictions = model.predict(dval)
    
    # Create a DataFrame with validation data
    val_df = pd.DataFrame({
        'manufacturer': df_val['manufacturer'].values,
        'price': y_val.values,
        'prediction': predictions
    })
    
    # Calculate metrics
    val_df['error'] = np.abs(val_df['prediction'] - val_df['price'])
    val_df['error_pct'] = (val_df['error'] / val_df['price']) * 100
    
    # Group by manufacturer
    accuracy_by_make = val_df.groupby('manufacturer').agg({
        'error': 'mean',
        'error_pct': 'mean',
        'price': 'count'
    }).sort_values('error_pct')
    
    # Rename columns
    accuracy_by_make.columns = ['MAE', 'MAPE', 'Sample_Size']
    
    # Format results
    accuracy_by_make['MAE'] = accuracy_by_make['MAE'].apply(lambda x: f"${x:,.2f}")
    accuracy_by_make['MAPE'] = accuracy_by_make['MAPE'].apply(lambda x: f"{x:.1f}%")
    
    print("\nPrediction Accuracy by Manufacturer (Top 10):")
    print(accuracy_by_make.head(10).to_string())
    
    return accuracy_by_make

def analyze_mileage_impact(model, X_val, y_val, df_val):
    """Analyze how mileage affects prediction accuracy with more granular brackets"""
    # Get predictions using XGBoost's predict method
    dval = xgb.DMatrix(X_val)
    predictions = model.predict(dval)
    
    # Create a DataFrame with validation data
    val_df = pd.DataFrame({
        'odometer': df_val['odometer'].values,
        'price': y_val.values,
        'prediction': predictions
    })
    
    # Calculate error metrics
    val_df['error'] = np.abs(val_df['prediction'] - val_df['price'])
    val_df['error_pct'] = (val_df['error'] / val_df['price']) * 100
    
    # Create more granular mileage bins
    bins = [0, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000, float('inf')]
    labels = ['0-10k', '10k-25k', '25k-50k', '50k-75k', '75k-100k', 
              '100k-125k', '125k-150k', '150k-175k', '175k-200k', '200k+']
    val_df['mileage_bin'] = pd.cut(val_df['odometer'], bins=bins, labels=labels)
    
    # Group by mileage bin
    accuracy_by_mileage = val_df.groupby('mileage_bin').agg({
        'error': 'mean',
        'error_pct': 'mean',
        'price': ['count', 'mean', 'std']
    }).sort_values(('error_pct', 'mean'))
    
    # Rename columns
    accuracy_by_mileage.columns = ['MAE', 'MAPE', 'Sample_Size', 'Avg_Price', 'Price_Std']
    
    # Format results
    accuracy_by_mileage['MAE'] = accuracy_by_mileage['MAE'].apply(lambda x: f"${x:,.2f}")
    accuracy_by_mileage['MAPE'] = accuracy_by_mileage['MAPE'].apply(lambda x: f"{x:.1f}%")
    accuracy_by_mileage['Avg_Price'] = accuracy_by_mileage['Avg_Price'].apply(lambda x: f"${x:,.2f}")
    accuracy_by_mileage['Price_Std'] = accuracy_by_mileage['Price_Std'].apply(lambda x: f"${x:,.2f}")
    
    print("\nPrediction Accuracy by Mileage Range (Detailed):")
    print(accuracy_by_mileage.to_string())
    
    return accuracy_by_mileage

def test_gui_integration():
    """Test GUI integration and prediction functionality"""
    try:
        import tkinter as tk
        from gui import VehicleAnalysisGUI
        
        # Create root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Create predictor instance
        predictor = VehiclePricePredictor()
        predictor.load()  # Load the saved model and artifacts
        
        # Initialize GUI
        gui = VehicleAnalysisGUI(root)
        
        # Test prediction with different vehicles
        test_cases = [
            {
                'year': 2015,
                'manufacturer': 'bmw',
                'model': '320i',
                'condition': 'good',
                'odometer': 80000,
                'fuel': 'gas',
                'title_status': 'clean',
                'transmission': 'automatic',
                'drive': 'rwd',
                'type': 'sedan',
                'paint_color': 'white'
            },
            {
                'year': 2018,
                'manufacturer': 'toyota',
                'model': 'camry',
                'condition': 'excellent',
                'odometer': 40000,
                'fuel': 'gas',
                'title_status': 'clean',
                'transmission': 'automatic',
                'drive': 'fwd',
                'type': 'sedan',
                'paint_color': 'silver'
            }
        ]
        
        print("\nTesting GUI predictions:")
        for vehicle in test_cases:
            # Use the predictor directly
            result = predictor.predict(vehicle)
            print(f"\nVehicle: {vehicle['year']} {vehicle['manufacturer'].upper()} {vehicle['model']}")
            print(f"Predicted price: ${result['prediction']:,.2f}")
            print(f"Price range: ${result['confidence_low']:,.2f} - ${result['confidence_high']:,.2f}")
        
        # Clean up
        root.destroy()
        
        print("\nGUI test completed successfully")
        return True
    except Exception as e:
        print(f"GUI test failed: {str(e)}")
        return False

def main():
    """Main function to train and save the model."""
    print("Loading and preparing data...")
    df = load_and_explore_data()
    
    # Use a smaller subset for testing
    df = df.sample(frac=0.3, random_state=42)  # Use 30% of the data
    print(f"\nUsing {len(df)} records for testing")
    
    print("Creating predictor...")
    predictor = VehiclePricePredictor()
    
    print("Cleaning data...")
    df = predictor.clean_data(df, is_training=True)
    
    print("Engineering features...")
    df = predictor.engineer_features(df, is_training=True)
    
    print("Preprocessing data...")
    X, y = predictor.preprocess_data(df, is_training=True)
    
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    predictor.scaler = StandardScaler()
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_val_scaled = predictor.scaler.transform(X_val)
    
    print("Training model...")
    predictor.model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    
    print("Saving model and artifacts...")
    predictor.save()
    
    # Get validation data - use the original DataFrame's index
    val_indices = y_val.index
    val_df = df.loc[val_indices].copy()
    
    # Analyze manufacturer accuracy
    accuracy_by_make = analyze_manufacturer_accuracy(predictor.model, X_val_scaled, y_val, val_df)
    
    # Analyze mileage impact
    accuracy_by_mileage = analyze_mileage_impact(predictor.model, X_val_scaled, y_val, val_df)
    
    # Test prediction with different mileages
    test_vehicle = {
        'year': 2015,
        'manufacturer': 'bmw',
        'model': '320i',
        'condition': 'good',
        'fuel': 'gas',
        'title_status': 'clean',
        'transmission': 'automatic',
        'drive': 'rwd',
        'type': 'sedan',
        'paint_color': 'white'
    }
    
    print("\nTesting with different mileages for 2015 BMW 320i:")
    for mileage in [40000, 80000, 120000, 150000]:
        test_vehicle['odometer'] = mileage
        result = predictor.predict(test_vehicle)
        print(f"\nMileage: {mileage:,} miles")
        print(f"Predicted price: ${result['prediction']:,.2f}")
        print(f"Price range: ${result['confidence_low']:,.2f} - ${result['confidence_high']:,.2f}")
    
    # Test GUI integration
    print("\nTesting GUI integration...")
    gui_test_result = test_gui_integration()
    
    if gui_test_result:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 
