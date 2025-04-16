import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from scipy import stats
import os
import joblib
import json

def load_data(nrows=None):
    """Load data from CSV file"""
    print(f"Loading data with {nrows if nrows else 'all'} rows...")
    df = pd.read_csv('data/vehicles.csv')
    print(f"Loaded {len(df)} rows")
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    print("Cleaning data...")
    print(f"Initial shape: {df.shape}")
    
    # Convert posting_date to datetime, handling mixed timezones
    df['posting_date'] = pd.to_datetime(df['posting_date'], utc=True)
    df['posting_year'] = df['posting_date'].dt.year
    df['posting_month'] = df['posting_date'].dt.month
    df['posting_day'] = df['posting_date'].dt.day
    df['posting_quarter'] = df['posting_date'].dt.quarter
    
    # Calculate seasonal features
    df['season'] = df['posting_month'] % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    df['is_weekend'] = df['posting_date'].dt.dayofweek >= 5
    
    # Remove rows with missing or invalid prices
    df = df[df['price'].notna() & (df['price'] > 0)]
    print(f"After price cleaning: {df.shape}")
    
    # Sophisticated outlier detection using IQR and Z-score
    def remove_outliers(df, column, z_threshold=4, iqr_multiplier=2.5):
        print(f"\nProcessing {column}:")
        print(f"Initial count: {len(df)}")
        
        # Calculate statistics
        z_scores = np.abs(stats.zscore(df[column]))
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Apply filters
        z_filter = z_scores < z_threshold
        iqr_filter = (df[column] >= lower_bound) & (df[column] <= upper_bound)
        
        # Count outliers
        z_outliers = (~z_filter).sum()
        iqr_outliers = (~iqr_filter).sum()
        print(f"Z-score outliers: {z_outliers}")
        print(f"IQR outliers: {iqr_outliers}")
        
        # Apply both filters
        filtered_df = df[z_filter & iqr_filter]
        print(f"Final count: {len(filtered_df)}")
        print(f"Removed: {len(df) - len(filtered_df)} rows")
        
        return filtered_df
    
    # Remove outliers for key numerical features
    df = remove_outliers(df, 'price', z_threshold=4, iqr_multiplier=2.5)
    
    # Clean odometer readings with less aggressive outlier detection
    df = df[df['odometer'].notna() & (df['odometer'] >= 0)]
    df = remove_outliers(df, 'odometer', z_threshold=5, iqr_multiplier=3.0)
    print(f"After odometer cleaning: {df.shape}")
    
    if len(df) == 0:
        print("Warning: No data remaining after cleaning. Adjusting outlier detection parameters...")
        # Reset to original data after price cleaning
        df = df[df['price'].notna() & (df['price'] > 0)]
        # Use more lenient parameters for odometer
        df = df[df['odometer'].notna() & (df['odometer'] >= 0)]
        print(f"Using lenient cleaning, shape: {df.shape}")
    
    # Clean year
    current_year = datetime.now().year
    df = df[
        df['year'].notna() & 
        (df['year'] >= 1900) & 
        (df['year'] <= current_year)
    ]
    print(f"After year cleaning: {df.shape}")
    
    # Calculate regional price indices
    regional_stats = df.groupby('state').agg({
        'price': ['mean', 'std', 'count']
    }).reset_index()
    regional_stats.columns = ['state', 'regional_mean_price', 'regional_std_price', 'regional_count']
    df = df.merge(regional_stats, on='state', how='left')
    df['price_deviation'] = (df['price'] - df['regional_mean_price']) / df['regional_std_price']
    
    # Fill missing categorical values
    categorical_cols = ['manufacturer', 'model', 'condition', 'fuel', 
                       'title_status', 'transmission', 'drive', 'size', 
                       'type', 'paint_color']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    # Calculate additional features
    df['age'] = current_year - df['year']
    df['price_per_mile'] = df['price'] / (df['odometer'] + 1)
    df['mileage_per_year'] = df['odometer'] / (df['age'] + 1)
    
    print(f"\nFinal cleaned data shape: {df.shape}")
    return df

def create_visualizations(df):
    """Create comprehensive visualizations of the data"""
    print("Creating visualizations...")
    
    # Create visualizations directory if it doesn't exist
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 1. Price Distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df['price'], bins=50)
    plt.title('Price Distribution')
    plt.xlabel('Price ($)')
    
    plt.subplot(1, 2, 2)
    sns.histplot(np.log1p(df['price']), bins=50)
    plt.title('Log Price Distribution')
    plt.xlabel('Log(Price)')
    plt.tight_layout()
    plt.savefig('visualizations/price_distribution.png')
    plt.close()
    
    # 2. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    
    # 3. Feature vs Price Scatter Plots
    features = ['odometer', 'year', 'age']
    plt.figure(figsize=(15, 5))
    for i, feature in enumerate(features, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(x=feature, y='price', data=df, alpha=0.3)
        plt.title(f'{feature} vs Price')
    plt.tight_layout()
    plt.savefig('visualizations/feature_scatter_plots.png')
    plt.close()
    
    # 4. Categorical Feature Analysis
    categorical_features = ['manufacturer', 'condition', 'fuel', 'transmission']
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(categorical_features, 1):
        plt.subplot(2, 2, i)
        sns.boxplot(x=feature, y='price', data=df)
        plt.xticks(rotation=45)
        plt.title(f'Price by {feature}')
    plt.tight_layout()
    plt.savefig('visualizations/categorical_analysis.png')
    plt.close()
    
    # 5. Time Series Analysis
    plt.figure(figsize=(12, 6))
    df.groupby('posting_date')['price'].mean().plot()
    plt.title('Average Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Average Price ($)')
    plt.tight_layout()
    plt.savefig('visualizations/price_time_series.png')
    plt.close()

def scale_features(X_train, X_val):
    """Scale features using RobustScaler"""
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled, scaler

def save_model(model, scaler, features, model_dir='models'):
    """Save the trained model, scaler, and feature information"""
    print("Saving model and artifacts...")
    
    # Create models directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save the model
    model.save_model(os.path.join(model_dir, 'model.txt'))
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    # Save feature information
    feature_info = {
        'features': features,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(model_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f)
    
    print(f"Model and artifacts saved to {model_dir}/")

def load_model(model_dir='models'):
    """Load the saved model, scaler, and feature information"""
    print("Loading model and artifacts...")
    
    # Load the model
    model = lgb.Booster(model_file=os.path.join(model_dir, 'model.txt'))
    
    # Load the scaler
    scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    
    # Load feature information
    with open(os.path.join(model_dir, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    return model, scaler, feature_info['features']

def train_model_with_cv(df, n_splits=5):
    """Train LightGBM model with cross-validation"""
    print("Preparing data for modeling...")
    
    # Prepare features and target
    features = ['year', 'odometer', 'age', 'price_per_mile', 'mileage_per_year',
                'posting_year', 'posting_month', 'posting_day', 'posting_quarter',
                'season', 'is_weekend', 'regional_mean_price', 'regional_std_price',
                'price_deviation', 'manufacturer', 'model', 'condition', 'fuel', 
                'title_status', 'transmission', 'drive', 'type', 'paint_color']
    
    # Keep only features that exist in the dataframe
    features = [f for f in features if f in df.columns]
    
    X = df[features].copy()
    y = df['price']
    
    # Encode categorical variables
    categorical_features = X.select_dtypes(include=['object']).columns
    for col in categorical_features:
        X[col] = pd.Categorical(X[col]).codes
    
    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    models = []
    scalers = []
    
    print(f"Starting {n_splits}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features
        X_train_scaled, X_val_scaled, scaler = scale_features(X_train, X_val)
        scalers.append(scaler)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        
        # Set parameters
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Make predictions
        y_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        print(f"\nFold {fold} Performance:")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: ${mae:,.2f}")
        
        cv_scores.append({
            'fold': fold,
            'r2': r2,
            'mse': mse,
            'mae': mae
        })
        models.append(model)
    
    # Calculate average performance
    avg_r2 = np.mean([score['r2'] for score in cv_scores])
    avg_mae = np.mean([score['mae'] for score in cv_scores])
    avg_mse = np.mean([score['mse'] for score in cv_scores])
    
    print(f"\nCross-Validation Results:")
    print(f"Average R² Score: {avg_r2:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: ${avg_mae:,.2f}")
    
    # Get the best model and scaler
    best_model_idx = np.argmin([score['mae'] for score in cv_scores])
    best_model = models[best_model_idx]
    best_scaler = scalers[best_model_idx]
    
    # Plot feature importance from the best model
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': best_model.feature_importance()
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()
    
    return best_model, best_scaler, features, cv_scores

def main():
    # Load data
    df = load_data()
    
    # Clean data
    df = clean_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Train model with cross-validation
    model, scaler, features, cv_scores = train_model_with_cv(df)
    
    # Save the model and artifacts
    save_model(model, scaler, features)
    
    print("\nAnalysis complete! Model and artifacts have been saved.")
    print("Check the visualizations directory for plots.")

if __name__ == "__main__":
    main() 