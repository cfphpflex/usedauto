# Advanced Vehicle Price Prediction Model

## Overview
This project implements an advanced machine learning model for predicting used vehicle prices using LightGBM (Light Gradient Boosting Machine). The model incorporates sophisticated feature engineering, proper data preprocessing, and robust prediction pipelines.

## Model Architecture

### 1. Data Preprocessing Pipeline
The model uses a comprehensive preprocessing pipeline implemented in the `VehiclePricePredictor` class:

```python
class VehiclePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_info = None
        self.feature_names = None
        self.category_mappings = {}
```

#### Key Preprocessing Steps:
- **Data Cleaning** (`clean_data()`):
  - Handles missing values
  - Removes outliers (prices < $100 or > $100,000)
  - Fills missing odometer readings with mean values
  - Fills missing years with median values
  - Handles missing categorical values with 'unknown'

- **Feature Engineering** (`engineer_features()`):
  - Vehicle age calculation (2024 - year)
  - Miles per year calculation
  - Price per mile (training only)
  - Price per year (training only)
  - Age-mileage interaction features

- **Data Preprocessing** (`preprocess_data()`):
  - Categorical variable standardization
  - One-hot encoding
  - Feature scaling
  - Feature name preservation

### 2. Model Training
The model uses LightGBM with optimized parameters:

```python
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
```

#### Training Process:
1. Data loading and cleaning
2. Feature engineering
3. Train/validation split
4. Model training with early stopping
5. Model evaluation using MAE and R² metrics
6. Artifact saving (model, scaler, feature info)

### 3. Prediction Pipeline
The prediction process follows these steps:
1. Input data validation
2. Feature preprocessing
3. Feature engineering
4. Model prediction
5. Price output

## Key Features

### 1. Robust Feature Engineering
- **Temporal Features**:
  - Vehicle age
  - Miles per year
  - Price per year (training)
  - Price per mile (training)

- **Interaction Features**:
  - Age-mileage interaction
  - Manufacturer-model combinations

### 2. Categorical Variable Handling
- Standardized lowercase conversion
- One-hot encoding
- Category mapping preservation
- Unknown category handling

### 3. Model Persistence
The model saves three key artifacts:
1. `model.joblib`: The trained LightGBM model
2. `scaler.pkl`: The StandardScaler for feature normalization
3. `feature_info.json`: Feature names and category mappings

## Usage

### Training the Model
```bash
python src/improved_model.py
```

### Making Predictions
```bash
python src/predict.py
```

### Using the GUI
```bash
python src/gui.py
```

## Performance Metrics
The model is evaluated using:
- Mean Absolute Error (MAE)
- R² Score
- Validation set performance
- Cross-validation results

## Dependencies
- pandas
- numpy
- lightgbm
- scikit-learn
- joblib
- pickle

## Future Improvements
1. Hyperparameter optimization
2. Additional feature engineering
3. Ensemble methods
4. Real-time price updates
5. Market trend analysis 