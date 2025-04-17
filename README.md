# Advanced Vehicle Price Prediction Model

## Project Overview
Analysis of large used cars dataset from Kaggle to determine key factors influencing car prices and provide data-driven recommendations for used car dealerships. The project follows the CRISP-DM methodology and uses RandomForest with GridSearch optimization.

### Business Objectives
- Help dealerships identify high-quality used cars for quick, profitable sales
- Provide clear insights on consumer value preferences in used cars
- Develop accurate price prediction model for market analysis

### Quality Criteria
- Low mileage
- No accidents
- Recent model year
- High resale price
- Clean title
- Fast time-to-sale (when available)

## CRISP-DM Implementation

### 1. Business Understanding
- Focus on used car market analysis
- Target audience: dealerships and resellers
- Key metrics: price prediction accuracy and feature importance

### 2. Data Understanding
- Dataset: 426,000 used car records (sampled from 3M)
- Key features include:
  - Vehicle specifications
  - Mileage and age
  - Condition and history
  - Market pricing

### 3. Data Preparation
- Data cleaning and preprocessing
- Feature engineering
- Handling missing values
- Outlier removal
- Categorical variable encoding

### 4. Modeling
- Model: RandomForest with GridSearch optimization
- Training pipeline:
  1. Data loading and cleaning
  2. Feature engineering
  3. Train/validation split
  4. Model training
  5. Performance evaluation
  6. Model persistence

### 5. Evaluation
Performance metrics:
- Mean Absolute Error (MAE)
- R² Score
- Validation set performance
- Cross-validation results

### 6. Deployment
Available interfaces:
- Command-line prediction tool
- GUI interface
- Model API integration

## Technical Implementation

### Feature Engineering
- Temporal features (vehicle age, miles/year)
- Price-related features (price/mile, price/year)
- Interaction features (age-mileage, manufacturer-model)
- Categorical variable standardization

### Model Architecture
```python
class VehiclePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_info = None
        self.feature_names = None
        self.category_mappings = {}
```

## Usage Instructions

### Installation
```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
```bash
# Train the model
python src/improved_model.py

# Make predictions
python src/predict.py

# Launch GUI interface
python src/gui.py
```

## Dependencies
- pandas
- numpy
- scikit-learn
- lightgbm
- joblib
- pickle

## Future Improvements
1. Hyperparameter optimization
2. Additional feature engineering
3. Ensemble methods
4. Real-time price updates
5. Market trend analysis


Requirements:
	•	Apply the CRISP-DM methodology (Cross-Industry Standard Process for Data Mining) to structure the analysis.
		CRISP-DM Framework
			1. load data
		   	2. prepare data
		   	3. Engineer Features
		   	4. Split Data
		   	5. Scale Features (both mmodel and predict features must match)
		   	6. Train model
            7. Predict used car price


	•	Perform data cleaning, exploration, modeling, and interpretation.
	•	Identify key features that drive car pricing.
	•	Communicate findings with actionable business insights.
##

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
"""

### Code Quality Requirements

1. **Internal Naming Convention Consistency**
   - All feature-related variables and JSON keys must use consistent naming:
     - Use `feature_names` for list of feature names
     - Use `feature_info` for the overall feature information structure
     - Use `numeric_features` for numeric feature names
     - Use `categorical_features` for categorical feature names
     - Use `category_mapping` for category mappings
   - Avoid mixing terms like `feature_columns` and `feature_names`
   - Documented all feature-related variables in docstrings
   - Ensured JSON files use consistent key names across the codebase


1🧹 Data Understanding & Preprocessing

Dataset: Dowanload from Kaggle and save it at ./data/vehicles.csv
Initial Features: make, model, year, price, mileage, fuel type, transmission, location, accident history, etc.

2. EDA & Cleaning:
	•	Visualizations:
	•	Price vs. mileage
	•	Distribution plots (log-transform applied if skewed)
	•	Correlation heatmaps
	•	Missing values:
	•	Median/mode imputation or KNN/LightGBM imputer
	•	Feature Engineering:
	•	car_age = current_year - year
	•	price_per_mile = price / mileage
	•	is_luxury_brand = brand in [BMW, Audi, Mercedes, Tesla]
	•	is_davis = location == "Davis, CA"

⸻

3. 🏗️ Modeling Strategy

Stacked Ensemble Architecture:
	•	Base Models:
	•	RandomForest, XGBoost, CatBoost
	•	Meta Model:
	•	Logistic Regression (classification) or Linear Regression (quality score prediction)

⸻

4. 🧪 Training Methodology

Target Options:
	•	Classification: high_quality = 1 if car meets quality threshold
	•	Regression: Predict a quality score (0–100)

Validation Strategy:
	•	StratifiedKFold for classification
	•	GroupKFold (to prevent leakage by car ID)
	•	Hyperparameter tuning via Optuna or Bayesian Optimization
	•	Address class imbalance with SMOTE or class_weight

⸻

5. 📊 Evaluation Metrics

Classification:
	•	F1-Score (prioritize precision)
	•	AUC-ROC
Regression:
	•	RMSE, MAE, R²
Domain-Specific:
	•	Time-to-sale or profit margin ranking, if available

⸻

6. ⚙️ Deployment Considerations
	•	API: Model served via FastAPI
	•	Automation: Pipelines using scikit-learn or MLFlow
	•	Retraining: Monthly updates
	•	Monitoring: Prometheus, Grafana, or Evidently for drift detection

⸻

7. 📍 Local Customization – Davis, CA
	•	Electric/Hybrid flag (is_ev) – local eco-conscious trends
	•	Local Popularity Score – via search data or sales
	•	Price Adjustment – normalized to Davis ZIP code median income


8. REQUIRED: Scripts to run, get vehicles.csv from Kaggle and put it into data folder
python src/improved_model.py
python src/predict.py
python src/gui.py

model, scaler, feature_info = load_model()

 