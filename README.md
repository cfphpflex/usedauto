# Advanced Used Vehicle Price Prediction Model

## Executive Summary

A data science project analyzing used car listings from Kaggle to predict vehicle prices and identify
high-quality cars. The work follows the CRISP-DM methodology and features more advanced models
(RandomForest, XGBoost, CatBoost), visual analysis, and deployment scripts. 
Key deliverables include data visualizations, a robust model training pipeline, and insights for dealership decision-making.


## Summary of Findings

- Low-mileage, newer cars = higher value and faster sales
- Luxury brands show higher resale and depreciation
- Local trends (e.g., eco-focus in Davis) influence pricing

## Next Steps

- Scheduled model retraining
- Integrate real-time pricing data
- Interactive visual dashboard


## Business Objectives

- Help dealerships identify high-quality used cars for resale
- Predict used car prices using ML techniques
- Provide actionable recommendations for used car investments
- Tailor analysis to specific locales (e.g., Davis, CA) after interviewing two dealers


# Model Steps
- Load and prepare the data
- Engineer features
- Train the model
- Save the model and artifacts
- Make a test prediction


## Quality Criteria for Best-Value Cars

- Low mileage
- No accidents
- Recent model year
- High resale price
- Clean title
- Fast time-to-sale (if available)
  
## Project Organization

- README.md: Summary and instructions
- notebooks/: Jupyter notebooks with full analysis
- src/: Model training, prediction, and GUI scripts
- data/vehicles.csv: Kaggle-sourced dataset
- models/: Saved artifacts (model.joblib, scaler.pkl, feature_info.json)

### 1. Data Understanding & Preprocessing

- Dataset: Kaggle - ./data/vehicles.csv
- Initial features: make, model, year, price, mileage, transmission, location, accident history
Advanced Vehicle Price Prediction Model
- Missing values handled via median/mode or ML imputer
- Outliers removed (price > $5000 or > $50,000)
- Feature engineering: car_age, price_per_mile, is_luxury_brand, is_davis
  
### 2. Exploratory Data Analysis

- Visualizations: Price vs. mileage scatter plot, Log-transformed price distribution, Correlation heatmaps,
Categorical breakdowns
- Tools: pandas, seaborn, matplotlib

### 3. Modeling Strategy
- Meta model: Linear Regression
- Base models: RandomForest, XGBoost, CatBoost (new for capstone)
- Classification: high_quality = 1
- Regression: quality_score in [0100]
 
 
  
## Lib. Dependencies

--- pandas
--- numpy
--- scikit-learn
--- seaborn
--- matplotlib
--- xgboost
--- catboost
--- joblib
--- optuna

## Run the model Instructions

## Step 1: Download dataset to ./data/vehicles.csv from Kaggle
## Step 2: Train the model
-- python src/improved_model.py
## Step 3: Predict a price
-- python src/predict.py
## Step 4: Launch GUI
-- python src/gui.py

git rm --cached src/models/model.joblib 2>/dev/null || true