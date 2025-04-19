#Advanced Vehicle Price Prediction Model

##Project Overview

A data science project analyzing used car listings from Kaggle to predict vehicle prices and identify
high-quality cars. The work follows the CRISP-DM methodology and features a stacked ensemble model
(RandomForest, XGBoost, CatBoost), visual analysis, and deployment scripts. Key deliverables include data
visualizations, a robust model training pipeline, and insights for dealership decision-making.

##Business Objectives

- Help dealerships identify high-quality used cars for resale
- Predict used car prices using ML techniques
- Provide actionable recommendations for used car investments
- Tailor analysis to specific locales (e.g., Davis, CA)


##Quality Criteria for High-Value Cars

- Low mileage
- No accidents
- Recent model year
- High resale price
- Clean title
- Fast time-to-sale (if available)
  
##Project Organization

- README.md: Summary and instructions
- notebooks/: Jupyter notebooks with full analysis
- src/: Model training, prediction, and GUI scripts
- data/vehicles.csv: Kaggle-sourced dataset
- models/: Saved artifacts (model.joblib, scaler.pkl, feature_info.json)

###1. Data Understanding & Preprocessing

- Dataset: Kaggle - ./data/vehicles.csv
- Initial features: make, model, year, price, mileage, transmission, location, accident history
Advanced Vehicle Price Prediction Model
- Missing values handled via median/mode or ML imputer
- Outliers removed (price < $100 or > $100,000)
- Feature engineering: car_age, price_per_mile, is_luxury_brand, is_davis
  
###2. Exploratory Data Analysis

- Visualizations: Price vs. mileage scatter plot, Log-transformed price distribution, Correlation heatmaps,
Categorical breakdowns
- Tools: pandas, seaborn, matplotlib

###3. Modeling Strategy

- Base models: RandomForest, XGBoost, CatBoost
- Meta model: Linear Regression
- Classification: high_quality = 1
- Regression: quality_score in [0100]

###4. Training Methodology

- StratifiedKFold or GroupKFold
- Hyperparameter tuning via Optuna
- Address imbalance using SMOTE or class_weight
- Metrics: F1, AUC-ROC, RMSE, MAE, R2

###5. Deployment

- CLI: python src/improved_model.py
- Prediction: python src/predict.py
- GUI: python src/gui.py
- Artifacts: model.joblib, scaler.pkl, feature_info.json
- Monitoring (planned): Prometheus/Grafana/Evidently
###6. Local Customization Davis, CA

- Electric/hybrid flag


##Advanced Vehicle Price Prediction Model

- Popularity score (future)
- ZIP code median income normalization
  
##Summary of Findings

- Low-mileage, newer cars = higher value and faster sales
- Luxury brands show higher resale and depreciation
- Local trends (e.g., eco-focus in Davis) influence pricing

##Next Steps

- Scheduled model retraining
- Integrate real-time pricing data
- Interactive visual dashboard

##Dependencies

pandas
numpy
scikit-learn
seaborn
matplotlib
xgboost
catboost
joblib
optuna

##Usage Instructions

# Step 1: Download dataset to ./data/vehicles.csv from Kaggle
# Step 2: Train the model
python src/improved_model.py
# Step 3: Predict a price
python src/predict.py
# Step 4: Launch GUI
