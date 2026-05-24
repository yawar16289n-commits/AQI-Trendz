# Pearls AQI Predictor - Project Overview

This document summarizes the requirements and architecture for the "Pearls AQI Predictor" project, based on the provided work file.

## Objective
To build a 100% serverless machine learning system capable of predicting the Air Quality Index (AQI) for a given city over the next 3 days.

## System Architecture & Pipelines

### 1. Feature Pipeline
- **Data Ingestion**: Fetch raw weather and pollutant data from external APIs (e.g., AQICN, OpenWeather).
- **Feature Engineering**: Compute model inputs and outputs, including time-based features (hour, day, month) and derived features (e.g., AQI change rate).
- **Storage**: Store these features in a Feature Store (such as Hopsworks or Vertex AI).
- **Backfilling**: Run the script for past dates to generate historical training data.

### 2. Training Pipeline
- **Data Retrieval**: Fetch historical features and targets from the Feature Store.
- **Model Training & Evaluation**: Train multiple machine learning models ranging from traditional statistical models (Scikit-learn: Random Forest, Ridge Regression) to advanced deep learning models (TensorFlow/PyTorch). Evaluate using metrics like RMSE, MAE, and R².
- **Model Registry**: Store the best performing model in a Model Registry.

### 3. Automation (CI/CD)
- Automate pipeline runs using tools like Apache Airflow or GitHub Actions.
- Schedule the feature script to run hourly and the training script to run daily.

### 4. Web Application (User Interface)
- Build a web app using Streamlit/Gradio and Flask/FastAPI.
- The app should load the model and features, compute predictions, and display them on an interactive, user-friendly dashboard.
- **Alerts**: Include notifications for hazardous AQI levels.

## Key Guidelines
- Conduct Exploratory Data Analysis (EDA) to discover trends.
- Implement Model Explainability using SHAP or LIME to show feature importance.

## Final Deliverables
1. An end-to-end AQI prediction system.
2. A scalable and automated ML pipeline.
3. An interactive real-time and forecasted AQI dashboard.
4. A comprehensive report documenting the entire project.
