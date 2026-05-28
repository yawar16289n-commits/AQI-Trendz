# AQI Predictor - Context Driven Development Implementation Plan

> **Last Updated**: 2026-05-23

## Current Progress Summary
- Chunk 1a: ✅ AQI & Weather historical data downloaded.
- Chunk 1b: ✅ Data Cleaning complete (Merged into `final_historical_data.csv`, 33,331 records).
- Chunk 1c: ✅ Exploratory Data Analysis complete (Plots saved in `eda_plots` folder).
- Chunk 1d: ✅ Basic Feature Engineering complete (Baseline time-features generated).
- Chunk 2b: ✅ feature_pipeline.py script written and executed.
- Chunk 2c: ✅ 33,331 rows of baseline features uploaded to Hopsworks Feature Store.
- Chunk 3a-d: ✅ Model Training Pipeline created. Trained LR, RF, XGBoost. Added lag features to achieve 0.93 R2 score. Best model (XGBoost) saved, with training history CSV enabled.

---

This plan incorporates your detailed steps, including data cleaning, advanced analytics (SHAP/LIME), and a mini-chunk breakdown.

## Selected ML Models
1. **Random Forest Regressor**: A robust tree-based ensemble method. It acts as a strong baseline that prevents overfitting.
2. **XGBoost Regressor / Ridge Regression**: Highly efficient gradient boosting model (or Ridge for a simpler linear baseline).
3. **LSTM / TensorFlow**: A deep learning model designed to capture complex temporal sequences and historical context over time.

---

## Context Driven Development (CDD) Mini-Chunks

### Chunk 1: Data Preparation & Exploratory Data Analysis (EDA)
- **Mini-chunk 1a**: Script to fetch historical raw weather and pollutant data from Open-Meteo.
- **Mini-chunk 1b**: Data Cleaning (handle missing values, normalize timestamps, remove outliers).
- **Mini-chunk 1c**: Perform EDA (Exploratory Data Analysis) in a Jupyter Notebook to identify correlations and seasonal trends.
- **Mini-chunk 1d**: Feature engineering (extract hour, day, month, compute AQI change rate, lag features).

### Chunk 2: Feature Pipeline & Hopsworks Storage
- **Mini-chunk 2a**: Setup Hopsworks account and API keys.
- **Mini-chunk 2b**: Create the `feature_pipeline.py` script.
- **Mini-chunk 2c**: Define Feature Groups and insert the processed data into the Hopsworks Feature Store.
- **Mini-chunk 2d**: Run the Historical Data Backfill script to populate the feature store.

## Phase 4: Automated CI/CD (GitHub Actions)

To make our AQI predictions operational and self-updating, we will set up automated pipelines that run periodically to fetch new data, retrain the model, and generate forecasts.

### Proposed Architecture for Chunk 4

1. **GitHub Repository Setup**: Initialize a Git repository connected to `https://github.com/yawar16289n-commits/AQI-Trendz.git` and commit our highly organized codebase.
2. **GitHub Secrets**: Securely store our Hopsworks API Key, Weather API credentials (if any), and other environment variables as GitHub Secrets.
3. **Hourly End-to-End Pipeline Workflow**: 
   - Create a `.github/workflows/hourly_aqi_pipeline.yml` action.
   - This workflow will trigger **every hour** via cron (`0 * * * *`).
   - It will run a unified script to:
     - Fetch the latest data (features and actual AQI).
     - Upload new features to Hopsworks.
     - **Retrain the model** immediately on the fresh dataset.
     - Predict the AQI for the **next 3 days** based on weather forecasts.
     - Upload the predictions to a Hopsworks Feature Group or Dataset for dashboard consumption.

*(Note: When we return to Chunk 3e to enhance the model, we will integrate residual analysis, feature interactions, and rolling averages as requested.)*

## Phase 3: Multi-Pollutant Model Training

To calculate the true US EPA AQI, we need to predict the concentrations of all critical pollutants, not just PM2.5. We will upgrade our architecture to a **Multi-Output Forecasting** model.

### Proposed Architecture for Multi-Pollutant Forecasting

1. **Feature Engineering Upgrade**: 
   - We will generate 1-hour and 24-hour lag features for *all 5* target pollutants: `pm2_5`, `pm10`, `nitrogen_dioxide`, `sulphur_dioxide`, and `carbon_monoxide`.
   - This expands our feature set to capture the autoregressive nature of every single gas.

2. **Multi-Output Model Training (`training_pipeline.py`)**: 
   - Instead of predicting a single target, our models will output a 5-dimensional vector.
   - We will wrap our winning XGBoost model (and the baselines) inside scikit-learn's `MultiOutputRegressor`. This effectively trains a dedicated, optimized XGBoost model for each of the 5 pollutants simultaneously.
   - We will evaluate the MAE and R² for each pollutant individually to ensure high accuracy across the board.

3. **Simultaneous Autoregressive Forecasting (`predict_pipeline.py`)**:
   - The 3-day forecast script will be upgraded to predict all 5 pollutants for *Hour 1*.
   - It will take those 5 predictions and feed them back into the feature vector as the `lag_1h` features for *Hour 2*.
   - This chain reaction will perfectly simulate the future state of all 5 AQI components for the next 72 hours.

> [!IMPORTANT]
> **User Review Required**:
> - Does wrapping XGBoost in a `MultiOutputRegressor` to simultaneously forecast all 5 pollutants sound like the right approach to you?
> - Once we predict all 5 raw concentrations, we will need to run them through the standard piecewise AQI formula to get the final AQI score (0-500 scale). We can implement this formula directly in the `predict_pipeline.py` script so the final CSV has the true AQI. Should I add the AQI calculation step to this chunk?

### Chunk 4: Automated CI/CD Pipeline (GitHub Actions)
- **Mini-chunk 4a**: Set up GitHub repository and store Hopsworks API keys as Secrets.
- **Mini-chunk 4b**: Create `.github/workflows/feature_pipeline.yml` (runs automatically every hour).
- **Mini-chunk 4c**: Create `.github/workflows/training_pipeline.yml` (runs automatically daily for model updates).

### Chunk 5: Web Application Dashboard & Explainability
- **Mini-chunk 5a**: Create the Web App. We will use **Streamlit** for the interactive dashboard and optionally **Flask/FastAPI** as the backend API if we want to decouple the frontend.
- **Mini-chunk 5b**: Load the best model and features from the Feature Store, and compute real-time predictions for the next 3 days.
- **Mini-chunk 5c**: Implement advanced analytics using SHAP or LIME for feature importance explanations.
- **Mini-chunk 5d**: Add UI alerts for hazardous AQI levels.

---

## User Review Required

> [!IMPORTANT]
> Please review the updated mini-chunks. I have saved this file directly in your project folder at `projects/implementation_plan.md` as requested, and I will continue to update it there as we progress. 

## Open Questions

1. For the Web Application, **Streamlit** is highly recommended because it's purely Python and perfect for ML dashboards. Using Flask as well would mean building a separate backend API and then a frontend to consume it. Do you want to keep it simple with just **Streamlit**, or do you strictly want a **Flask/FastAPI** backend powering a frontend?
2. Shall we begin with **Chunk 1: Data Preparation & Exploratory Data Analysis (EDA)** right now?
