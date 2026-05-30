# Serverless Air Quality Forecast System

## 📌 Project Overview
This project is an end-to-end, fully automated **Air Quality Index (AQI) and Pollutant Forecasting System**. It fetches live weather and AQI data, engineers time-series features, and uses a trained **XGBoost** model to predict AQI and 5 major pollutants (PM2.5, PM10, NO2, SO2, CO) for the next 72 hours. 

The entire pipeline is designed to be **Serverless & Cloud-Native**, meaning no local computing power is required once deployed. It utilizes **Hopsworks** for its Feature Store and Model Registry, and **GitHub Actions** for scheduling and automation.

---

## 🏗 Architecture & Technologies

- **Data Sources**: [Open-Meteo API](https://open-meteo.com/) for historical and forecasted Weather & Air Quality data.
- **Machine Learning**: `xgboost` with `MultiOutputRegressor` to predict multiple target variables simultaneously.
- **Feature Store & Model Registry**: [Hopsworks](https://www.hopsworks.ai/) (Version 2 schemas applied).
- **Automation (CI/CD)**: **GitHub Actions** handles hourly data fetching/inference and daily model retraining.
- **User Interface**: Interactive, dynamic frontend built with **Streamlit**. Interpretability provided via **SHAP** values.

---

## 📂 Project Structure & How It Works

The project is broken down into modular scripts to allow for distributed pipeline execution:

### 1. Data Pipeline (`scripts/data_prep/`)
- **`fetch_latest.py`**: Pulls the last 7 days of historical weather/AQI data (for feature lags) and the next 3 days of forecasted weather.
- **`feature_engineering_latest.py`**: Computes rolling averages and time-lagged features (1h, 2h, 6h, 24h) to give the model temporal context.

### 2. ML Ops & Hopsworks Integration (`scripts/pipelines/`)
- **`hopsworks_feature_upload.py`**: Uploads the engineered historical data to the `aqi_weather_features` Feature Group on Hopsworks.
- **`hopsworks_training.py`**: Pulls the entire dataset from the Feature Store, trains the XGBoost model, evaluates it (R² Score), and registers the new version in the Hopsworks Model Registry.
- **`predict_pipeline.py`**: Downloads the latest active model from the Registry, fetches lag data from the Feature Store, predicts the next 72 hours, and pushes the results to the `aqi_predictions` (v2) Feature Group.

### 3. Frontend
- **`dashboard.py`**: A visually rich Streamlit dashboard that reads predictions directly from Hopsworks and displays them with metric cards, dynamic line charts, and SHAP feature importance plots.

---

## 🚀 Cloud Automation (GitHub Actions)

This project contains two automated workflows in `.github/workflows/`:
1. **`data_pipeline.yml`**: Runs **every hour**. It triggers data fetching, feature engineering, feature store uploads, and the prediction pipeline.
2. **`training_pipeline.yml`**: Runs **every 24 hours** (at midnight). It triggers model retraining on the newly accumulated data to ensure the model stays accurate over time.

*(Note: To activate these pipelines, you must add your `HOPSWORKS_API_KEY` to your GitHub Repository Secrets).*

---

## 💻 Local Setup (If running manually)

Even though the project runs in the cloud, you can test it locally.

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Environment Variables**:
   Create a `.env` file in the root directory and add your Hopsworks API key:
   ```env
   HOPSWORKS_API_KEY="your_api_key_here"
   ```

3. **Run the Dashboard Locally**:
   ```bash
   streamlit run dashboard.py
   ```

---

## 🔄 Status / Recent Updates
- **[Completed]** Migrated local CSV data pipelines to Hopsworks Feature Store.
- **[Completed]** Model Registry implementation and dynamic model fetching.
- **[Completed]** Handled Schema compatibility issues (`version=2` created for predictions).
- **[Completed]** Automated GitHub Actions workflows generated.
- **[Current]** Project is fully operational and awaiting GitHub push to begin live cloud orchestration.
