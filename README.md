# 🌍 AQIPrediction_10Pearls_Cohort8

**Live App:** [AQI Trendz Karachi Dashboard](https://aqi-trendz-10pearls-cohort8.streamlit.app/)

> **⚠️ Note on Loading Speed:** The dashboard runs a live machine learning prediction engine on startup. Due to latency with the Hopsworks Feature Store, the initial page load may take up to **30 seconds**. Please be patient! Once loaded, the predictions are cached for a smooth experience.
## 📌 Overview

**AQIPrediction_10Pearls_Cohort8** is a Machine Learning-based web application designed to forecast the Air Quality Index (AQI) and 5 major pollutants for the next **3 Days (72 hours)**. Focusing on Karachi, Pakistan, the project aims to provide users with early insights into future air quality conditions, enabling them to make informed decisions regarding outdoor activities, travel, and health-related precautions.

The system leverages a fully automated **Serverless & Cloud-Native** pipeline. It continuously ingests data, performs feature engineering, and uses predictive machine learning techniques (XGBoost & Random Forest) to analyze historical AQI patterns. An interactive Streamlit dashboard allows users to seamlessly access predictions, view historical trends, and explore model explanations via SHAP.

## ✨ Features

- **3-Day Forecast:** Predicts overall US EPA AQI and specific levels for PM2.5, PM10, NO₂, SO₂, and CO.
- **Advanced Machine Learning:** Utilizes tuned **XGBoost** and **Random Forest** multi-output regressors dynamically selected based on R² scores.
- **Automated Data Pipeline:** Scheduled GitHub Actions fetch live weather/AQI data hourly and update the central feature store.
- **Feature Engineering:** Computes complex historical lags (1h, 2h, 6h, 24h) and rolling statistics to capture temporal pollution persistence.
- **Interactive Dashboard:** A dark-themed, user-friendly Streamlit interface offering interactive charts with EPA-standard zone bands.
- **Explainable AI (XAI):** Built-in SHAP (SHapley Additive exPlanations) visualizer allows users to understand exactly which environmental factors (e.g., wind speed, temperature) are driving current pollution predictions.

## 🛠 Tech Stack

- **Core Languages:** Python 3.10
- **Data Engineering:** Pandas, NumPy, Open-Meteo API
- **Machine Learning:** Scikit-learn, XGBoost, SHAP
- **Cloud Infrastructure:** Hopsworks (Feature Store & Model Registry), GitHub Actions (CI/CD & Cron Scheduling)
- **Web Interface:** Streamlit
- **Data Visualization:** Plotly, Matplotlib, Seaborn

## 🔄 Project Workflow

1. **Data Collection:** Automated scripts fetch hourly ground-truth AQI and meteorological data.
2. **Data Cleaning & Preprocessing:** Handles missing values and merges environmental factors.
3. **Feature Engineering:** Generates time-lags, rolling means, and cyclical time variables (hour, day of week) and pushes them to the **Hopsworks Feature Store**.
4. **Model Training & Evaluation:** Models are trained on historical data, evaluated, and the best-performing models are deployed to the **Hopsworks Model Registry**.
5. **AQI Forecast Generation:** An autoregressive pipeline fetches the model, predicts the next 72 hours iteratively, and caches the results.
6. **Interactive Visualization:** The Streamlit dashboard reads the cached predictions and renders them for the end user.

## 🎯 Objective

The primary objective of this project is to develop a highly reliable and interpretable AQI forecasting system. By open-sourcing these predictions through an accessible dashboard, the project helps individuals and organizations in Karachi anticipate dangerous air quality spikes and take proactive measures to minimize severe health risks associated with particulate matter and chemical pollution.
