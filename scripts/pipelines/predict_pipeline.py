import pandas as pd
import numpy as np
import os
import sys
import joblib
import json
import requests
from datetime import datetime, timedelta

# Setup path so we can import our new utils module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils.aqi_calculator import get_overall_aqi

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
HISTORICAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'featured_historical_data.csv')
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'predictions.csv')

def fetch_weather_forecast():
    """Fetch 3-day hourly weather forecast for Karachi from Open-Meteo."""
    print("Fetching 3-day weather forecast...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 24.8608,
        "longitude": 67.0104,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "forecast_days": 3,
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    df = pd.DataFrame(data['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    return df

def predict_future_aqi():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return
        
    model = joblib.load(MODEL_PATH)
    forecast_df = fetch_weather_forecast()
    
    # Add basic time features
    forecast_df['hour'] = forecast_df['time'].dt.hour
    forecast_df['day'] = forecast_df['time'].dt.day
    forecast_df['month'] = forecast_df['time'].dt.month
    forecast_df['day_of_week'] = forecast_df['time'].dt.dayofweek
    forecast_df['is_weekend'] = forecast_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Load historical data to seed the 24-hour lags
    hist_df = pd.read_csv(HISTORICAL_DATA_PATH)
    hist_df['time'] = pd.to_datetime(hist_df['time'])
    
    # Initialize lag arrays with the most recent 24 hours
    recent_pm25 = hist_df['pm2_5'].tail(24).tolist()
    recent_pm10 = hist_df['pm10'].tail(24).tolist()
    recent_no2 = hist_df['nitrogen_dioxide'].tail(24).tolist()
    recent_so2 = hist_df['sulphur_dioxide'].tail(24).tolist()
    recent_co = hist_df['carbon_monoxide'].tail(24).tolist()
    
    pred_records = []
    
    print(f"Predicting all 5 AQI pollutants for the next 72 hours (Multi-Output Autoregression)...")
    
    for i, row in forecast_df.iterrows():
        # Build feature vector matching training features perfectly
        features = pd.DataFrame([{
            'temperature_2m': row['temperature_2m'],
            'relative_humidity_2m': row['relative_humidity_2m'],
            'precipitation': row['precipitation'],
            'surface_pressure': row['surface_pressure'],
            'wind_speed_10m': row['wind_speed_10m'],
            'wind_direction_10m': row['wind_direction_10m'],
            'hour': row['hour'],
            'day': row['day'],
            'month': row['month'],
            'day_of_week': row['day_of_week'],
            'is_weekend': row['is_weekend'],
            'pm2_5_lag_1h': recent_pm25[-1],
            'pm2_5_lag_24h': recent_pm25[-24],
            'pm10_lag_1h': recent_pm10[-1],
            'pm10_lag_24h': recent_pm10[-24],
            'nitrogen_dioxide_lag_1h': recent_no2[-1],
            'nitrogen_dioxide_lag_24h': recent_no2[-24],
            'sulphur_dioxide_lag_1h': recent_so2[-1],
            'sulphur_dioxide_lag_24h': recent_so2[-24],
            'carbon_monoxide_lag_1h': recent_co[-1],
            'carbon_monoxide_lag_24h': recent_co[-24],
        }])
        
        preds = model.predict(features)[0]
        p_pm25, p_pm10, p_no2, p_so2, p_co = preds
        
        # Using the refactored AQI utility
        final_aqi = get_overall_aqi(p_pm25, p_pm10, p_no2, p_so2, p_co)
        
        pred_records.append({
            'time': row['time'],
            'pred_pm2_5': p_pm25,
            'pred_pm10': p_pm10,
            'pred_no2': p_no2,
            'pred_so2': p_so2,
            'pred_co': p_co,
            'US_EPA_AQI': final_aqi
        })
        
        recent_pm25.append(p_pm25)
        recent_pm10.append(p_pm10)
        recent_no2.append(p_no2)
        recent_so2.append(p_so2)
        recent_co.append(p_co)
        
    output_df = pd.DataFrame(pred_records)
    
    # Merge weather data for context in dashboard
    final_output = pd.merge(output_df, forecast_df[['time', 'temperature_2m', 'wind_speed_10m']], on='time')
    
    final_output.to_csv(PREDICTIONS_PATH, index=False)
    print(f"[SUCCESS] 3-Day Multi-Pollutant Forecast saved to {PREDICTIONS_PATH}")
    
    print("\nFirst 10 hours forecast preview:")
    preview = final_output[['time', 'pred_pm2_5', 'pred_pm10', 'US_EPA_AQI']].head(10)
    print(preview.to_string(index=False))

if __name__ == "__main__":
    predict_future_aqi()
