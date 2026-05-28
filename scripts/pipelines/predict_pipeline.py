import pandas as pd
import numpy as np
import os
import joblib
import json
import requests
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
HISTORICAL_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'featured_historical_data.csv')
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'predictions.csv')

def calc_aqi_subindex(c, breakpoints):
    """Calculate US EPA AQI subindex given concentration c and list of tuples (C_low, C_high, I_low, I_high)"""
    for (c_low, c_high, i_low, i_high) in breakpoints:
        if c_low <= c <= c_high:
            # US EPA Formula
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return round(aqi)
    
    # If it exceeds the max breakpoint, extrapolate from the highest interval
    last = breakpoints[-1]
    aqi = ((last[3] - last[2]) / (last[1] - last[0])) * (c - last[0]) + last[2]
    return round(aqi)

def get_overall_aqi(pm25, pm10, no2, so2, co):
    """Convert raw hourly concentrations to US EPA AQI (approximated for 1-hour)."""
    
    # PM2.5 (ug/m3)
    pm25_bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    
    # PM10 (ug/m3)
    pm10_bp = [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500)
    ]
    
    # NO2 (ppb) -> note: Open-Meteo returns ug/m3, we assume 1 ug/m3 NO2 ≈ 0.53 ppb at standard temp/pressure
    no2_ppb = no2 * 0.53
    no2_bp = [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500)
    ]
    
    # SO2 (ppb) -> 1 ug/m3 SO2 ≈ 0.38 ppb
    so2_ppb = so2 * 0.38
    so2_bp = [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500)
    ]
    
    # CO (ppm) -> Open-Meteo returns ug/m3, 1 ug/m3 CO ≈ 0.000873 ppm
    co_ppm = co * 0.000873
    co_bp = [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500)
    ]
    
    # Calculate sub-indices
    aqi_pm25 = calc_aqi_subindex(pm25, pm25_bp)
    aqi_pm10 = calc_aqi_subindex(pm10, pm10_bp)
    aqi_no2 = calc_aqi_subindex(no2_ppb, no2_bp)
    aqi_so2 = calc_aqi_subindex(so2_ppb, so2_bp)
    aqi_co = calc_aqi_subindex(co_ppm, co_bp)
    
    # Overall AQI is the maximum of all sub-indices
    return max(aqi_pm25, aqi_pm10, aqi_no2, aqi_so2, aqi_co)


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
        
        # Predict all 5 pollutants
        preds = model.predict(features)[0]
        # MultiOutputRegressor order based on TARGET_COLS list:
        # ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
        p_pm25, p_pm10, p_no2, p_so2, p_co = preds
        
        # Calculate official US EPA AQI
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
        
        # Append predictions to the tracking arrays to serve as lags for the next loop iteration
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
