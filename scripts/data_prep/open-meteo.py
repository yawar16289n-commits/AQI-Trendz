import openmeteo_requests
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Setup the Open-Meteo API client
session = requests.Session()
openmeteo = openmeteo_requests.Client(session=session)

def fetch_chunked_data(lat, lon, start_date_str, end_date_str, url, hourly_vars):
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    current_start = start_dt
    all_dfs = []
    
    while current_start <= end_dt:
        # Fetch in 60-day chunks to prevent 504 Timeouts
        current_end = current_start + timedelta(days=60)
        if current_end > end_dt:
            current_end = end_dt
            
        c_start_str = current_start.strftime("%Y-%m-%d")
        c_end_str = current_end.strftime("%Y-%m-%d")
        print(f"   -> Fetching {c_start_str} to {c_end_str}")
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": c_start_str,
            "end_date": c_end_str,
            "hourly": hourly_vars,
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            hourly = response.Hourly()
            hourly_data = {
                "time": pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left"
                )
            }
            
            for i, var in enumerate(hourly_vars):
                hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()
                
            all_dfs.append(pd.DataFrame(data=hourly_data))
        except Exception as e:
            print(f"      [Warning] Chunk failed: {e}")
            
        current_start = current_end + timedelta(days=1)
        time.sleep(1) # Rate limiting
        
    return pd.concat(all_dfs, ignore_index=True)

def fetch_historical_weather(lat, lon, start_date, end_date):
    print("Fetching historical weather (chunked)...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m", "surface_pressure"]
    return fetch_chunked_data(lat, lon, start_date, end_date, url, hourly_vars)

def fetch_historical_aqi(lat, lon, start_date, end_date):
    print("Fetching historical AQI (chunked)...")
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    hourly_vars = ["pm10", "pm2_5", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone", "us_aqi"]
    return fetch_chunked_data(lat, lon, start_date, end_date, url, hourly_vars)

def fetch_weather_forecast(lat, lon, forecast_days=3):
    print(f"Fetching {forecast_days}-day weather forecast...")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "wind_speed_10m", "wind_direction_10m"],
        "forecast_days": forecast_days,
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    hourly_data = {
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    
    hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
    hourly_data["relative_humidity_2m"] = hourly.Variables(1).ValuesAsNumpy()
    hourly_data["precipitation"] = hourly.Variables(2).ValuesAsNumpy()
    hourly_data["surface_pressure"] = hourly.Variables(3).ValuesAsNumpy()
    hourly_data["wind_speed_10m"] = hourly.Variables(4).ValuesAsNumpy()
    hourly_data["wind_direction_10m"] = hourly.Variables(5).ValuesAsNumpy()
    
    return pd.DataFrame(data=hourly_data)

if __name__ == "__main__":
    LAT = 24.933
    LON = 67.033
    
    # Using small recent window to test
    START_DATE = "2023-01-01"
    END_DATE = "2024-05-20"
    
    try:
        # 1. Fetch historical datasets
        weather_df = fetch_historical_weather(LAT, LON, START_DATE, END_DATE)
        aqi_df = fetch_historical_aqi(LAT, LON, START_DATE, END_DATE)
        
        # Merge them
        print("Merging historical datasets...")
        historical_df = pd.merge(aqi_df, weather_df, on="time", how="inner")
        
        # Save to CSV
        historical_df.to_csv("karachi_historical_data.csv", index=False)
        print(f"[SUCCESS] Saved {len(historical_df)} historical records to karachi_historical_data.csv")
        
        # 2. Fetch future forecast (This will be used later in Chunk 5 for predictions)
        forecast_df = fetch_weather_forecast(LAT, LON, forecast_days=3)
        forecast_df.to_csv("karachi_forecast_data.csv", index=False)
        print(f"[SUCCESS] Saved {len(forecast_df)} forecast records to karachi_forecast_data.csv")
        
    except Exception as e:
        print(f"[ERROR] Error occurred: {e}")