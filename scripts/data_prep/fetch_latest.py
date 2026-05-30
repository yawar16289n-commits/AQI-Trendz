"""
fetch_latest.py
---------------
Fetches the last 7 days of AQI + weather data from Open-Meteo APIs.
Designed to run both locally and on GitHub Actions (no .env needed for API keys
since Open-Meteo is free and public).

Output: data/raw/latest_raw.csv
"""

import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'latest_raw.csv')

LAT = 24.933
LON = 67.033

def fetch_latest():
    print("=== Fetching Latest Data from Open-Meteo ===")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Setup client with cache and retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # ── 1. WEATHER: Last 7 days + 3-day forecast ──────────────────────────────
    print("Fetching weather forecast (past 7 days + next 3 days)...")
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
        ],
        "past_days": 7,
        "forecast_days": 3,
    }
    weather_resp = openmeteo.weather_api(weather_url, params=weather_params)[0]
    hourly = weather_resp.Hourly()

    weather_df = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "temperature_2m":       hourly.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
        "precipitation":        hourly.Variables(2).ValuesAsNumpy(),
        "surface_pressure":     hourly.Variables(3).ValuesAsNumpy(),
        "wind_speed_10m":       hourly.Variables(4).ValuesAsNumpy(),
        "wind_direction_10m":   hourly.Variables(5).ValuesAsNumpy(),
        "cloud_cover":          hourly.Variables(6).ValuesAsNumpy(),
    })
    print(f"  Weather rows fetched: {len(weather_df)}")

    # ── 2. AQI: Last 7 days (historical only — no future AQI available) ───────
    print("Fetching AQI data (past 7 days)...")
    today = datetime.utcnow().strftime('%Y-%m-%d')
    seven_days_ago = (datetime.utcnow() - pd.Timedelta(days=7)).strftime('%Y-%m-%d')

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": seven_days_ago,
        "end_date": today,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "us_aqi"],
    }
    aqi_resp = openmeteo.weather_api(aqi_url, params=aqi_params)[0]
    hourly_aqi = aqi_resp.Hourly()

    aqi_df = pd.DataFrame({
        "time": pd.date_range(
            start=pd.to_datetime(hourly_aqi.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly_aqi.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly_aqi.Interval()),
            inclusive="left",
        ),
        "pm10":               hourly_aqi.Variables(0).ValuesAsNumpy(),
        "pm2_5":              hourly_aqi.Variables(1).ValuesAsNumpy(),
        "carbon_monoxide":    hourly_aqi.Variables(2).ValuesAsNumpy(),
        "nitrogen_dioxide":   hourly_aqi.Variables(3).ValuesAsNumpy(),
        "sulphur_dioxide":    hourly_aqi.Variables(4).ValuesAsNumpy(),
        "us_aqi":             hourly_aqi.Variables(5).ValuesAsNumpy(),
    })
    print(f"  AQI rows fetched: {len(aqi_df)}")

    # ── 3. Merge (inner join on time — only rows with BOTH weather & AQI) ─────
    merged = pd.merge(aqi_df, weather_df, on="time", how="inner")
    merged.dropna(subset=["pm2_5", "pm10"], inplace=True)
    print(f"  Merged rows (historical only): {len(merged)}")

    # Save the weather-only future rows separately (for prediction later)
    future_weather = weather_df[~weather_df["time"].isin(aqi_df["time"])].copy()
    future_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'future_weather.csv')
    future_weather.to_csv(future_path, index=False)
    print(f"  Future weather rows (for forecast): {len(future_weather)} -> saved to {future_path}")

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Latest raw data saved to {OUTPUT_PATH}")
    return merged

if __name__ == "__main__":
    fetch_latest()
