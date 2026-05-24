import openmeteo_requests
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_data(lat, lon, start_date, end_date):
    print(f"Fetching data for Karachi (Lat: {lat}, Lon: {lon}) from {start_date} to {end_date}...")
    
    # Setup the Open-Meteo API client
    session = requests.Session()
    openmeteo = openmeteo_requests.Client(session = session)

    # 1. Fetch Air Quality Data
    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]
    }
    print("-> Requesting Air Quality data...")
    aqi_responses = openmeteo.weather_api(aqi_url, params=aqi_params)
    aqi_response = aqi_responses[0]
    
    # Process AQI hourly data
    hourly_aqi = aqi_response.Hourly()
    aqi_data = {"time": pd.date_range(
        start = pd.to_datetime(hourly_aqi.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly_aqi.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly_aqi.Interval()),
        inclusive = "left"
    )}
    
    aqi_data["pm10"] = hourly_aqi.Variables(0).ValuesAsNumpy()
    aqi_data["pm2_5"] = hourly_aqi.Variables(1).ValuesAsNumpy()
    aqi_data["carbon_monoxide"] = hourly_aqi.Variables(2).ValuesAsNumpy()
    aqi_data["nitrogen_dioxide"] = hourly_aqi.Variables(3).ValuesAsNumpy()
    aqi_data["sulphur_dioxide"] = hourly_aqi.Variables(4).ValuesAsNumpy()
    aqi_data["ozone"] = hourly_aqi.Variables(5).ValuesAsNumpy()
    aqi_data["us_aqi"] = hourly_aqi.Variables(6).ValuesAsNumpy()
    
    aqi_df = pd.DataFrame(data = aqi_data)
    
    # 2. Fetch Historical Weather Data
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure", "wind_speed_10m", "wind_direction_10m", "cloud_cover"]
    }
    print("-> Requesting Historical Weather data...")
    weather_responses = openmeteo.weather_api(weather_url, params=weather_params)
    weather_response = weather_responses[0]
    
    # Process Weather hourly data
    hourly_weather = weather_response.Hourly()
    weather_data = {"time": pd.date_range(
        start = pd.to_datetime(hourly_weather.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly_weather.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly_weather.Interval()),
        inclusive = "left"
    )}
    
    weather_data["temperature_2m"] = hourly_weather.Variables(0).ValuesAsNumpy()
    weather_data["relative_humidity_2m"] = hourly_weather.Variables(1).ValuesAsNumpy()
    weather_data["precipitation"] = hourly_weather.Variables(2).ValuesAsNumpy()
    weather_data["surface_pressure"] = hourly_weather.Variables(3).ValuesAsNumpy()
    weather_data["wind_speed_10m"] = hourly_weather.Variables(4).ValuesAsNumpy()
    weather_data["wind_direction_10m"] = hourly_weather.Variables(5).ValuesAsNumpy()
    weather_data["cloud_cover"] = hourly_weather.Variables(6).ValuesAsNumpy()
    
    weather_df = pd.DataFrame(data = weather_data)
    
    # 3. Merge the datasets on 'time'
    print("-> Merging datasets...")
    merged_df = pd.merge(aqi_df, weather_df, on="time", how="inner")
    
    # Save to CSV
    output_file = "historical_aqi_weather_karachi.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(merged_df)} hourly records to {output_file}!")
    
    return merged_df

if __name__ == "__main__":
    LAT = 24.933
    LON = 67.033
    
    end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    start_date = "2022-08-01"
    
    try:
        df = fetch_historical_data(LAT, LON, start_date, end_date)
        print("\nFirst 5 rows of the combined dataset:")
        print(df.head())
    except Exception as e:
        print(f"Error fetching data: {e}")
