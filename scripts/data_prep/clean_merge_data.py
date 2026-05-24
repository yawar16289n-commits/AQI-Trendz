import pandas as pd

def clean_and_merge():
    print("Loading weather data...")
    # Read weather data, skipping the 3 metadata rows
    weather_df = pd.read_csv('../../data/raw/weather-hsitory.csv', skiprows=3)
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    
    # Drop rows where all weather features are NaN
    weather_cols = [col for col in weather_df.columns if col != 'time']
    weather_df.dropna(subset=weather_cols, how='all', inplace=True)
    
    # Forward fill any intermittent missing values in weather data
    weather_df.ffill(inplace=True)
    
    print("Loading cleaned AQI data...")
    aqi_df = pd.read_csv('../../data/cleaned/aqi_cleaned.csv')
    aqi_df['time'] = pd.to_datetime(aqi_df['time'])
    
    print("Merging datasets on 'time'...")
    # Inner merge to keep only rows where we have both AQI and Weather data
    merged_df = pd.merge(aqi_df, weather_df, on='time', how='inner')
    
    # Rename columns to be Hopsworks compliant (lowercase, no spaces, no special chars)
    # Hopsworks feature names must match ^[a-z][a-z0-9_]*$
    def clean_col_name(col):
        col = col.lower()
        col = col.split('(')[0].strip() # Remove units like (μg/m³) or (°C)
        col = col.replace(' ', '_').replace('-', '_')
        return col
        
    merged_df.columns = [clean_col_name(col) for col in merged_df.columns]
    
    print(f"Merged shape: {merged_df.shape}")
    
    output_file = '../../data/cleaned/final_historical_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Successfully saved merged data to {output_file}")
    
    print("\nFirst 3 rows:")
    # Safe print without special characters
    print(merged_df.head(3).to_string(index=False))

if __name__ == "__main__":
    clean_and_merge()
