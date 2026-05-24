import hopsworks
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def upload_to_hopsworks(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Hopsworks expects timestamp columns to be properly parsed or strings. Let's make sure it's datetime.
    df['time'] = pd.to_datetime(df['time'])
    
    print("Connecting to Hopsworks...")
    # This will use the HOPSWORKS_API_KEY environment variable automatically
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    print("Creating/Retrieving Feature Group 'aqi_karachi'...")
    # Create a feature group
    aqi_fg = fs.get_or_create_feature_group(
        name="aqi_karachi",
        version=1,
        primary_key=["time"], 
        description="Historical AQI data for Karachi, Pakistan",
        event_time="time" # Tells Hopsworks this is time-series data
    )
    
    print("Uploading data to Feature Store... (This may take a few minutes)")
    aqi_fg.insert(df)
    
    print("✅ Successfully uploaded AQI data to Hopsworks Feature Store!")

if __name__ == "__main__":
    upload_to_hopsworks('../../data/cleaned/featured_historical_data.csv')
