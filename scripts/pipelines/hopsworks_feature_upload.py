"""
hopsworks_feature_upload.py
----------------------------
Uploads ONLY NEW rows from data/cleaned/latest_featured.csv to the Hopsworks
Feature Group. Compares timestamps already in Hopsworks and skips duplicates.

Works both locally (reads HOPSWORKS_API_KEY from .env) and on GitHub Actions
(reads HOPSWORKS_API_KEY from environment secrets).
"""

import os
import sys
import pandas as pd
import hopsworks
from dotenv import load_dotenv

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

# Load .env locally; on GitHub Actions the secret is already in the environment
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'latest_featured.csv')
FG_NAME    = "aqi_weather_features"
FG_VERSION = 1


def upload_incremental():
    print("=== Incremental Upload to Hopsworks Feature Store ===")

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY not found.")
        sys.exit(1)

    print("Logging into Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    print(f"Loading latest featured data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    print(f"  Total rows available: {len(df)}")

    fg = fs.get_or_create_feature_group(
        name=FG_NAME,
        version=FG_VERSION,
        description="Historical AQI and Weather data with Lags and Rolling Features",
        primary_key=["time"],
        event_time="time",
        online_enabled=True,
    )

    # Check what timestamps are already in Hopsworks
    try:
        existing = fg.read()
        
        # --- ENFORCE SCHEMA MATCH ---
        # Hopsworks complains if new data has extra columns or wrong types compared to v1 schema
        if 'us_aqi' in df.columns:
            df = df.drop(columns=['us_aqi'])
        if 'cloud_cover' in df.columns:
            df = df.drop(columns=['cloud_cover'])
        if 'ozone' not in df.columns and 'ozone' in existing.columns:
             # Add ozone back as NaN if it was in the original schema but missing here
             df['ozone'] = 0.0
             
        if 'relative_humidity_2m' in df.columns:
            df['relative_humidity_2m'] = df['relative_humidity_2m'].fillna(0).astype('int64')
        if 'wind_direction_10m' in df.columns:
            df['wind_direction_10m'] = df['wind_direction_10m'].fillna(0).astype('int64')
        
        # Ensure only columns present in Hopsworks are kept
        cols_to_keep = [c for c in df.columns if c in existing.columns]
        df = df[cols_to_keep]
        # ----------------------------

        if len(existing) > 0:
            existing['time'] = pd.to_datetime(existing['time'], utc=True)
            max_existing_time = existing['time'].max()
            new_rows = df[df['time'] > max_existing_time]
            print(f"  Last timestamp in Hopsworks: {max_existing_time}")
            print(f"  New rows to insert: {len(new_rows)}")
        else:
            new_rows = df
            print(f"  Feature Group is empty — inserting all {len(new_rows)} rows.")
    except Exception:
        new_rows = df
        print(f"  Could not read existing data — inserting all {len(new_rows)} rows.")

    if len(new_rows) == 0:
        print("[INFO] No new rows to insert. Feature Store is already up to date.")
        return

    fg.insert(new_rows, write_options={"wait_for_job": False})
    print(f"\n[SUCCESS] Inserted {len(new_rows)} new rows into Feature Group '{FG_NAME}'.")


if __name__ == "__main__":
    upload_incremental()
