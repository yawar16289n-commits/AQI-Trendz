"""
feature_engineering_latest.py
-------------------------------
Reads the latest raw data (data/raw/latest_raw.csv), computes lag and rolling
features, and saves a feature-engineered version ready for Hopsworks ingestion.

Keeps the same feature schema as the full historical featured_historical_data.csv
so the trained model sees identical columns.

Output: data/cleaned/latest_featured.csv
"""

import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

INPUT_PATH  = os.path.join(PROJECT_ROOT, 'data', 'raw', 'latest_raw.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'latest_featured.csv')

POLLUTANTS = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    # Time features
    df['hour']        = df['time'].dt.hour
    df['day']         = df['time'].dt.day
    df['month']       = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend']  = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Lag features
    for pol in POLLUTANTS:
        df[f'{pol}_lag_1h']  = df[pol].shift(1)
        df[f'{pol}_lag_2h']  = df[pol].shift(2)
        df[f'{pol}_lag_6h']  = df[pol].shift(6)
        df[f'{pol}_lag_24h'] = df[pol].shift(24)

    # Rolling features
    for pol in POLLUTANTS:
        df[f'{pol}_roll_mean_6h']  = df[pol].shift(1).rolling(window=6,  min_periods=1).mean()
        df[f'{pol}_roll_std_6h']   = df[pol].shift(1).rolling(window=6,  min_periods=1).std().fillna(0)
        df[f'{pol}_roll_mean_24h'] = df[pol].shift(1).rolling(window=24, min_periods=1).mean()
        df[f'{pol}_roll_std_24h']  = df[pol].shift(1).rolling(window=24, min_periods=1).std().fillna(0)

    df.dropna(inplace=True)
    return df


def run():
    print("=== Feature Engineering on Latest Data ===")
    df = pd.read_csv(INPUT_PATH)
    print(f"  Raw rows loaded: {len(df)}")

    featured = engineer_features(df)
    print(f"  Rows after feature engineering: {len(featured)}")
    print(f"  Total features: {len(featured.columns)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    featured.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SUCCESS] Latest featured data saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
