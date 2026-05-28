import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'final_historical_data.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'featured_historical_data.csv')


def perform_feature_engineering():
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    df['time'] = pd.to_datetime(df['time'])

    # Sort chronologically
    df = df.sort_values('time').reset_index(drop=True)

    # ── BASIC TIME FEATURES (Baseline) ─────────────────────────────────────────
    print("Adding basic time features...")
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # ── ADVANCED FEATURE 1: LAG FEATURES (Multi-Pollutant) ─────────────────────
    print("Adding 1h and 24h lag features for all 5 target pollutants...")
    
    pollutants = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
    
    for pol in pollutants:
        df[f'{pol}_lag_1h'] = df[pol].shift(1)
        df[f'{pol}_lag_24h'] = df[pol].shift(24)

    # ── Drop NaN rows introduced by the 24h lag ───────────────────────────────
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"  Dropped {before - after} rows with NaN from lags.")

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[SUCCESS] Feature Engineering completed! Total Features: {len(df.columns)}")
    print(f"[SUCCESS] Data saved to {OUTPUT_PATH}")
    print(f"  Total rows: {len(df)}")
    print(f"\nFeature list:")
    print(list(df.columns))


if __name__ == "__main__":
    perform_feature_engineering()
