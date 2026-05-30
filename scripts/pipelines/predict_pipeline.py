"""
predict_pipeline.py
--------------------
Cloud-native prediction pipeline:
  1. Reads the 3-day future weather forecast (data/raw/future_weather.csv)
  2. Downloads the latest XGBoost model from Hopsworks Model Registry
  3. Fetches the last 24 hours of lag data from Hopsworks Feature Store
  4. Constructs the feature vector for each future hour
  5. Runs predictions for all 5 pollutants
  6. Computes the US EPA AQI from predicted PM2.5
  7. Saves predictions to data/cleaned/predictions.csv (for dashboard)
  8. Pushes predictions to Hopsworks 'aqi_predictions' Feature Group

Works both locally (reads HOPSWORKS_API_KEY from .env) and on GitHub Actions.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

FUTURE_WEATHER_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'future_weather.csv')
PREDICTIONS_PATH    = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'predictions.csv')
LOCAL_MODEL_PATH    = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')

POLLUTANTS = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
LAG_HOURS  = [1, 2, 6, 24]

# US EPA AQI breakpoints for PM2.5
def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            return round(((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo)
    return 500


def run():
    print("=== Predict Pipeline (Cloud-Native) ===")

    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY not found.")
        sys.exit(1)

    # ── 1. Login to Hopsworks ─────────────────────────────────────────────────
    print("Logging into Hopsworks...")
    project = hopsworks.login()
    fs  = project.get_feature_store()
    mr  = project.get_model_registry()

    # ── 2. Load model from Hopsworks Model Registry ───────────────────────────
    print("Downloading model from Hopsworks Model Registry...")
    try:
        hw_model = mr.get_best_model(
            name="xgboost_multi_aqi",
            metric="r2",
            direction="max"
        )
        model_dir  = hw_model.download()
        model_path = os.path.join(model_dir, "xgboost_multi_aqi.pkl")
        model      = joblib.load(model_path)
        print(f"  Model loaded from Registry (r2={hw_model.training_metrics.get('r2', 'N/A')})")
    except Exception as e:
        print(f"  WARNING: Could not load model from Registry ({e}). Falling back to local model.")
        if not os.path.exists(LOCAL_MODEL_PATH):
            print("ERROR: No local model found either. Please run hopsworks_training.py first.")
            sys.exit(1)
        model = joblib.load(LOCAL_MODEL_PATH)
        print(f"  Local model loaded from {LOCAL_MODEL_PATH}")

    # ── 3. Fetch last 24h of lag data from Hopsworks Feature Store ────────────
    print("Fetching last 24 hours of data from Feature Store for lag context...")
    fg = fs.get_feature_group(name="aqi_weather_features", version=1)
    historical = fg.read()
    historical['time'] = pd.to_datetime(historical['time'], utc=True)
    historical = historical.sort_values('time').tail(48)  # keep extra for safety
    print(f"  Historical rows fetched for lag context: {len(historical)}")

    # ── 4. Load future weather ────────────────────────────────────────────────
    print(f"Loading future weather from {FUTURE_WEATHER_PATH}...")
    future = pd.read_csv(FUTURE_WEATHER_PATH)
    future['time'] = pd.to_datetime(future['time'], utc=True)
    future = future.sort_values('time').reset_index(drop=True)
    print(f"  Future weather rows: {len(future)}")

    # ── 5. Build feature rows for each future hour ────────────────────────────
    print("Building feature vectors for future hours...")
    rows = []

    # Combine history + future into one frame for rolling lookback
    combined_pollutants = historical[['time'] + POLLUTANTS].copy()

    for i, row in future.iterrows():
        t = row['time']

        # Time features
        feat = {
            'time':        t,
            'hour':        t.hour,
            'day':         t.day,
            'month':       t.month,
            'day_of_week': t.dayofweek,
            'is_weekend':  int(t.dayofweek >= 5),
        }

        # Weather features
        for col in ['temperature_2m', 'relative_humidity_2m', 'precipitation',
                    'surface_pressure', 'wind_speed_10m', 'wind_direction_10m', 'cloud_cover']:
            feat[col] = row.get(col, np.nan)

        # Lag and rolling features from combined historical context
        for pol in POLLUTANTS:
            for lag in LAG_HOURS:
                lag_time = t - pd.Timedelta(hours=lag)
                past_rows = combined_pollutants[combined_pollutants['time'] <= lag_time]
                if len(past_rows) > 0:
                    feat[f'{pol}_lag_{lag}h'] = past_rows.iloc[-1][pol]
                else:
                    feat[f'{pol}_lag_{lag}h'] = np.nan

            # Rolling stats from past 24 rows available
            past_24 = combined_pollutants[
                (combined_pollutants['time'] < t) &
                (combined_pollutants['time'] >= t - pd.Timedelta(hours=24))
            ][pol]
            past_6 = combined_pollutants[
                (combined_pollutants['time'] < t) &
                (combined_pollutants['time'] >= t - pd.Timedelta(hours=6))
            ][pol]

            feat[f'{pol}_roll_mean_6h']  = past_6.mean()  if len(past_6)  > 0 else np.nan
            feat[f'{pol}_roll_std_6h']   = past_6.std()   if len(past_6)  > 1 else 0.0
            feat[f'{pol}_roll_mean_24h'] = past_24.mean() if len(past_24) > 0 else np.nan
            feat[f'{pol}_roll_std_24h']  = past_24.std()  if len(past_24) > 1 else 0.0

        rows.append(feat)

    X_future = pd.DataFrame(rows)

    # ── 6. Run predictions ────────────────────────────────────────────────────
    # Extract the exact feature names and order that the model was trained on
    train_features = [c for c in historical.columns if c not in POLLUTANTS and c != 'time']
    
    # Ensure X_future has exactly these columns in this order
    for c in train_features:
        if c not in X_future.columns:
            X_future[c] = 0.0
            
    X_input = X_future[train_features].fillna(0)

    print("Running XGBoost predictions...")
    preds = model.predict(X_input)
    preds_df = pd.DataFrame(preds, columns=[f'pred_{p}' for p in POLLUTANTS])

    # Compute US EPA AQI from predicted PM2.5
    preds_df['US_EPA_AQI'] = preds_df['pred_pm2_5'].apply(pm25_to_aqi)

    # Combine with time and weather
    result = pd.concat([X_future[['time'] + list(future.columns[1:])].reset_index(drop=True),
                        preds_df], axis=1)
    result['time'] = result['time'].astype(str)

    # ── 7. Save locally for dashboard ────────────────────────────────────────
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    result.to_csv(PREDICTIONS_PATH, index=False)
    print(f"  Predictions saved locally to {PREDICTIONS_PATH}")

    # ── 8. Push predictions to Hopsworks ─────────────────────────────────────
    print("Pushing predictions to Hopsworks 'aqi_predictions' Feature Group...")
    result['time'] = pd.to_datetime(result['time'], utc=True)

    pred_fg = fs.get_or_create_feature_group(
        name="aqi_predictions",
        version=2,  # Bump version to avoid schema mismatch with older predictions
        description="72-hour AQI and pollutant predictions",
        primary_key=["time"],
        event_time="time",
        online_enabled=True,
    )
    pred_fg.insert(result, write_options={"wait_for_job": False})

    print(f"\n[SUCCESS] Generated {len(result)} prediction rows and pushed to Hopsworks!")


if __name__ == "__main__":
    run()
