"""
Multi-Output Training Pipeline - Chunk 3
Trains MultiOutputRegressor baseline models (Linear Regression, Random Forest, XGBoost)
on the featured historical AQI data to predict 5 AQI pollutants simultaneously.
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from datetime import datetime

# Setup path so we can import our new utils module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils.evaluation import evaluate_multioutput_model, print_comparison_table, save_history_to_csv

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'featured_historical_data.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TEST_RATIO = 0.20       # 20% of data for testing (chronological split)

# Primary targets: all 5 AQI pollutants
TARGET_COLS = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']

# Features to use for training
FEATURE_COLS = [
    'temperature_2m',
    'relative_humidity_2m',
    'precipitation',
    'surface_pressure',
    'wind_speed_10m',
    'wind_direction_10m',
    'hour',
    'day',
    'month',
    'day_of_week',
    'is_weekend',
    # Lag features for all 5 pollutants
    'pm2_5_lag_1h', 'pm2_5_lag_2h', 'pm2_5_lag_6h', 'pm2_5_lag_24h',
    'pm10_lag_1h', 'pm10_lag_2h', 'pm10_lag_6h', 'pm10_lag_24h',
    'nitrogen_dioxide_lag_1h', 'nitrogen_dioxide_lag_2h', 'nitrogen_dioxide_lag_6h', 'nitrogen_dioxide_lag_24h',
    'sulphur_dioxide_lag_1h', 'sulphur_dioxide_lag_2h', 'sulphur_dioxide_lag_6h', 'sulphur_dioxide_lag_24h',
    'carbon_monoxide_lag_1h', 'carbon_monoxide_lag_2h', 'carbon_monoxide_lag_6h', 'carbon_monoxide_lag_24h',
    # Rolling features: 6h & 24h mean+std (Run B)
    'pm2_5_roll_mean_6h', 'pm2_5_roll_std_6h', 'pm2_5_roll_mean_24h', 'pm2_5_roll_std_24h',
    'pm10_roll_mean_6h', 'pm10_roll_std_6h', 'pm10_roll_mean_24h', 'pm10_roll_std_24h',
    'nitrogen_dioxide_roll_mean_6h', 'nitrogen_dioxide_roll_std_6h', 'nitrogen_dioxide_roll_mean_24h', 'nitrogen_dioxide_roll_std_24h',
    'sulphur_dioxide_roll_mean_6h', 'sulphur_dioxide_roll_std_6h', 'sulphur_dioxide_roll_mean_24h', 'sulphur_dioxide_roll_std_24h',
    'carbon_monoxide_roll_mean_6h', 'carbon_monoxide_roll_std_6h', 'carbon_monoxide_roll_mean_24h', 'carbon_monoxide_roll_std_24h',
]

def load_data():
    """Load and prepare the featured dataset."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"  Total records: {len(df)}")
    return df

def chronological_split(df):
    """Split data chronologically (no data leakage from future)."""
    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\n--- Chronological Split ---")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Test:  {len(test_df)} rows")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COLS]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COLS]

    return X_train, y_train, X_test, y_test

def train_models(X_train, y_train, X_test, y_test):
    """Train all baseline multi-output models and return results."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        'XGBoost (MultiOutput)': MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
            )
        ),
    }

    results = []
    trained_models = {}

    print("\n=== Training Multi-Output Baseline Models ===")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        # Using the refactored evaluation script that calculates AQI MAE
        metrics = evaluate_multioutput_model(name, model, X_test, y_test, TARGET_COLS)
        results.append(metrics)
        trained_models[name] = model

    return results, trained_models

def save_best_model(results, trained_models, X_train):
    """Save the best performing model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Pick the model with the lowest average MAE
    best = min(results, key=lambda x: x['mae'])
    best_name = best['name']
    best_model = trained_models[best_name]

    print(f"\n=== Best Multi-Output Model: {best_name} (Avg MAE: {best['mae']}, AQI MAE: {best['aqi_mae']}) ===")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"  Model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_name': best_name,
        'targets': TARGET_COLS,
        'features': FEATURE_COLS,
        'metrics': best,
        'trained_at': datetime.now().isoformat(),
        'train_samples': int(X_train.shape[0]),
    }
    meta_path = os.path.join(MODELS_DIR, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved to {meta_path}")

    # Save all results for comparison
    results_path = os.path.join(MODELS_DIR, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return best

def main():
    df = load_data()
    X_train, y_train, X_test, y_test = chronological_split(df)
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    print_comparison_table(results)
    best = save_best_model(results, trained_models, X_train)
    save_history_to_csv(results, MODELS_DIR, experiment_note="RUN-B | Lags: 1h,2h,6h,24h + Rolling: 6h & 24h mean+std | Testing additional 24h rolling on top of Run-A")
    print(f"\n[DONE] Training pipeline complete. Best model: {best['name']}")

if __name__ == "__main__":
    main()
