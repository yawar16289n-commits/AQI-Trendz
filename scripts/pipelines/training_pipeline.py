"""
Multi-Output Training Pipeline - Chunk 3
Trains MultiOutputRegressor baseline models (Linear Regression, Random Forest, XGBoost)
on the featured historical AQI data to predict 5 AQI pollutants simultaneously.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# XGBoost
from xgboost import XGBRegressor

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

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
    # Advanced Feature 1: Lag features for all 5 pollutants
    'pm2_5_lag_1h', 'pm2_5_lag_24h',
    'pm10_lag_1h', 'pm10_lag_24h',
    'nitrogen_dioxide_lag_1h', 'nitrogen_dioxide_lag_24h',
    'sulphur_dioxide_lag_1h', 'sulphur_dioxide_lag_24h',
    'carbon_monoxide_lag_1h', 'carbon_monoxide_lag_24h',
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

def evaluate_model(name, model, X_test, y_test):
    """Evaluate a multi-output model and return metrics dict."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each target
    maes = []
    rmses = []
    r2s = []
    
    print(f"\n  [{name}] Per-Pollutant Evaluation:")
    for i, target in enumerate(TARGET_COLS):
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        print(f"    {target:20s}: MAE={mae:6.2f} | R2={r2:6.4f}")
        
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_r2 = np.mean(r2s)
    
    print(f"    -> AVERAGE SCORE       : MAE={avg_mae:6.2f} | R2={avg_r2:6.4f}")

    return {
        'name': name,
        'mae': round(avg_mae, 4),
        'rmse': round(avg_rmse, 4),
        'r2': round(avg_r2, 4),
    }

def train_models(X_train, y_train, X_test, y_test):
    """Train all baseline multi-output models and return results."""
    models = {
        'Linear Regression': LinearRegression(), # LR supports multi-output natively
        'Random Forest': RandomForestRegressor( # RF supports multi-output natively
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        'XGBoost (MultiOutput)': MultiOutputRegressor( # XGBoost needs wrapper for strict multi-output
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
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model

    return results, trained_models

def save_history_to_csv(results, target_cols):
    """Save training results to a history CSV file."""
    history_file = os.path.join(MODELS_DIR, 'model_history.csv')
    timestamp = datetime.now().isoformat()
    
    rows = []
    for r in results:
        row = {
            'timestamp': timestamp,
            'model_name': r['name'],
            'target': "MULTI_OUTPUT",
            'mae': r['mae'],
            'rmse': r['rmse'],
            'r2': r['r2']
        }
        rows.append(row)
    
    df_new = pd.DataFrame(rows)
    if os.path.exists(history_file):
        df_hist = pd.read_csv(history_file)
        df_hist = pd.concat([df_hist, df_new], ignore_index=True)
    else:
        df_hist = df_new
        
    df_hist.to_csv(history_file, index=False)
    print(f"  Training history appended to {history_file}")

def save_best_model(results, trained_models, X_train):
    """Save the best performing model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Pick the model with the lowest average MAE
    best = min(results, key=lambda x: x['mae'])
    best_name = best['name']
    best_model = trained_models[best_name]

    print(f"\n=== Best Multi-Output Model: {best_name} (Avg MAE: {best['mae']}) ===")

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

def print_comparison_table(results):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 60)
    print("      MULTI-OUTPUT BASELINE MODEL COMPARISON (Averages)")
    print("=" * 60)
    print(f"  {'Model':<25} {'Avg MAE':>8} {'Avg RMSE':>10} {'Avg R2':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['mae']):
        print(f"  {r['name']:<25} {r['mae']:>8.4f} {r['rmse']:>10.4f} {r['r2']:>8.4f}")
    print("=" * 60)

def main():
    df = load_data()
    X_train, y_train, X_test, y_test = chronological_split(df)
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    print_comparison_table(results)
    best = save_best_model(results, trained_models, X_train)
    save_history_to_csv(results, TARGET_COLS)
    print(f"\n[DONE] Training pipeline complete. Best model: {best['name']}")

if __name__ == "__main__":
    main()
