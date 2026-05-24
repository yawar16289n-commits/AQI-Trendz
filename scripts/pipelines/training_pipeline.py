"""
Training Pipeline - Chunk 3
Trains baseline models (Linear Regression, Random Forest, XGBoost) on
the featured historical AQI data using a chronological train/test split.
Evaluates models and saves the best one.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# XGBoost
from xgboost import XGBRegressor


# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'featured_historical_data.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
TARGET_COL = 'pm2_5'   # Primary target: predict PM2.5
TEST_RATIO = 0.20       # 20% of data for testing (chronological split)

# Features to use for training (exclude target + time + other pollutants we might predict later)
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
    # Advanced Feature 1: Lag features
    'pm2_5_lag_1h',
    'pm2_5_lag_24h',
    'pm10_lag_1h',
    'pm10_lag_24h',
]


def load_data():
    """Load and prepare the featured dataset."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"  Total records: {len(df)}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    return df


def chronological_split(df):
    """Split data chronologically (no data leakage from future)."""
    split_idx = int(len(df) * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"\n--- Chronological Split ---")
    print(f"  Train: {len(train_df)} rows ({train_df['time'].min()} to {train_df['time'].max()})")
    print(f"  Test:  {len(test_df)} rows ({test_df['time'].min()} to {test_df['time'].max()})")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return X_train, y_train, X_test, y_test


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a trained model and return metrics dict."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n  [{name}]")
    print(f"    MAE:  {mae:.4f}")
    print(f"    RMSE: {rmse:.4f}")
    print(f"    R2:   {r2:.4f}")

    return {
        'name': name,
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
    }


def train_models(X_train, y_train, X_test, y_test):
    """Train all baseline models and return results."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),
        'XGBoost': XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    print("\n=== Training Baseline Models ===")
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model

    return results, trained_models


def save_best_model(results, trained_models, X_train):
    """Save the best performing model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Pick the model with the lowest MAE
    best = min(results, key=lambda x: x['mae'])
    best_name = best['name']
    best_model = trained_models[best_name]

    print(f"\n=== Best Model: {best_name} (MAE: {best['mae']}) ===")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"  Model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_name': best_name,
        'target': TARGET_COL,
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
    print(f"  All results saved to {results_path}")

    return best


def print_comparison_table(results):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 60)
    print("          BASELINE MODEL COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x['mae']):
        print(f"  {r['name']:<25} {r['mae']:>8.4f} {r['rmse']:>8.4f} {r['r2']:>8.4f}")
    print("=" * 60)


def save_history_to_csv(results, target_col):
    """Save training results to a history CSV file."""
    history_file = os.path.join(MODELS_DIR, 'model_history.csv')
    timestamp = datetime.now().isoformat()
    
    rows = []
    for r in results:
        row = {
            'timestamp': timestamp,
            'model_name': r['name'],
            'target': target_col,
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


def main():
    df = load_data()
    X_train, y_train, X_test, y_test = chronological_split(df)
    results, trained_models = train_models(X_train, y_train, X_test, y_test)
    print_comparison_table(results)
    best = save_best_model(results, trained_models, X_train)
    save_history_to_csv(results, TARGET_COL)
    print(f"\n[DONE] Training pipeline complete. Best model: {best['name']}")


if __name__ == "__main__":
    main()
