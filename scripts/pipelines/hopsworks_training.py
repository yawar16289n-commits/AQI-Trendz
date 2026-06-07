import os
import sys
import pandas as pd
import hopsworks
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

def train_and_upload_model():
    print("=== Training Model from Hopsworks & Uploading to Registry ===")
    
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        print("ERROR: HOPSWORKS_API_KEY not found.")
        sys.exit(1)
        
    print("Logging into Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # 1. Fetch Data from Feature Store
    print("Fetching 'aqi_weather_features' from Feature Store...")
    try:
        fg = fs.get_feature_group(name="aqi_weather_features", version=1)
        df = fg.read()
        print(f"Successfully downloaded {len(df)} rows.")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return
        
    # Sort chronologically just in case
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 2. Prepare Training Data
    targets = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
    features = [c for c in df.columns if c not in targets and c != 'time']
    
    X = df[features]
    y = df[targets]
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # NOTE: RF uses controlled params to balance model file size vs accuracy.
    # n_estimators=60 + max_depth=15 + min_samples_leaf=2 gives strong R2
    # without the file ballooning to 1GB+ (compress=3 keeps it ~20-25MB).
    # max_features is left at default (1.0 = all features) — 'sqrt' hurts regression R2.
    models_to_train = {
        "xgboost_multi_aqi": MultiOutputRegressor(XGBRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1
        )),
        "rf_multi_aqi": MultiOutputRegressor(RandomForestRegressor(
            n_estimators=60, max_depth=15, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
        "lr_multi_aqi": MultiOutputRegressor(LinearRegression(n_jobs=-1))
    }
    
    mr = project.get_model_registry()
    
    for model_name, model in models_to_train.items():
        print(f"\n--- Training {model_name} ---")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{model_name} Performance: MAE = {mae:.4f}, R2 = {r2:.4f}")
        
        print(f"Retraining {model_name} on 100% of data for production deployment...")
        model.fit(X, y)
        
        # Save locally in a specific directory for this model
        model_dir = os.path.join(PROJECT_ROOT, 'models', 'hopsworks', model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{model_name}.pkl')
        
        print(f"Saving {model_name} locally (compressed)...")
        joblib.dump(model, model_path, compress=3)  # compress=3 significantly reduces file size
        
        print(f"Uploading {model_name} to Hopsworks Model Registry...")
        hw_model = mr.python.create_model(
            name=model_name,
            metrics={"mae": mae, "r2": r2},
            description=f"MultiOutputRegressor for 5 AQI Pollutants ({model_name})",
            input_example=X_train.sample(1)
        )
        hw_model.save(model_dir)
        print(f"[SUCCESS] {model_name} successfully trained and registered!")
        
    print("\n[SUCCESS] All models trained and uploaded.")

if __name__ == "__main__":
    train_and_upload_model()
