import os
import sys
import pandas as pd
import hopsworks
import joblib
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
    
    # 3. Train Model
    print("Training Multi-Output XGBoost model...")
    base_xgb = XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        n_jobs=-1
    )
    model = MultiOutputRegressor(base_xgb)
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance: MAE = {mae:.4f}, R2 = {r2:.4f}")
    
    # 5. Save and Upload to Model Registry
    model_dir = os.path.join(PROJECT_ROOT, 'models', 'hopsworks')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost_multi_aqi.pkl')
    
    print("Saving model locally...")
    joblib.dump(model, model_path)
    
    print("Uploading to Hopsworks Model Registry...")
    mr = project.get_model_registry()
    
    hw_model = mr.python.create_model(
        name="xgboost_multi_aqi",
        metrics={"mae": mae, "r2": r2},
        description="XGBoost MultiOutputRegressor for 5 AQI Pollutants",
        input_example=X_train.sample(1)
    )
    
    hw_model.save(model_dir)
    print("[SUCCESS] Model successfully trained and registered in Hopsworks!")

if __name__ == "__main__":
    train_and_upload_model()
