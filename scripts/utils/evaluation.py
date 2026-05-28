import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .aqi_calculator import get_overall_aqi

def evaluate_multioutput_model(name, model, X_test, y_test, target_cols):
    """
    Evaluate a multi-output model.
    Calculates per-pollutant metrics and also calculates the true US EPA AQI error.
    """
    y_pred = model.predict(X_test)
    
    # Per-target metrics
    maes = []
    rmses = []
    r2s = []
    
    print(f"\n  [{name}] Per-Pollutant Evaluation:")
    for i, target in enumerate(target_cols):
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
    
    # Calculate True US EPA AQI Error
    aqi_true = []
    aqi_pred = []
    
    for i in range(len(y_test)):
        # y_test columns expected: pm2_5, pm10, no2, so2, co
        t_pm25, t_pm10, t_no2, t_so2, t_co = y_test.iloc[i]
        p_pm25, p_pm10, p_no2, p_so2, p_co = y_pred[i]
        
        true_aqi = get_overall_aqi(t_pm25, t_pm10, t_no2, t_so2, t_co)
        pred_aqi = get_overall_aqi(p_pm25, p_pm10, p_no2, p_so2, p_co)
        
        aqi_true.append(true_aqi)
        aqi_pred.append(pred_aqi)
        
    aqi_mae = mean_absolute_error(aqi_true, aqi_pred)
    
    print(f"    -> AVERAGE SCORE       : MAE={avg_mae:6.2f} | R2={avg_r2:6.4f}")
    print(f"    -> TRUE US EPA AQI MAE : {aqi_mae:6.2f} index points")

    return {
        'name': name,
        'mae': round(avg_mae, 4),
        'rmse': round(avg_rmse, 4),
        'r2': round(avg_r2, 4),
        'aqi_mae': round(aqi_mae, 4)
    }

def print_comparison_table(results):
    print("\n" + "=" * 75)
    print("      MULTI-OUTPUT BASELINE MODEL COMPARISON")
    print("=" * 75)
    print(f"  {'Model':<25} {'Avg MAE':>8} {'Avg RMSE':>10} {'Avg R2':>8} | {'AQI MAE':>8}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x['mae']):
        print(f"  {r['name']:<25} {r['mae']:>8.4f} {r['rmse']:>10.4f} {r['r2']:>8.4f} | {r['aqi_mae']:>8.4f}")
    print("=" * 75)

def save_history_to_csv(results, models_dir, experiment_note="No notes"):
    history_file = os.path.join(models_dir, 'model_history.csv')
    timestamp = datetime.now().isoformat()
    
    rows = []
    for r in results:
        row = {
            'timestamp': timestamp,
            'model_name': r['name'],
            'target': "MULTI_OUTPUT",
            'mae': r['mae'],
            'rmse': r['rmse'],
            'r2': r['r2'],
            'aqi_mae': r.get('aqi_mae', np.nan),
            'experiment_note': experiment_note
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
