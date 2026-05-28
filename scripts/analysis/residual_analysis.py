import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

from scripts.pipelines.training_pipeline import load_data, chronological_split

def perform_residual_analysis():
    print("=== Phase 5 Step 4: Residual Analysis ===")
    
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
    if not os.path.exists(model_path):
        print("Best model not found!")
        return
        
    print("Loading data and model...")
    df = load_data()
    X_train, y_train, X_test, y_test = chronological_split(df)
    model = joblib.load(model_path)
    
    print("Predicting on Test Set...")
    y_pred = model.predict(X_test)
    
    targets = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
    
    out_dir = os.path.join(PROJECT_ROOT, 'visualizations')
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    for i, target in enumerate(targets):
        actual = y_test.iloc[:, i].values
        pred = y_pred[:, i]
        residuals = actual - pred
        
        # Simple scatter vs actual
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=actual, y=residuals, alpha=0.3)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f"{target} Residuals vs. Actual Values")
        plt.xlabel(f"Actual {target} Concentration")
        plt.ylabel("Residual (Actual - Predicted)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'residual_vs_actual_{target}.png'))
        plt.close()
        
        # Extreme errors
        extreme_under = np.sum(residuals > 20)
        extreme_over = np.sum(residuals < -20)
        
        print(f"\n--- {target.upper()} ---")
        print(f"  Severe Under-predictions (Actual > Predicted + 20): {extreme_under} times")
        print(f"  Severe Over-predictions (Actual < Predicted - 20): {extreme_over} times")
        
    print(f"\n[SUCCESS] Residual Analysis complete for all pollutants! Plots saved to {out_dir}")
    
if __name__ == "__main__":
    perform_residual_analysis()
