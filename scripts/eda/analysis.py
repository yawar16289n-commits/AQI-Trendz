"""
Pre-Training Analysis Script
Analyzes the baseline model results and data characteristics:
1. Target distribution (PM2.5)
2. Correlation check between features and target
3. Feature importance from the trained Random Forest model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'cleaned', 'featured_historical_data.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'eda_plots')

FEATURE_COLS = [
    'temperature_2m', 'relative_humidity_2m', 'precipitation',
    'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
    'hour', 'day', 'month', 'day_of_week', 'is_weekend',
]
TARGET_COL = 'pm2_5'


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'])

    # ── 1. Target Distribution ─────────────────────────────────────────────────
    print("\n=== 1. TARGET DISTRIBUTION (PM2.5) ===")
    print(df[TARGET_COL].describe().to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(df[TARGET_COL], bins=80, color='#e74c3c', edgecolor='black', alpha=0.7)
    axes[0].set_title('PM2.5 Distribution', fontsize=14)
    axes[0].set_xlabel('PM2.5 (ug/m3)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(df[TARGET_COL].mean(), color='blue', linestyle='--', label=f"Mean: {df[TARGET_COL].mean():.1f}")
    axes[0].axvline(df[TARGET_COL].median(), color='green', linestyle='--', label=f"Median: {df[TARGET_COL].median():.1f}")
    axes[0].legend()

    # Box plot
    axes[1].boxplot(df[TARGET_COL], vert=True)
    axes[1].set_title('PM2.5 Box Plot', fontsize=14)
    axes[1].set_ylabel('PM2.5 (ug/m3)')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'target_distribution.png'), dpi=200)
    plt.close()
    print("  -> Saved target_distribution.png")

    # ── 2. Correlation Check ───────────────────────────────────────────────────
    print("\n=== 2. CORRELATION WITH TARGET (PM2.5) ===")
    corr_with_target = df[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print(corr_with_target.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr_with_target.values]
    corr_with_target.plot(kind='barh', color=colors, ax=ax, edgecolor='black')
    ax.set_title('Feature Correlation with PM2.5', fontsize=14)
    ax.set_xlabel('Pearson Correlation')
    ax.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_with_target.png'), dpi=200)
    plt.close()
    print("  -> Saved correlation_with_target.png")

    # ── 3. Feature Importance (from Random Forest) ─────────────────────────────
    print("\n=== 3. FEATURE IMPORTANCE (Random Forest) ===")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True)

        print(feat_imp.sort_values(ascending=False).to_string())

        fig, ax = plt.subplots(figsize=(10, 6))
        feat_imp.plot(kind='barh', color='#3498db', edgecolor='black', ax=ax)
        ax.set_title('Random Forest Feature Importance (Baseline)', fontsize=14)
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'feature_importance_baseline.png'), dpi=200)
        plt.close()
        print("  -> Saved feature_importance_baseline.png")
    else:
        print("  [SKIP] No trained model found at", MODEL_PATH)

    # ── 4. Autocorrelation of PM2.5 ───────────────────────────────────────────
    print("\n=== 4. PM2.5 AUTOCORRELATION ===")
    autocorr_vals = [df[TARGET_COL].autocorr(lag=lag) for lag in [1, 2, 3, 6, 12, 24, 48, 72]]
    lag_labels = [1, 2, 3, 6, 12, 24, 48, 72]
    print("  Lag (hours) -> Autocorrelation")
    for lag, ac in zip(lag_labels, autocorr_vals):
        print(f"    {lag:>3}h  ->  {ac:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(l) + 'h' for l in lag_labels], autocorr_vals, color='#9b59b6', edgecolor='black')
    ax.set_title('PM2.5 Autocorrelation at Different Lags', fontsize=14)
    ax.set_xlabel('Lag (hours)')
    ax.set_ylabel('Autocorrelation')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'pm25_autocorrelation.png'), dpi=200)
    plt.close()
    print("  -> Saved pm25_autocorrelation.png")

    print("\n[DONE] All analysis plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
