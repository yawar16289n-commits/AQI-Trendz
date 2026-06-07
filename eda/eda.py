"""
eda.py
-----------
Advanced Multi-Pollutant EDA for AQI-Trendz.
Generates 8 key exploratory data analysis visualizations:
6 core pollutant trend charts + 2 AQI distribution charts.

Data Source: Hopsworks Feature Store (aqi_weather_features v1)
Output:      eda/plots/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
PLOTS_DIR    = os.path.join(SCRIPT_DIR, 'plots')
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#1a1a2e",
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "grid.color": "#2d3561",
    "grid.alpha": 0.5,
})

POLLUTANTS = ['pm2_5', 'pm10', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
COLORS = ['#e94560', '#0f3460', '#533483', '#e2b96f', '#16c79a']
POLL_COLORS = dict(zip(POLLUTANTS, COLORS))
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# WHO 24-hour guidelines (µg/m³) used as a baseline for "danger" levels
WHO_LIMITS = {
    'pm2_5': 15,
    'pm10': 45,
    'nitrogen_dioxide': 25,
    'sulphur_dioxide': 40,
    'carbon_monoxide': 4000
}

AQI_COLORS = {
    'Good': '#00e400',
    'Moderate': '#ffff00',
    'Unhealthy (Sensitive)': '#ff7e00',
    'Unhealthy': '#ff0000',
    'Very Unhealthy': '#8f3f97',
    'Hazardous': '#7e0023',
}

def save(fig, filename):
    fig.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close(fig)
    print(f"  Saved → eda/plots/{filename}")

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

def aqi_category(aqi):
    if aqi <= 50: return 'Good'
    if aqi <= 100: return 'Moderate'
    if aqi <= 150: return 'Unhealthy (Sensitive)'
    if aqi <= 200: return 'Unhealthy'
    if aqi <= 300: return 'Very Unhealthy'
    return 'Hazardous'

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    print("Connecting to Hopsworks Feature Store...")
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_weather_features", version=1)
    df = fg.read()
    
    # Keep only base columns
    cols = ['time'] + POLLUTANTS + ['temperature_2m', 'relative_humidity_2m', 
                                    'precipitation', 'surface_pressure', 
                                    'wind_speed_10m', 'wind_direction_10m']
    df = df[[c for c in cols if c in df.columns]].copy()
    
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    df['hour']       = df['time'].dt.hour
    df['month']      = df['time'].dt.month
    df['day_of_week'] = df['time'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Calculate AQI
    df['aqi'] = df['pm2_5'].apply(pm25_to_aqi)
    df['aqi_cat'] = df['aqi'].apply(aqi_category)
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# EXPLORATIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_eda(df):
    print("\nGenerating 8 EDA Visualizations...")

    # 1. The "Leading Pollutant" Analysis (Ratio to WHO Limits)
    for p in POLLUTANTS:
        df[f'{p}_who_ratio'] = df[p] / WHO_LIMITS[p]
    ratio_cols = [f'{p}_who_ratio' for p in POLLUTANTS]
    df['leading_pollutant'] = df[ratio_cols].idxmax(axis=1).str.replace('_who_ratio', '')
    
    lead_counts = df['leading_pollutant'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(lead_counts.index, lead_counts.values, color=[POLL_COLORS[p] for p in lead_counts.index])
    ax.set_title("01. The 'Leading Pollutant'\n(Which pollutant exceeds WHO limits the most?)", fontweight='bold')
    ax.set_ylabel("Number of Hours as the Leading Threat")
    plt.xticks(rotation=15)
    save(fig, "01_leading_pollutant.png")

    # 2. Pollutant Co-occurrence (Traffic vs Dust)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(data=df.sample(2000), x='carbon_monoxide', y='nitrogen_dioxide', 
                    alpha=0.5, color='#e2b96f', ax=axes[0])
    axes[0].set_title("Traffic Signature: NO₂ vs CO", fontweight='bold')
    sns.scatterplot(data=df.sample(2000), x='pm10', y='pm2_5', 
                    alpha=0.5, color='#e94560', ax=axes[1])
    axes[1].set_title("Particulate Signature: PM2.5 vs PM10", fontweight='bold')
    fig.suptitle("02. Pollutant Co-occurrence Signatures", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, "02_co_occurrence.png")

    # 3. Wind Source Profiling per Pollutant
    if 'wind_direction_10m' in df.columns:
        df['wind_dir_bin'] = (df['wind_direction_10m'] // 22.5).astype(int) % 16
        dir_labels = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        angles_plot = np.append(angles, angles[0])
        
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), subplot_kw=dict(polar=True))
        for i, p in enumerate(POLLUTANTS):
            dir_agg = df.groupby('wind_dir_bin')[p].mean()
            v_min, v_max = dir_agg.min(), dir_agg.max()
            norm_agg = (dir_agg - v_min) / (v_max - v_min) if v_max > v_min else dir_agg
            values = [norm_agg.get(j, 0) for j in range(16)]
            values.append(values[0])
            axes[i].fill(angles_plot, values, alpha=0.4, color=COLORS[i])
            axes[i].plot(angles_plot, values, color=COLORS[i], lw=2)
            axes[i].set_xticks(angles)
            axes[i].set_xticklabels(dir_labels, fontsize=8)
            axes[i].set_title(p.upper(), fontweight='bold', pad=15)
            axes[i].set_yticks([])
            
        fig.suptitle("03. Wind Source Profiles (Normalized Shapes)\nWhere does each pollutant blow in from?", 
                     fontsize=14, fontweight='bold', y=1.05)
        fig.tight_layout()
        save(fig, "03_wind_profiles.png")

    # 4. The Weekend/Weekday Pollutant Shift
    wknd_shift = []
    for p in POLLUTANTS:
        weekday_avg = df[df['is_weekend'] == 0][p].mean()
        weekend_avg = df[df['is_weekend'] == 1][p].mean()
        pct_change = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        wknd_shift.append({'Pollutant': p, 'Weekend Shift %': pct_change})
        
    shift_df = pd.DataFrame(wknd_shift)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=shift_df, x='Pollutant', y='Weekend Shift %', 
                palette=COLORS, ax=ax, edgecolor='#333')
    ax.axhline(0, color='white', lw=1.5)
    ax.set_title("04. Weekend vs Weekday Shift\n(Negative % means it drops on weekends)", fontweight='bold')
    ax.set_ylabel("% Change on Weekends")
    fig.tight_layout()
    save(fig, "04_weekend_shift.png")

    # 5. Weather Interaction Profiles (Correlation with Weather)
    weather_vars = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'surface_pressure']
    weather_corr = df[POLLUTANTS + weather_vars].corr().loc[POLLUTANTS, weather_vars]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(weather_corr, annot=True, cmap='PRGn', center=0, fmt='.2f', 
                linewidths=1, ax=ax, cbar_kws={'label': 'Pearson Correlation'})
    ax.set_title("05. How Weather Impacts Each Pollutant", fontweight='bold')
    fig.tight_layout()
    save(fig, "05_weather_impacts.png")

    # 6. Seasonal Ratio Shifts (PM2.5 / PM10)
    df['pm_ratio'] = df['pm2_5'] / df['pm10']
    monthly_ratio = df.groupby('month')['pm_ratio'].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(MONTH_NAMES, monthly_ratio.values, marker='o', lw=3, color='#e94560')
    ax.axvspan('Jul', 'Sep', alpha=0.2, color='#0f3460', label='Monsoon Season')
    ax.set_title("06. PM2.5 to PM10 Ratio over the Year\n(Higher = More fine toxic smog, Lower = More coarse dust)", fontweight='bold')
    ax.set_ylabel("Ratio (PM2.5 / PM10)")
    ax.legend()
    fig.tight_layout()
    save(fig, "06_pm_ratio_seasonality.png")

    # 7. AQI Category Pie Chart
    cat_counts = df['aqi_cat'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    wedge_colors = [AQI_COLORS.get(c, '#888') for c in cat_counts.index]
    wedges, texts, autotexts = ax.pie(
        cat_counts.values, labels=cat_counts.index,
        colors=wedge_colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.82,
        textprops={'color': 'white', 'fontsize': 11}
    )
    for a in autotexts:
        a.set_fontweight('bold')
    ax.set_title("07. US EPA AQI Category Distribution", fontsize=14, fontweight='bold')
    fig.tight_layout()
    save(fig, "01_aqi_category_pie.png")
    
    # 8. Hour vs Day of Week Heatmap for AQI
    pivot_hw = df.pivot_table(values='aqi', index='hour', columns='day_of_week', aggfunc='mean')
    pivot_hw.columns = DAY_NAMES
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_hw, ax=ax, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Avg US EPA AQI'})
    ax.set_title("08. US EPA AQI Heatmap: Hour of Day vs Day of Week", fontsize=14, fontweight='bold')
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    fig.tight_layout()
    save(fig, "02_heatmap_hour_vs_dow_aqi.png")

if __name__ == "__main__":
    print("-" * 65)
    print("  AQI-Trendz: Advanced Multi-Pollutant EDA")
    print("-" * 65)
    df = load_data()
    run_eda(df)
    print("\n" + "-" * 65)
    print("  EDA Complete! All plots saved directly to eda/plots/")
    print("-" * 65)
