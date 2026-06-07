import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PLOTS_DIR    = os.path.join(PROJECT_ROOT, 'eda_plots_extra')
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

AQI_COLORS = {
    'Good': '#00e400',
    'Moderate': '#ffff00',
    'Unhealthy (Sensitive)': '#ff7e00',
    'Unhealthy': '#ff0000',
    'Very Unhealthy': '#8f3f97',
    'Hazardous': '#7e0023',
}

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

def run_extra_eda():
    print("Connecting to Hopsworks Feature Store...")
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_weather_features", version=1)
    df = fg.read()
    
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df.sort_values('time').reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Calculate AQI
    df['aqi'] = df['pm2_5'].apply(pm25_to_aqi)
    df['aqi_cat'] = df['aqi'].apply(aqi_category)
    
    # 1. AQI Category Pie Chart
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
    ax.set_title("01. US EPA AQI Category Distribution", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_aqi_category_pie.png"))
    plt.close(fig)
    print("Saved -> eda_plots_extra/01_aqi_category_pie.png")
    
    # 2. Hour vs Day of Week Heatmap for AQI
    pivot_hw = df.pivot_table(values='aqi', index='hour', columns='day_of_week', aggfunc='mean')
    pivot_hw.columns = DAY_NAMES
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_hw, ax=ax, cmap='YlOrRd', annot=False,
                cbar_kws={'label': 'Avg US EPA AQI'})
    ax.set_title("02. US EPA AQI Heatmap: Hour of Day vs Day of Week", fontsize=14, fontweight='bold')
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_heatmap_hour_vs_dow_aqi.png"))
    plt.close(fig)
    print("Saved -> eda_plots_extra/02_heatmap_hour_vs_dow_aqi.png")

if __name__ == "__main__":
    run_extra_eda()
