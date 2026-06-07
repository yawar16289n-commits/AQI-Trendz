import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import hopsworks
import openmeteo_requests
import requests_cache
from retry_requests import retry
from dotenv import load_dotenv

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AQI-Trendz Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, '.env'))

# Custom CSS
st.markdown("""
<style>
/* Typography & Custom Font styling */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.95);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Glassmorphism containers */
.glass-panel {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

.hero-banner {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 35px;
    margin-bottom: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
}

.hero-banner h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 8px;
    background: linear-gradient(to right, #60a5fa, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-banner p {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 900px;
}

/* Custom Grid Metrics */
.metric-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin-bottom: 25px;
}

.metric-card {
    background: rgba(30, 41, 59, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 22px rgba(0,0,0,0.15);
}

.metric-card .label {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.metric-card .value {
    font-size: 28px;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
    color: #f8fafc;
    margin-top: 8px;
}

.metric-card .unit {
    font-size: 12px;
    color: #64748b;
    font-weight: 500;
}

/* Alert Banner */
.alert-banner {
    background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(15, 23, 42, 0.25) 100%);
    border: 1px solid rgba(220, 38, 38, 0.3);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 15px;
}
.alert-title {
    color: #fca5a5;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    font-size: 16px;
    margin-bottom: 2px;
}
.alert-desc {
    color: #f8fafc;
    font-size: 13.5px;
}

/* Badge Styling mapping old to new */
.source-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; margin-left: 12px;
    vertical-align: middle;
}
.badge-live   { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
.badge-local  { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }

</style>
""", unsafe_allow_html=True)

# --- PATHS ---
PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, 'data', 'cleaned', 'predictions.csv')
HISTORY_PATH     = os.path.join(SCRIPT_DIR, 'models', 'model_history.csv')
LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'best_model.pkl')
FEATURED_DATA    = os.path.join(SCRIPT_DIR, 'assets', 'shap_background.csv')

# --- HELPERS ---
def get_aqi_status(aqi):
    if aqi <= 50:  return "Good",                           "status-good"
    if aqi <= 100: return "Moderate",                       "status-mod"
    if aqi <= 150: return "Unhealthy for Sensitive Groups", "status-sens"
    return              "Unhealthy",                        "status-bad"

# --- DATA LOADERS ---
@st.cache_data(ttl=900)  # refresh every 15 minutes
def load_current_actuals_openmeteo():
    """Fetch the real, live current AQI and weather from Open-Meteo."""
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)
        
        # AQI Fetch
        aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aqi_params = {
            "latitude": 24.933,
            "longitude": 67.033,
            "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "us_aqi"],
        }
        resp = openmeteo.weather_api(aqi_url, params=aqi_params)[0]
        current_aqi = resp.Current()
        
        # Weather Fetch
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": 24.933,
            "longitude": 67.033,
            "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
        }
        resp_w = openmeteo.weather_api(weather_url, params=weather_params)[0]
        current_w = resp_w.Current()
        
        return {
            "pm10": current_aqi.Variables(0).Value(),
            "pm2_5": current_aqi.Variables(1).Value(),
            "carbon_monoxide": current_aqi.Variables(2).Value(),
            "nitrogen_dioxide": current_aqi.Variables(3).Value(),
            "sulphur_dioxide": current_aqi.Variables(4).Value(),
            "us_aqi": current_aqi.Variables(5).Value(),
            "temperature": current_w.Variables(0).Value(),
            "humidity": current_w.Variables(1).Value(),
            "wind_speed": current_w.Variables(2).Value(),
            "time": pd.to_datetime(current_aqi.Time(), unit="s", utc=True)
        }
    except Exception as e:
        return None
@st.cache_data(ttl=3600)   # refresh cache every hour
def load_predictions_hopsworks():
    """Try to load predictions from Hopsworks; fall back to local CSV."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        return None, "no_key"
    try:
        project = hopsworks.login()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="aqi_predictions", version=2)
        df = fg.read()
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)
        # Keep only future rows
        now = pd.Timestamp.utcnow().tz_localize(None)
        df['time_naive'] = df['time'].dt.tz_localize(None)
        df = df[df['time_naive'] >= now].drop(columns=['time_naive'])
        return df, "hopsworks"
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_predictions_local():
    if os.path.exists(PREDICTIONS_PATH):
        df = pd.read_csv(PREDICTIONS_PATH)
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_data
@st.cache_data
def load_history():
    return None

@st.cache_resource
def load_model_and_data():
    """Load model and featured data — tries Hopsworks Registry first, falls back to local."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    model = None
    best_model_name = "Unknown Model"
    best_r2 = -float('inf')
    best_mae = float('nan')
    
    if api_key:
        try:
            project = hopsworks.login()
            mr = project.get_model_registry()
            
            # Find the absolute best model dynamically
            model_names = ["xgboost_multi_aqi", "rf_multi_aqi", "lr_multi_aqi"]
            best_hw_model = None

            for name in model_names:
                try:
                    hw_model = mr.get_best_model(name=name, metric="r2", direction="max")
                    r2 = hw_model.training_metrics.get('r2', -float('inf'))
                    if r2 > best_r2:
                        best_r2 = r2
                        best_mae = hw_model.training_metrics.get('mae', float('nan'))
                        best_hw_model = hw_model
                        best_model_name = name
                except Exception:
                    continue
                    
            if best_hw_model:
                model_dir = best_hw_model.download()
                model_path = os.path.join(model_dir, f"{best_model_name}.pkl")
                model = joblib.load(model_path)
        except Exception:
            model = None

    if model is None and os.path.exists(LOCAL_MODEL_PATH):
        model = joblib.load(LOCAL_MODEL_PATH)
        best_model_name = "Local Model"
        best_r2 = float('nan')
        best_mae = float('nan')

    # Fetch SHAP background data dynamically from Hopsworks to avoid CSV files
    df = None
    if api_key:
        try:
            project = hopsworks.login()
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_weather_features", version=1)
            raw_df = fg.read()
            raw_df['time'] = pd.to_datetime(raw_df['time'], utc=True)
            raw_df = raw_df.sort_values('time')
            # Only keep the last 150 rows to keep it lightning fast
            df = raw_df.tail(150).reset_index(drop=True)
        except Exception:
            df = None

    return model, df, best_model_name, best_r2, best_mae

# --- SIDEBAR ---
st.sidebar.markdown("## 🌍 AQI-Trendz")
st.sidebar.markdown("Advanced Multi-Pollutant Forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["3-Day Forecast", "Model Diagnostics", "Feature Explainability (SHAP)"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by AI · Hopsworks · Open-Meteo")

# ── Load predictions (Hopsworks → local fallback) ─────────────────────────────
preds_df = load_predictions_local()
data_source = "local"
source_label = "local"
if preds_df is None:
    preds_df, data_source = load_predictions_hopsworks()
    source_label = "live"

# =============================================================================
# PAGE: 3-DAY FORECAST
# =============================================================================
def aqi_details(aqi):
    if aqi <= 50:
        return "Good", "#10b981", "rgba(16, 185, 129, 0.12)", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "#fbbf24", "rgba(245, 158, 11, 0.12)", "Air quality is acceptable; however, there may be concern for sensitive people."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#fb923c", "rgba(249, 115, 22, 0.12)", "Members of sensitive groups may experience health effects. The public is less affected."
    elif aqi <= 200:
        return "Unhealthy", "#ef4444", "rgba(239, 68, 68, 0.12)", "Everyone may begin to experience health effects; sensitive groups feel serious impacts."
    elif aqi <= 300:
        return "Very Unhealthy", "#c084fc", "rgba(139, 92, 246, 0.12)", "Health alert: The risk of health effects is significantly increased for everyone."
    else:
        return "Hazardous", "#f87171", "rgba(127, 29, 29, 0.25)", "Health warning of emergency conditions: The entire population is likely to be affected."

if page == "3-Day Forecast":
    badge = (
        '<span class="source-badge badge-live">\u2601 Live from Hopsworks</span>'
        if source_label == "live"
        else '<span class="source-badge badge-local">\U0001f4c1 Local cache</span>'
    )
    st.markdown(f"""
    <div class="hero-banner">
        <h1>Karachi AQI Dashboard {badge}</h1>
        <p>Live Air Quality Index and 3-Day Multi-Pollutant Forecast powered by Best Model and Hopsworks.</p>
    </div>
    """, unsafe_allow_html=True)

    actuals = load_current_actuals_openmeteo()

    if actuals is not None:
        aqi_val  = float(actuals.get('us_aqi', 0))
        temp     = float(actuals.get('temperature', 0))
        hum      = float(actuals.get('humidity', 0))
        wind     = float(actuals.get('wind_speed', 0))
        ts_str   = pd.to_datetime(actuals.get('time', pd.Timestamp.utcnow())).strftime('%d %b %Y, %I:%M %p UTC')
        cat, clr, bg, desc = aqi_details(aqi_val)

        if aqi_val > 100:
            st.markdown(f"""
            <div class="alert-banner">
                <div style="font-size:24px;">&#9888;&#65039;</div>
                <div>
                    <div class="alert-title">Air Quality Alert: AQI is {aqi_val:.0f} &#8212; {cat}</div>
                    <div class="alert-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="glass-panel">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px; flex-wrap:wrap; gap:10px;">
                <h3 style="color:#f8fafc; margin:0; font-size:1.1rem;">&#127777;&#65039; Live Conditions &mdash; Karachi
                    <span style="font-size:11px; color:#64748b; font-weight:400; margin-left:8px;">Updated {ts_str}</span>
                </h3>
                <div style="display:flex; gap:25px; align-items:center;">
                    <div style="text-align:center;">
                        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Temp</div>
                        <div style="font-size:22px; font-weight:800; color:#f8fafc;">{temp:.1f}&#176;C</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Humidity</div>
                        <div style="font-size:22px; font-weight:800; color:#f8fafc;">{hum:.0f}%</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px;">Wind</div>
                        <div style="font-size:22px; font-weight:800; color:#f8fafc;">{wind:.1f} km/h</div>
                    </div>
                </div>
            </div>
            <div style="display:flex; align-items:baseline; gap:15px; margin-bottom:20px;">
                <span style="font-size:4rem; font-weight:800; color:{clr}; line-height:1;">{aqi_val:.0f}</span>
                <div>
                    <span style="font-size:1rem; font-weight:700; padding:5px 14px; border-radius:20px; background:{bg}; color:{clr}; border:1px solid {clr}44;">{cat}</span>
                    <p style="color:#94a3b8; font-size:0.85rem; margin-top:8px; margin-bottom:0;">{desc}</p>
                </div>
            </div>
            <div class="metric-container">
                <div class="metric-card"><div class="label">PM 2.5</div><div class="value">{actuals.get('pm2_5', 0):.1f}</div><div class="unit">&#956;g/m&#179;</div></div>
                <div class="metric-card"><div class="label">PM 10</div><div class="value">{actuals.get('pm10', 0):.1f}</div><div class="unit">&#956;g/m&#179;</div></div>
                <div class="metric-card"><div class="label">NO&#8322;</div><div class="value">{actuals.get('nitrogen_dioxide', 0):.1f}</div><div class="unit">&#956;g/m&#179;</div></div>
                <div class="metric-card"><div class="label">SO&#8322;</div><div class="value">{actuals.get('sulphur_dioxide', 0):.1f}</div><div class="unit">&#956;g/m&#179;</div></div>
                <div class="metric-card"><div class="label">CO</div><div class="value">{actuals.get('carbon_monoxide', 0):.1f}</div><div class="unit">&#956;g/m&#179;</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── 3-DAY FORECAST CARDS ──────────────────────────────────────────────────
    if preds_df is not None and not preds_df.empty:
        preds_df['date'] = preds_df['time'].dt.date
        aqi_col = 'US_EPA_AQI' if 'US_EPA_AQI' in preds_df.columns else 'us_epa_aqi'
        daily_p = preds_df.groupby('date')[aqi_col].mean().reset_index().head(3)
        day_labels = ['Tomorrow', 'Day 2', 'Day 3']

        st.markdown('<h3 style="color:#f8fafc; margin:30px 0 15px;">&#128302; 3-Day AQI Forecast</h3>', unsafe_allow_html=True)
        cols = st.columns(len(daily_p))
        for i, (_, row) in enumerate(daily_p.iterrows()):
            aqi_v = float(row[aqi_col])
            c, clr, bg, desc = aqi_details(aqi_v)
            lbl      = day_labels[i] if i < len(day_labels) else f"Day {i+1}"
            date_str = pd.Timestamp(row['date']).strftime('%A, %d %b')
            with cols[i]:
                st.markdown(f"""
                <div style="background:linear-gradient(to bottom,{bg},rgba(30,41,59,0.45));
                            border:1px solid {clr}33; border-left:5px solid {clr};
                            border-radius:20px; padding:22px 20px; height:100%;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                        <span style="font-size:15px;font-weight:700;font-family:'Outfit';color:#f8fafc;">{lbl}</span>
                        <span style="font-size:12px;color:#94a3b8;">{date_str}</span>
                    </div>
                    <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:10px;">
                        <span style="font-size:44px;font-weight:800;font-family:'Outfit';color:{clr};line-height:1;">{aqi_v:.0f}</span>
                        <span style="font-size:12px;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">AQI</span>
                    </div>
                    <span style="font-size:11px;font-weight:700;padding:5px 10px;border-radius:20px;
                                text-transform:uppercase;letter-spacing:0.5px;
                                color:{clr};background:{bg};border:1px solid {clr}44;">{c}</span>
                    <p style="color:#94a3b8;font-size:12.5px;margin-top:14px;line-height:1.5;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No predictions available. Run the predict_pipeline.py first or wait for the hourly GitHub Action to execute.")

    # ── LAST 24H AQI CHART ─────────────────────────────────────────────────────
    try:
        from datetime import datetime, timezone, timedelta
        cache_s  = requests_cache.CachedSession('.cache', expire_after=1800)
        retry_s  = retry(cache_s, retries=5, backoff_factor=0.2)
        om_cli   = openmeteo_requests.Client(session=retry_s)
        now_utc  = pd.Timestamp.now(tz='UTC')
        # Fetch yesterday + today to guarantee a full 24h window
        start_str = (now_utc - pd.Timedelta(hours=24)).strftime('%Y-%m-%d')
        end_str   = now_utc.strftime('%Y-%m-%d')
        hr = om_cli.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            "latitude": 24.933, "longitude": 67.033,
            "start_date": start_str, "end_date": end_str,
            "hourly": "us_aqi"
        })[0].Hourly()
        hist24 = pd.DataFrame({
            "time": pd.date_range(
                start=pd.to_datetime(hr.Time(), unit="s", utc=True),
                end=pd.to_datetime(hr.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hr.Interval()), inclusive="left"
            ),
            "aqi": hr.Variables(0).ValuesAsNumpy()
        })
        hist24['aqi'] = hist24['aqi'].round().astype(int)
        # Keep only the last 24 hours up to now
        cutoff = now_utc - pd.Timedelta(hours=24)
        hist24 = hist24[(hist24['time'] >= cutoff) & (hist24['time'] <= now_utc)].dropna()

        if not hist24.empty:
            st.markdown('<h3 style="color:#f8fafc; margin:25px 0 10px;">&#128200; AQI &mdash; Last 24 Hours</h3>', unsafe_allow_html=True)
            fig24 = go.Figure()

            # Only draw bands for the zone(s) the data actually sits in
            aqi_min = hist24['aqi'].min()
            aqi_max = hist24['aqi'].max()
            bands = [
                (0,   50,  "#10b981"),  # Good
                (50,  100, "#fbbf24"),  # Moderate
                (100, 150, "#fb923c"),  # Unhealthy for Sensitive Groups
                (150, 200, "#ef4444"),  # Unhealthy
                (200, 300, "#a855f7"),  # Very Unhealthy
                (300, 500, "#7f1d1d"),  # Hazardous
            ]
            for b_lo, b_hi, b_col in bands:
                if aqi_max >= b_lo and aqi_min <= b_hi:
                    fig24.add_hrect(
                        y0=max(b_lo, aqi_min - 5),
                        y1=min(b_hi, aqi_max + 5),
                        line_width=0, fillcolor=b_col, opacity=0.12
                    )

            fig24.add_trace(go.Scatter(
                x=hist24['time'], y=hist24['aqi'],
                mode='lines+markers', name='AQI',
                line=dict(width=3, color='#60a5fa', shape='spline'),
                marker=dict(size=6, color='#60a5fa'),
                fill='tozeroy', fillcolor='rgba(96,165,250,0.08)'
            ))
            fig24.update_layout(
                template="plotly_dark", height=280,
                xaxis_title="Hour (UTC)", yaxis_title="US EPA AQI",
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,33,62,0.6)',
                yaxis=dict(range=[max(0, aqi_min - 10), aqi_max + 10])
            )
            st.plotly_chart(fig24, use_container_width=True)
    except Exception as e:
        st.info(f"Could not load today's AQI history: {e}")



# =============================================================================
# PAGE: MODEL DIAGNOSTICS
# =============================================================================
elif page == "Model Diagnostics":
    st.markdown("""
    <div class="hero-banner">
        <h1>&#128200; Model Diagnostics</h1>
        <p>Track how the XGBoost, Random Forest and Linear Regression models have performed across training runs.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── HOW THE MODEL WORKS explanation cards ──
    st.markdown('<h3 style="color:#f8fafc; margin-bottom:15px;">&#129504; How the Prediction Works</h3>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="glass-panel">
            <div style="font-size:2rem; margin-bottom:10px;">&#127760;</div>
            <h4 style="color:#60a5fa; margin-bottom:8px; font-size:1.1rem;">Step 1: Data Fetch</h4>
            <p style="color:#94a3b8; font-size:12.5px; line-height:1.5;">Fetches the last 7 days of actual AQI and the 3-day weather forecast from Open-Meteo's API.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-panel">
            <div style="font-size:2rem; margin-bottom:10px;">&#9881;&#65039;</div>
            <h4 style="color:#fbbf24; margin-bottom:8px; font-size:1.1rem;">Step 2: Feature Eng</h4>
            <p style="color:#94a3b8; font-size:12.5px; line-height:1.5;">Creates lag features (PM2.5 from 1h, 6h, 24h ago) and rolling averages to capture pollution trends.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="glass-panel">
            <div style="font-size:2rem; margin-bottom:10px;">&#128187;</div>
            <h4 style="color:#a78bfa; margin-bottom:8px; font-size:1.1rem;">Step 3: Training</h4>
            <p style="color:#94a3b8; font-size:12.5px; line-height:1.5;">A Multi-Output model is trained on Hopsworks using historical data to learn weather patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="glass-panel">
            <div style="font-size:2rem; margin-bottom:10px;">&#129302;</div>
            <h4 style="color:#10b981;">Step 4: Prediction</h4>
            <p style="color:#94a3b8; font-size:12.5px; line-height:1.5;">The best model predicts PM2.5, PM10, etc. for the next 72 hours, converting them into EPA AQI scores.</p>
        </div>
        """, unsafe_allow_html=True)

    model, data, best_model_name, best_r2, best_mae = load_model_and_data()

    st.markdown('<h3 style="color:#f8fafc; margin:20px 0 15px;">🚀 Active Production Model (Hopsworks Registry)</h3>', unsafe_allow_html=True)
    if best_r2 != -float('inf') and not pd.isna(best_r2):
        color = '#10b981' if 'xg' in str(best_model_name).lower() else '#60a5fa' if 'rf' in str(best_model_name).lower() else '#fbbf24'
        st.markdown(f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:20px; text-align:center;">
            <p style="color:#94a3b8; font-size:12px; margin:0 0 5px 0;">Currently Serving Predictions</p>
            <h3 style="color:{color}; margin:0 0 15px 0; font-size:1.4rem;">{best_model_name}</h3>
            <div style="display:flex; justify-content:center; gap:30px;">
                <div>
                    <span style="color:#94a3b8; font-size:13px; display:block;">R² Score</span>
                    <span style="color:#f8fafc; font-size:24px; font-weight:700;">{best_r2:.4f}</span>
                </div>
                <div>
                    <span style="color:#94a3b8; font-size:13px; display:block;">Mean Abs Error (MAE)</span>
                    <span style="color:#f8fafc; font-size:24px; font-weight:700;">{best_mae:.4f}</span>
                </div>
            </div>
            <p style="color:#64748b; font-size:12px; margin-top:15px; text-align:left;">
            <i>This model was dynamically selected by the Hopsworks pipeline because it achieved the highest R² score during the last training run. It outperformed all other candidates on the multi-pollutant objective.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No model history found in Hopsworks Registry. Using local fallback model.")




elif page == "Feature Explainability (SHAP)":
    st.markdown("""
    <div class="hero-banner">
        <h1>&#129504; Feature Explainability (SHAP)</h1>
        <p>Using SHAP (SHapley Additive exPlanations) to understand which weather and time features drive each pollutant prediction up or down.</p>
    </div>
    """, unsafe_allow_html=True)

    model, data, best_model_name, best_r2, best_mae = load_model_and_data()

    if model is None:
        st.error("Could not load model. Please run hopsworks_training.py first.")
    elif data is None:
        st.error("Featured historical data not found.")
    else:
        drop_cols = ['time', 'pm10', 'pm2_5', 'nitrogen_dioxide', 'sulphur_dioxide', 'carbon_monoxide']
        X = data.drop(columns=[c for c in drop_cols if c in data.columns])
        X_sample = X.sample(min(500, len(X)), random_state=42)

        # ── PM 2.5 SHAP + Feature Insights ────────────────────────────────────
        estimator = model.estimators_[0]
        estimator_type = type(estimator).__name__

        with st.spinner("Computing SHAP values for PM 2.5..."):
            try:
                if 'Linear' in estimator_type or 'Ridge' in estimator_type:
                    explainer = shap.LinearExplainer(estimator, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                elif 'RandomForest' in estimator_type or 'XGB' in estimator_type:
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    explainer = shap.Explainer(estimator, X_sample)
                    shap_values = explainer(X_sample).values
            except Exception as e:
                # Fallback for newer XGBoost versions that crash TreeExplainer
                explainer = shap.Explainer(estimator.predict, X_sample)
                shap_values = explainer(X_sample).values

        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<h3 style="color:#f8fafc; margin-bottom:4px;">🌐 SHAP Summary — PM 2.5</h3>', unsafe_allow_html=True)
            st.caption("Each dot is one data sample. Red = high feature value, Blue = low. Right = pushes prediction UP, Left = pulls DOWN.")
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            shap.summary_plot(shap_values, X_sample, show=False, plot_size=None, color_bar=True)
            ax = plt.gca()
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8', labelsize=10)
            ax.xaxis.label.set_color('#94a3b8')
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            plt.style.use('default')

        with col2:
            st.markdown('<h3 style="color:#f8fafc; margin-bottom:14px;">📖 What the Features Mean</h3>', unsafe_allow_html=True)
            insights = [
                {
                    "icon": "🌫️",
                    "title": "Historical Lags",
                    "subtitle": "lag_1h, lag_2h, lag_6h, lag_24h",
                    "color": "#3b82f6",
                    "bg": "rgba(59,130,246,0.08)",
                    "text": "The model's memory. Particulate matter is persistent — recent PM 2.5 readings are the strongest predictor of the next hour."
                },
                {
                    "icon": "📊",
                    "title": "Rolling Statistics",
                    "subtitle": "rolling_mean_6h/24h, rolling_std_24h",
                    "color": "#8b5cf6",
                    "bg": "rgba(139,92,246,0.08)",
                    "text": "Smoothed trends and volatility. High rolling std = unstable air quality, signalling a possible pollution episode."
                },
                {
                    "icon": "💨",
                    "title": "Weather Parameters",
                    "subtitle": "wind_speed, humidity, temperature",
                    "color": "#10b981",
                    "bg": "rgba(16,185,129,0.08)",
                    "text": "High wind disperses pollutants. Stagnant humid air traps them. Cold inversions lock particulates near ground level."
                },
                {
                    "icon": "🕐",
                    "title": "Temporal Features",
                    "subtitle": "hour, day_of_week, month",
                    "color": "#fbbf24",
                    "bg": "rgba(251,191,36,0.08)",
                    "text": "Rush-hour cycles, weekday traffic, and Karachi's seasons — winter inversions trap pollution, summer sea breezes disperse it."
                },
            ]
            for ins in insights:
                st.markdown(f"""
                <div style="background:{ins['bg']}; border-left:4px solid {ins['color']};
                            border-radius:8px; padding:16px 18px; margin-bottom:16px;">
                    <div style="font-size:22px; font-weight:800; color:{ins['color']}; margin-bottom:3px;">
                        {ins['icon']} {ins['title']}
                    </div>
                    <div style="font-size:14px; color:{ins['color']}; opacity:0.7; margin-bottom:10px; font-style:italic;">
                        {ins['subtitle']}
                    </div>
                    <div style="font-size:16px; color:#e2e8f0; line-height:1.75;">
                        {ins['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # How to read
        st.markdown("---")
        st.markdown('<h4 style="color:#f8fafc;">❓ How to read the SHAP plot</h4>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.markdown("""<div style="background:rgba(239,68,68,0.08); border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:22px;">🔴</div>
            <div style="color:#f87171; font-weight:700; margin:6px 0;">Red dot</div>
            <div style="color:#94a3b8; font-size:13px;">Feature value was HIGH for that sample</div>
            </div>""", unsafe_allow_html=True)
        with g2:
            st.markdown("""<div style="background:rgba(59,130,246,0.08); border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:22px;">🔵</div>
            <div style="color:#60a5fa; font-weight:700; margin:6px 0;">Blue dot</div>
            <div style="color:#94a3b8; font-size:13px;">Feature value was LOW for that sample</div>
            </div>""", unsafe_allow_html=True)
        with g3:
            st.markdown("""<div style="background:rgba(16,185,129,0.08); border-radius:10px; padding:14px; text-align:center;">
            <div style="font-size:22px;">↔️</div>
            <div style="color:#10b981; font-weight:700; margin:6px 0;">Dot position</div>
            <div style="color:#94a3b8; font-size:13px;">Right = pushes prediction UP. Left = pulls DOWN.</div>
            </div>""", unsafe_allow_html=True)

        # ── MULTI-POLLUTANT SHAP GRID ──────────────────────────────────────────
        st.markdown("---")
        st.markdown('<h3 style="color:#f8fafc; margin:10px 0;">🧪 SHAP by Pollutant</h3>', unsafe_allow_html=True)
        st.caption("Feature importance breakdown for each individual pollutant the model predicts.")

        pollutants = [
            (0, "PM 2.5",          "#3b82f6"),
            (1, "PM 10",           "#8b5cf6"),
            (2, "NO₂",             "#10b981"),
            (3, "SO₂",             "#fbbf24"),
            (4, "Carbon Monoxide", "#f87171"),
        ]

        # Compute all SHAP values upfront
        shap_by_pollutant = {}
        with st.spinner("Computing SHAP for all pollutants..."):
            for idx, name, color in pollutants:
                est = model.estimators_[idx]
                est_type = type(est).__name__
                if 'Linear' in est_type or 'Ridge' in est_type:
                    exp = shap.LinearExplainer(est, X_sample)
                elif 'RandomForest' in est_type or 'XGB' in est_type:
                    exp = shap.TreeExplainer(est)
                else:
                    exp = shap.Explainer(est, X_sample)
                shap_by_pollutant[idx] = exp.shap_values(X_sample)

        # Render in 2-column grid
        pairs = [(pollutants[i], pollutants[i+1]) if i+1 < len(pollutants) else (pollutants[i], None)
                 for i in range(0, len(pollutants), 2)]

        for left, right in pairs:
            c1, c2 = st.columns(2)
            for col, item in zip([c1, c2], [left, right]):
                if item is None:
                    continue
                idx, name, color = item
                with col:
                    st.markdown(f'<h4 style="color:{color}; margin-bottom:4px;">● {name}</h4>', unsafe_allow_html=True)
                    plt.style.use('dark_background')
                    fig_p, ax_p = plt.subplots(figsize=(6, 5))
                    fig_p.patch.set_facecolor('#0f172a')
                    ax_p.set_facecolor('#1e293b')
                    shap.summary_plot(
                        shap_by_pollutant[idx], X_sample,
                        show=False, plot_size=None,
                        color_bar=False, max_display=10
                    )
                    ax_p = plt.gca()
                    ax_p.set_facecolor('#1e293b')
                    ax_p.tick_params(colors='#94a3b8', labelsize=9)
                    ax_p.xaxis.label.set_color('#94a3b8')
                    ax_p.set_xlabel(f"SHAP value ({name})", color='#94a3b8', fontsize=9)
                    ax_p.spines['bottom'].set_color('#334155')
                    ax_p.spines['left'].set_color('#334155')
                    ax_p.spines['top'].set_visible(False)
                    ax_p.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_p, use_container_width=True)
                    plt.close()
                    plt.style.use('default')


