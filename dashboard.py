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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; }
    .stMetric {
        background: linear-gradient(135deg, #1e2530, #252d3a);
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #2e3a4e;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .status-good    { color: #00cc96; font-weight: 700; }
    .status-mod     { color: #faca2b; font-weight: 700; }
    .status-sens    { color: #ffa421; font-weight: 700; }
    .status-bad     { color: #ff4b4b; font-weight: 700; }
    .section-header { 
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
        text-transform: uppercase; color: #8899aa; margin-bottom: 0.5rem;
    }
    .source-badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 600; margin-left: 8px;
    }
    .badge-live   { background: #003d1e; color: #00cc96; border: 1px solid #00cc96; }
    .badge-local  { background: #2d1a00; color: #ffa421; border: 1px solid #ffa421; }
</style>
""", unsafe_allow_html=True)

# --- PATHS ---
PREDICTIONS_PATH = os.path.join(SCRIPT_DIR, 'data', 'cleaned', 'predictions.csv')
HISTORY_PATH     = os.path.join(SCRIPT_DIR, 'models', 'model_history.csv')
LOCAL_MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'best_model.pkl')
FEATURED_DATA    = os.path.join(SCRIPT_DIR, 'data', 'cleaned', 'featured_historical_data.csv')

# --- HELPERS ---
def get_aqi_status(aqi):
    if aqi <= 50:  return "Good",                           "status-good"
    if aqi <= 100: return "Moderate",                       "status-mod"
    if aqi <= 150: return "Unhealthy for Sensitive Groups", "status-sens"
    return              "Unhealthy",                        "status-bad"

# --- DATA LOADERS ---
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
def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return None

@st.cache_resource
def load_model_and_data():
    """Load model and featured data — tries Hopsworks Registry first, falls back to local."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    model = None
    if api_key:
        try:
            project = hopsworks.login()
            mr = project.get_model_registry()
            hw_model = mr.get_best_model(name="xgboost_multi_aqi", metric="r2", direction="max")
            model_dir = hw_model.download()
            model_path = os.path.join(model_dir, "xgboost_multi_aqi.pkl")
            model = joblib.load(model_path)
        except Exception:
            model = None

    if model is None and os.path.exists(LOCAL_MODEL_PATH):
        model = joblib.load(LOCAL_MODEL_PATH)

    # Data for SHAP (use local featured CSV — it's large but only loaded once)
    if os.path.exists(FEATURED_DATA):
        df = pd.read_csv(FEATURED_DATA)
    else:
        df = None

    return model, df

# --- SIDEBAR ---
st.sidebar.markdown("## 🌍 AQI-Trendz")
st.sidebar.markdown("Advanced Multi-Pollutant Forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["3-Day Forecast", "Model Diagnostics", "Feature Explainability (SHAP)"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Powered by XGBoost · Hopsworks · Open-Meteo")

# ── Load predictions (Hopsworks → local fallback) ─────────────────────────────
preds_df, data_source = load_predictions_hopsworks()
source_label = "live"
if preds_df is None or len(preds_df) == 0:
    preds_df = load_predictions_local()
    source_label = "local"

# =============================================================================
# PAGE: 3-DAY FORECAST
# =============================================================================
if page == "3-Day Forecast":
    badge = (
        '<span class="source-badge badge-live">☁ Live from Hopsworks</span>'
        if source_label == "live"
        else '<span class="source-badge badge-local">📁 Local cache</span>'
    )
    st.markdown(f"# 🌍 72-Hour Air Quality Forecast {badge}", unsafe_allow_html=True)

    if preds_df is not None and not preds_df.empty:
        current = preds_df.iloc[0]
        aqi_val = current.get('US_EPA_AQI', current.get('pred_pm2_5', 0))
        status, css = get_aqi_status(float(aqi_val))

        st.markdown(
            f'<p class="section-header">Current Status</p>'
            f'<h2 class="{css}">{status} — AQI {float(aqi_val):.0f}</h2>',
            unsafe_allow_html=True
        )

        if float(aqi_val) > 100:
            st.error(f"⚠️ Air Quality Alert: AQI is {float(aqi_val):.0f}. Sensitive groups should avoid outdoor activity.")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("PM 2.5",  f"{current.get('pred_pm2_5', 0):.1f} µg/m³")
        col2.metric("PM 10",   f"{current.get('pred_pm10', 0):.1f} µg/m³")
        col3.metric("NO₂",     f"{current.get('pred_nitrogen_dioxide', 0):.1f} µg/m³")
        col4.metric("SO₂",     f"{current.get('pred_sulphur_dioxide', 0):.1f} µg/m³")
        col5.metric("CO",      f"{current.get('pred_carbon_monoxide', 0):.1f} µg/m³")

        st.markdown("---")

        # AQI trend chart
        st.subheader("AQI Forecast — Next 72 Hours")
        fig = go.Figure()
        fig.add_hrect(y0=0,   y1=50,  line_width=0, fillcolor="green",  opacity=0.08, annotation_text="Good",        annotation_position="top left")
        fig.add_hrect(y0=50,  y1=100, line_width=0, fillcolor="yellow", opacity=0.08, annotation_text="Moderate",    annotation_position="top left")
        fig.add_hrect(y0=100, y1=150, line_width=0, fillcolor="orange", opacity=0.08, annotation_text="Sensitive",   annotation_position="top left")
        fig.add_hrect(y0=150, y1=500, line_width=0, fillcolor="red",    opacity=0.08, annotation_text="Unhealthy",   annotation_position="top left")
        fig.add_trace(go.Scatter(
            x=preds_df['time'], y=preds_df['US_EPA_AQI'],
            mode='lines', name='AQI',
            line=dict(width=3, color='#00d4ff', shape='spline'),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'
        ))
        fig.update_layout(template="plotly_dark", height=380,
                          xaxis_title="Time", yaxis_title="US EPA AQI",
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Pollutant breakdown
        st.subheader("Pollutant Breakdown")
        pol_map = {
            'pred_pm2_5': 'PM 2.5', 'pred_pm10': 'PM 10',
            'pred_nitrogen_dioxide': 'NO₂', 'pred_sulphur_dioxide': 'SO₂',
            'pred_carbon_monoxide': 'CO'
        }
        colors = ['#00d4ff', '#7b61ff', '#ff6b6b', '#ffd166', '#06d6a0']
        fig2 = go.Figure()
        for (col, label), color in zip(pol_map.items(), colors):
            if col in preds_df.columns:
                fig2.add_trace(go.Scatter(
                    x=preds_df['time'], y=preds_df[col],
                    mode='lines', name=label,
                    stackgroup='one',
                    line=dict(width=1, color=color)
                ))
        fig2.update_layout(template="plotly_dark", height=320,
                           xaxis_title="Time", yaxis_title="Concentration (µg/m³)",
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("No predictions available. Run the predict_pipeline.py first or wait for the hourly GitHub Action to execute.")

# =============================================================================
# PAGE: MODEL DIAGNOSTICS
# =============================================================================
elif page == "Model Diagnostics":
    st.title("📈 Model Performance History")
    hist_df = load_history()

    if hist_df is not None:
        st.dataframe(
            hist_df.sort_values('mae')
                   .style.highlight_min(subset=['mae', 'aqi_mae'], color='lightgreen'),
            use_container_width=True
        )
        st.subheader("AQI MAE per Experiment")
        fig = px.bar(hist_df, x='experiment_note', y='aqi_mae',
                     color='model_name', barmode='group', template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model history found at models/model_history.csv")

    st.markdown("---")
    st.subheader("Residual Analysis Plots")
    col1, col2 = st.columns(2)
    for path, col in [
        ('visualizations/residual_vs_actual_pm2_5.png', col1),
        ('visualizations/residual_vs_hour.png', col2)
    ]:
        full = os.path.join(SCRIPT_DIR, path)
        if os.path.exists(full):
            col.image(full, use_column_width=True)
        else:
            col.info("Run residual_analysis.py to generate this plot.")

# =============================================================================
# PAGE: SHAP EXPLAINABILITY
# =============================================================================
elif page == "Feature Explainability (SHAP)":
    st.title("🧠 Feature Explainability (SHAP)")
    st.markdown("Using **SHAP** to understand exactly what drives each pollution prediction.")

    with st.expander("❓ How to read these charts"):
        st.markdown("""
| Element | Meaning |
|---|---|
| **Dot to the RIGHT (positive SHAP)** | This feature is **pushing the prediction UP** (more pollution) |
| **Dot to the LEFT (negative SHAP)**  | This feature is **dragging the prediction DOWN** (less pollution) |
| **Red dot**  | The feature's value was **HIGH** for that sample |
| **Blue dot** | The feature's value was **LOW** for that sample |

**Example:** Red dots for `wind_speed_10m` on the LEFT side means "high wind is dragging pollution down" — that makes physical sense!
        """)

    target_dict = {
        'PM 2.5': 0, 'PM 10': 1, 'NO₂ (Nitrogen Dioxide)': 2,
        'SO₂ (Sulphur Dioxide)': 3, 'CO (Carbon Monoxide)': 4
    }
    selected_target = st.selectbox("Select Pollutant to Explain", list(target_dict.keys()))
    target_idx = target_dict[selected_target]

    with st.spinner("Loading model and computing SHAP values..."):
        model, data = load_model_and_data()

        if model is None:
            st.error("Could not load model. Please run hopsworks_training.py first.")
        elif data is None:
            st.error("Featured historical data not found.")
        else:
            drop_cols = ['time', 'pm10', 'pm2_5', 'nitrogen_dioxide',
                         'sulphur_dioxide', 'carbon_monoxide']
            X = data.drop(columns=[c for c in drop_cols if c in data.columns])
            X_sample = X.sample(min(1000, len(X)), random_state=42)

            xgb_estimator = model.estimators_[target_idx]
            explainer = shap.TreeExplainer(xgb_estimator)
            shap_values = explainer.shap_values(X_sample)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"Global Feature Importance — {selected_target}")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False, plot_size=None)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.subheader("Top Features (Mean |SHAP|)")
                mean_shap = pd.DataFrame({
                    'Feature': X_sample.columns,
                    'Mean |SHAP|': np.abs(shap_values).mean(axis=0)
                }).sort_values('Mean |SHAP|', ascending=False).head(15)

                fig2 = px.bar(
                    mean_shap, x='Mean |SHAP|', y='Feature',
                    orientation='h', template='plotly_dark',
                    color='Mean |SHAP|',
                    color_continuous_scale='Blues'
                )
                fig2.update_layout(height=500, yaxis={'categoryorder': 'total ascending'},
                                   margin=dict(l=0, r=0, t=10, b=0),
                                   coloraxis_showscale=False)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")
            st.subheader("Single Prediction Waterfall")
            st.markdown("Pick a row to see exactly how each feature pushed or pulled that one prediction.")
            row_idx = st.slider("Sample row index", 0, len(X_sample) - 1, 0)

            fig3, ax3 = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[row_idx],
                    base_values=explainer.expected_value,
                    data=X_sample.iloc[row_idx],
                    feature_names=list(X_sample.columns)
                ),
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()
