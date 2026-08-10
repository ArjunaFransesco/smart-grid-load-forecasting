import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="⚡ Smart Grid Load Demand & Renewable Dispatch Engine",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.3rem; font-weight: 800; color: #0f172a; margin-bottom: 5px; }
    .sub-title { font-size: 1.1rem; color: #475569; margin-bottom: 25px; }
    .metric-box { background: #f1f5f9; padding: 15px; border-radius: 8px; border-left: 5px solid #2563eb; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Smart Grid Load Demand & Renewable Dispatch Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Developed by <b>Arjuna Fransesco</b> | Advanced Machine Learning & Energy AI Portfolio</div>', unsafe_allow_html=True)

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, "models", "lightgbm_grid_model.joblib"))
    feature_cols = joblib.load(os.path.join(base_dir, "models", "feature_columns.joblib"))
    with open(os.path.join(base_dir, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    return model, feature_cols, metrics

model, feature_cols, metrics = load_models()

# Sidebar Controls
st.sidebar.header("🎛️ Weather & Grid Telemetry Simulation")
temp_input = st.sidebar.slider("Ambient Temperature (°C)", -10.0, 42.0, 24.5, step=0.5)
humidity_input = st.sidebar.slider("Relative Humidity (%)", 10.0, 100.0, 60.0, step=1.0)
wind_input = st.sidebar.slider("Wind Speed (m/s)", 0.0, 25.0, 7.5, step=0.5)
solar_input = st.sidebar.slider("Solar Irradiance (W/m²)", 0.0, 1000.0, 550.0, step=10.0)
hour_input = st.sidebar.slider("Hour of Day (0-23)", 0, 23, 14)
dayofweek_input = st.sidebar.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
month_input = st.sidebar.slider("Month of Year (1-12)", 1, 12, 7)
is_weekend = 1 if dayofweek_input in ["Saturday", "Sunday"] else 0

dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
dow_val = dow_map[dayofweek_input]

# Calculate Renewable Generation
solar_gen = max(0.0, round(solar_input * 0.45, 2))
wind_gen = max(0.0, round((wind_input - 3.0)**2 * 2.8 if 3.0 <= wind_input < 12.0 else (226.8 if wind_input >= 12.0 else 0.0), 2))
renewable_tot = round(solar_gen + wind_gen, 2)

# Build feature row for prediction
hour_sin = np.sin(2 * np.pi * hour_input / 24)
hour_cos = np.cos(2 * np.pi * hour_input / 24)
month_sin = np.sin(2 * np.pi * month_input / 12)
month_cos = np.cos(2 * np.pi * month_input / 12)
dayofweek_sin = np.sin(2 * np.pi * dow_val / 7)
dayofweek_cos = np.cos(2 * np.pi * dow_val / 7)

cooling_deg = max(0.0, temp_input - 22.0)
heating_deg = max(0.0, 14.0 - temp_input)
temp_sq = temp_input ** 2

# Base proxy lags
est_base_load = 1450.0 + cooling_deg * 25.0 + heating_deg * 20.0 - (200.0 if is_weekend else 0.0)
load_lag_1h = est_base_load * 0.98
load_lag_2h = est_base_load * 0.96
load_lag_24h = est_base_load * 1.02
load_lag_168h = est_base_load * 0.99
load_rolling_mean_24h = est_base_load
load_rolling_std_24h = 45.0
temp_rolling_mean_6h = temp_input

input_data = pd.DataFrame([{
    "temperature_c": temp_input,
    "humidity_pct": humidity_input,
    "wind_speed_ms": wind_input,
    "solar_irradiance_wm2": solar_input,
    "renewable_total_mw": renewable_tot,
    "is_weekend": is_weekend,
    "hour_sin": hour_sin,
    "hour_cos": hour_cos,
    "month_sin": month_sin,
    "month_cos": month_cos,
    "dayofweek_sin": dayofweek_sin,
    "dayofweek_cos": dayofweek_cos,
    "cooling_degree": cooling_deg,
    "heating_degree": heating_deg,
    "temp_squared": temp_sq,
    "load_lag_1h": load_lag_1h,
    "load_lag_2h": load_lag_2h,
    "load_lag_24h": load_lag_24h,
    "load_lag_168h": load_lag_168h,
    "load_rolling_mean_24h": load_rolling_mean_24h,
    "load_rolling_std_24h": load_rolling_std_24h,
    "temp_rolling_mean_6h": temp_rolling_mean_6h
}])[feature_cols]

predicted_load = model.predict(input_data)[0]
net_demand = max(0.0, predicted_load - renewable_tot)
renewable_pct = min(100.0, (renewable_tot / predicted_load) * 100)

# Metrics Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("⚡ Forecasted Grid Load", f"{predicted_load:,.1f} MW")
with c2:
    st.metric("🌱 Renewable Generation", f"{renewable_tot:,.1f} MW", delta=f"{renewable_pct:.1f}% of Grid")
with c3:
    st.metric("🏭 Net Thermal Dispatch Needed", f"{net_demand:,.1f} MW")
with c4:
    st.metric("🎯 Model Accuracy (MAPE)", f"{metrics['LightGBM_Regressor']['MAPE_Percent']}%", delta="R² = " + str(metrics['LightGBM_Regressor']['R2_Score']))

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 24-Hour Forecast Curve", "🚨 Grid Stability & Peak Shaving", "📑 Benchmark Specifications"])

with tab1:
    st.subheader("Hourly Load Profile vs Renewable Generation")
    st.image("reports/load_forecast_actual_vs_pred.png", use_container_width=True)

with tab2:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("💡 Automated Grid Dispatch Advisory")
        if predicted_load > 2200:
            st.error("🚨 CRITICAL PEAK DEMAND WARNING: Initiate Battery Energy Storage System (BESS) peak-shaving discharge and alert peaker plants.")
        elif renewable_pct > 40:
            st.success("✅ HIGH RENEWABLE PENETRATION: Optimal conditions to route excess green power into BESS storage / EV charging networks.")
        else:
            st.info("ℹ️ NOMINAL GRID OPERATION: Load within standard operating reserves. Base-load combined cycle units active.")

        st.write(f"**Solar Generation (Estimated):** `{solar_gen} MW`")
        st.write(f"**Wind Generation (Estimated):** `{wind_gen} MW`")
        st.write(f"**Cooling Degree Factor:** `{cooling_deg:.1f} °C`")
    
    with col_b:
        st.subheader("Top Predictive Features")
        st.image("reports/feature_importance.png", use_container_width=True)

with tab3:
    st.json(metrics)
