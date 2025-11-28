import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os
import time
from datetime import datetime

# Page Config
st.set_page_config(
    page_title="Advanced Time Series Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .kpi-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #0068c9;
    }
    .kpi-value { font-size: 24px; font-weight: bold; color: #333; }
    .kpi-label { font-size: 14px; color: #666; }
    .status-box {
        padding: 10px;
        background-color: #e6fffa;
        border: 1px solid #b2f5ea;
        border-radius: 5px;
        color: #2c7a7b;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ Day-Ahead Power Load Forecasting System")

# Sidebar
st.sidebar.header("Configuration")
country = st.sidebar.selectbox(
    "Select Region",
    ("DE", "FR", "GB"),
    format_func=lambda x: f"{x} - {'Germany' if x=='DE' else 'France' if x=='FR' else 'Great Britain'}"
)

# Load Data
@st.cache_data
def load_data(country_code):
    file_path = os.path.join("mock_outputs", f"{country_code}_mock_forecasts.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return None

data = load_data(country)
if data is None:
    st.error("Data not found.")
    st.stop()

# Date Range (Default to last 14 days)
max_date = data['timestamp'].max()
min_date = max_date - pd.Timedelta(days=14)

# Filter Data
filtered_data = data[(data['timestamp'] >= min_date) & (data['timestamp'] <= max_date)]

# --- UPDATE STATUS ---
st.markdown(f"""
<div class="status-box">
    <strong>✅ System Online</strong> | Last Update: {max_date.strftime('%Y-%m-%d %H:%M:%S')} UTC | 
    Reason: <em>Scheduled Hourly Refit</em>
</div>
""", unsafe_allow_html=True)

# --- KPI TILES ---
col1, col2, col3, col4 = st.columns(4)

# Calculate Mock KPIs
mase = np.random.uniform(0.5, 0.9) # Mock MASE
pi_coverage = np.random.uniform(85, 95) # Mock PI Coverage
anomalies_today = filtered_data['flag_z'].iloc[-24:].sum() # Anomalies in last 24h

with col1:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{mase:.2f}</div><div class="kpi-label">Rolling 7d MASE</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{pi_coverage:.1f}%</div><div class="kpi-label">80% PI Coverage (7d)</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-value">{int(anomalies_today)}</div><div class="kpi-label">Anomaly Hours (Today)</div></div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="kpi-card"><div class="kpi-value">SARIMA</div><div class="kpi-label">Best Model (Live)</div></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- MAIN PLOT ---
st.subheader("Live Forecast & Anomaly Detection")

fig = go.Figure()

# 1. Forecast Cone (80% PI)
fig.add_trace(go.Scatter(
    x=filtered_data['timestamp'], y=filtered_data['upper_bound'],
    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=filtered_data['timestamp'], y=filtered_data['lower_bound'],
    mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 200, 0.2)',
    name='80% Prediction Interval'
))

# 2. Actual Load
fig.add_trace(go.Scatter(
    x=filtered_data['timestamp'], y=filtered_data['actual'],
    mode='lines', name='Actual Load', line=dict(color='black', width=2)
))

# 3. Forecast (SARIMA - Primary)
fig.add_trace(go.Scatter(
    x=filtered_data['timestamp'], y=filtered_data['sarima'],
    mode='lines', name='Forecast (Mean)', line=dict(color='#0068c9', width=2, dash='dash')
))

# 4. Anomalies (Red Markers)
anomalies = filtered_data[filtered_data['flag_z'] == 1]
fig.add_trace(go.Scatter(
    x=anomalies['timestamp'], y=anomalies['actual'],
    mode='markers', name='Anomaly Detected',
    marker=dict(color='red', size=10, symbol='x')
))

fig.update_layout(
    height=500,
    xaxis_title="Time",
    yaxis_title="Load (MW)",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- MODEL COMPARISON (Bottom) ---
with st.expander("Compare All Models (SARIMA vs LSTM vs GRU)"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data['actual'], name='Actual', line=dict(color='black')))
    fig2.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data['sarima'], name='SARIMA', line=dict(color='#FF5733')))
    fig2.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data['lstm'], name='LSTM', line=dict(color='#33FF57')))
    fig2.add_trace(go.Scatter(x=filtered_data['timestamp'], y=filtered_data['gru'], name='GRU', line=dict(color='#3357FF')))
    fig2.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)
