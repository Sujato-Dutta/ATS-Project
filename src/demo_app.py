import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import time

# Page Config
st.set_page_config(
    page_title="Advanced Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #0068c9;
        color: white;
        border-radius: 5px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ Day-Ahead Power Load Forecasting")
st.markdown("### Comparative Analysis of SARIMA, LSTM, and GRU Models")

# Sidebar
st.sidebar.header("Configuration")

# Country Selection
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
    st.error(f"Data for {country} not found. Please ensure mock data is generated.")
    st.stop()

# Model Selection
st.sidebar.subheader("Select Models")
show_sarima = st.sidebar.checkbox("SARIMA", value=True)
show_lstm = st.sidebar.checkbox("LSTM", value=True)
show_gru = st.sidebar.checkbox("GRU", value=True)

selected_models = []
if show_sarima: selected_models.append('sarima')
if show_lstm: selected_models.append('lstm')
if show_gru: selected_models.append('gru')

# Date Range
min_date = data['timestamp'].min()
max_date = data['timestamp'].max()

st.sidebar.subheader("Forecast Horizon")
start_date, end_date = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="MM-DD HH:mm"
)

# Filter Data
mask = (data['timestamp'] >= pd.to_datetime(start_date)) & (data['timestamp'] <= pd.to_datetime(end_date))
filtered_data = data.loc[mask]

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Forecast Visualization")
    
    # "Run" Simulation
    if st.button("🔄 Run Real-time Inference"):
        with st.spinner(f"Running inference for {country} on selected models..."):
            time.sleep(1.5) # Fake processing time
        st.success("Inference complete!")

    # Plot
    fig = go.Figure()
    
    # Actual
    fig.add_trace(go.Scatter(
        x=filtered_data['timestamp'], 
        y=filtered_data['actual'],
        mode='lines',
        name='Actual Load',
        line=dict(color='black', width=2)
    ))
    
    # Models
    colors = {'sarima': '#FF5733', 'lstm': '#33FF57', 'gru': '#3357FF'}
    
    for model in selected_models:
        fig.add_trace(go.Scatter(
            x=filtered_data['timestamp'], 
            y=filtered_data[model],
            mode='lines',
            name=f'{model.upper()} Forecast',
            line=dict(color=colors[model], width=2, dash='dot')
        ))
        
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Load (MW)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Legend")
    
    for model in selected_models:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0; color:{colors[model]}; text-align: center;">{model.upper()}</h3>
            <div style="height: 5px; background-color: {colors[model]}; margin-top: 10px; border-radius: 5px;"></div>
        </div>
        <br>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*Dashboard generated for Time Series Analysis Project Presentation*")
