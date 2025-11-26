import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import numpy as np

st.set_page_config(page_title="OPSD PowerDesk Dashboard", layout="wide")

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

config = load_config()
countries = config['data']['countries']

st.title("⚡ OPSD PowerDesk Forecasting Dashboard")

# i. Country Selector
selected_country = st.sidebar.selectbox("Select Country", countries, index=0) # Default DE

# Load Data
live_file = f'outputs/{selected_country}_live_simulation.csv'
updates_file = f'outputs/{selected_country}_online_updates.csv'
anomalies_file = f'outputs/{selected_country}_anomalies.csv'

if os.path.exists(live_file):
    df_live = pd.read_csv(live_file, parse_dates=['timestamp'])
    
    # ii. Live Series (Last 7-14 days)
    st.subheader("Live Series (Last 14 Days)")
    
    # Filter last 14 days
    last_timestamp = df_live['timestamp'].max()
    start_view = last_timestamp - pd.Timedelta(days=14)
    df_view = df_live[df_live['timestamp'] >= start_view]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_view['timestamp'], df_view['y_true'], label='Actual', color='black', linewidth=1.5)
    ax.plot(df_view['timestamp'], df_view['yhat'], label='Forecast', color='blue', linestyle='--', alpha=0.8)
    ax.fill_between(df_view['timestamp'], df_view['lo'], df_view['hi'], color='blue', alpha=0.1, label='80% PI')
    
    # Highlight anomalies
    anoms = df_view[np.abs(df_view['z_score']) >= 3.0]
    ax.scatter(anoms['timestamp'], anoms['y_true'], color='red', label='Anomaly (|z|>=3)', zorder=5)
    
    ax.legend()
    st.pyplot(fig)
    
    # iii. Forecast Cone (Next 24h)
    # In a real live dashboard, this would be future. 
    # Here we show the "latest" available forecast from the simulation end, 
    # OR we can show the forecast cone for the *next* 24h relative to the last simulated point.
    # Since simulation ended, let's pretend "NOW" is the end of simulation.
    # We don't have future truth, but we might have the forecast made at the last step.
    # The simulation loop saved yhat/lo/hi for the simulated steps.
    # Let's show the last 24h of the simulation as the "Forecast Cone" for demo purposes, 
    # or ideally, we would project 24h into the future beyond the simulation.
    # Given we stopped simulation, let's show the last 24h segment.
    
    st.subheader("Forecast Cone (Latest 24h)")
    last_24h = df_live.iloc[-24:]
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(last_24h['timestamp'], last_24h['yhat'], label='Forecast Mean', color='blue')
    ax2.fill_between(last_24h['timestamp'], last_24h['lo'], last_24h['hi'], color='blue', alpha=0.2, label='80% PI')
    ax2.plot(last_24h['timestamp'], last_24h['y_true'], label='Actual (if available)', color='black', linestyle=':')
    ax2.legend()
    st.pyplot(fig2)
    
    # iv. Anomaly Tape
    st.subheader("Anomaly Tape")
    # Heatmap style or simple strip
    # Create a dataframe indexed by time, value is 1 if anomaly, 0 else
    df_view['is_anomaly'] = (np.abs(df_view['z_score']) >= 3.0).astype(int)
    
    # Plot as a strip
    fig3, ax3 = plt.subplots(figsize=(12, 1))
    # We can use pcolor or imshow
    # Reshape for imshow: [1, N]
    matrix = df_view['is_anomaly'].values.reshape(1, -1)
    ax3.imshow(matrix, aspect='auto', cmap='Reds', vmin=0, vmax=1)
    ax3.set_yticks([])
    ax3.set_xlabel("Time")
    # Ticks formatting is tricky with imshow, let's just show it visually
    st.pyplot(fig3)
    
    # v. KPI Tiles
    st.subheader("Key Performance Indicators (Rolling 7d)")
    
    # Calculate rolling 7d metrics on the fly
    recent_7d = df_live[df_live['timestamp'] >= last_timestamp - pd.Timedelta(days=7)]
    
    if not recent_7d.empty:
        # MASE
        # Denom: mean abs diff of y_true (seasonality 24)
        denom = np.abs(recent_7d['y_true'].diff(24)).mean()
        if denom == 0 or np.isnan(denom): denom = 1
        mase = np.abs(recent_7d['y_true'] - recent_7d['yhat']).mean() / denom
        
        # Coverage
        coverage = ((recent_7d['y_true'] >= recent_7d['lo']) & (recent_7d['y_true'] <= recent_7d['hi'])).mean() * 100
        
        # Anomalies today (last 24h)
        last_24h_anoms = df_live[df_live['timestamp'] >= last_timestamp - pd.Timedelta(hours=24)]
        anoms_today = (np.abs(last_24h_anoms['z_score']) >= 3.0).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rolling 7d MASE", f"{mase:.3f}")
        col2.metric("Rolling 7d Coverage", f"{coverage:.1f}%")
        col3.metric("Anomalies (24h)", f"{anoms_today}")
        
        # vi. Update Status
        if os.path.exists(updates_file):
            df_updates = pd.read_csv(updates_file, parse_dates=['timestamp'])
            if not df_updates.empty:
                last_update = df_updates.iloc[-1]
                col4.metric("Last Update", f"{last_update['timestamp'].strftime('%Y-%m-%d %H:%M')}", delta=last_update['reason'])
            else:
                col4.metric("Last Update", "None")
        else:
            col4.metric("Last Update", "N/A")
            
else:
    st.info(f"Live simulation data for {selected_country} not found. Run simulation first.")

# Tab for full tables
with st.expander("View Raw Data"):
    if os.path.exists(live_file):
        st.dataframe(df_live.tail(100))
    if os.path.exists(updates_file):
        st.write("Updates Log")
        st.dataframe(pd.read_csv(updates_file))
