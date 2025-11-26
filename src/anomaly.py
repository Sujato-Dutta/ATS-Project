import pandas as pd
import numpy as np
from scipy.stats import zscore
import yaml
import os

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def detect_anomalies_zscore(residuals, window_hours=336, threshold=3.0):
    # Rolling mean and std
    # min_periods = 168 as requested
    rolling_mean = residuals.rolling(window=window_hours, min_periods=168).mean()
    rolling_std = residuals.rolling(window=window_hours, min_periods=168).std()
    
    z_scores = (residuals - rolling_mean) / rolling_std
    anomalies = np.abs(z_scores) >= threshold
    
    return z_scores, anomalies

def detect_anomalies_cusum(residuals, k=0.5, h=5.0):
    # CUSUM on standardized residuals (using global stats for simplicity or rolling?)
    # Usually CUSUM is on standardized process. Let's use the rolling z-scores if available, 
    # or standardize globally. Prompt says "Optional CUSUM on zt". So use z_scores!
    
    # We need z_scores first. 
    # This function will take z_scores as input ideally.
    pass 

def run_anomaly_detection(config):
    countries = config['data']['countries']
    # Hardcoded requirements from prompt override config if needed, but let's use config or defaults
    z_window = 336
    z_thresh = 3.0
    cusum_k = 0.5
    cusum_h = 5.0
    
    for country in countries:
        # Load forecasts (Test set)
        file_path = f'outputs/{country}_forecasts_test.csv'
        if not os.path.exists(file_path):
            print(f"Skipping {country}, forecasts not found.")
            continue
            
        print(f"Detecting anomalies for {country}...")
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Compute residuals
        df['residual'] = df['y_true'] - df['yhat']
        
        # Z-score
        z_scores, anomalies_z = detect_anomalies_zscore(df['residual'], z_window, z_thresh)
        df['z_score'] = z_scores
        df['flag_z'] = anomalies_z.astype(int)
        
        # CUSUM on zt
        # S+ = max(0, S+ + zt - k)
        # S- = max(0, S- - zt - k) -> usually -zt - k for negative drift? 
        # Standard CUSUM for mean shift: 
        # S_hi[i] = max(0, S_hi[i-1] + z[i] - k)
        # S_lo[i] = max(0, S_lo[i-1] - z[i] - k)
        
        s_hi = np.zeros(len(df))
        s_lo = np.zeros(len(df))
        flag_cusum = np.zeros(len(df), dtype=int)
        
        z_values = df['z_score'].fillna(0).values # Fill NaN with 0 for CUSUM
        
        for i in range(1, len(df)):
            s_hi[i] = max(0, s_hi[i-1] + z_values[i] - cusum_k)
            s_lo[i] = max(0, s_lo[i-1] - z_values[i] - cusum_k)
            
            if s_hi[i] > cusum_h or s_lo[i] > cusum_h:
                flag_cusum[i] = 1
                # Reset after alarm? "alarm when..." - usually reset.
                s_hi[i] = 0
                s_lo[i] = 0
                
        df['flag_cusum'] = flag_cusum
        
        # Save outputs/<CC>_anomalies.csv
        # Columns: timestamp, y_true, yhat, z_resid, flag_z, [flag_cusum]
        output_df = df[['y_true', 'yhat', 'z_score', 'flag_z', 'flag_cusum']].copy()
        output_df.rename(columns={'z_score': 'z_resid'}, inplace=True)
        output_df.to_csv(f'outputs/{country}_anomalies.csv')
        print(f"  Found {df['flag_z'].sum()} Z-anomalies. Saved to outputs/{country}_anomalies.csv")

if __name__ == "__main__":
    config = load_config()
    run_anomaly_detection(config)
