import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yaml
import os
import ast
import time
from datetime import timedelta
import warnings

warnings.filterwarnings("ignore")

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model_orders(filepath='outputs/selected_models.txt'):
    orders = {}
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        content = f.read()
    blocks = content.split('\n\n')
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        country = lines[0].strip(':')
        order_str = lines[1].split(':')[1].strip()
        seasonal_str = lines[2].split(':')[1].strip()
        orders[country] = {
            'order': ast.literal_eval(order_str),
            'seasonal_order': ast.literal_eval(seasonal_str)
        }
    return orders

def calculate_rolling_metrics(results_df, window_days=7):
    # Calculate rolling MASE and Coverage over last window_days
    # Need training data for MASE denominator? Approximation: use recent diffs.
    if len(results_df) < 24 * window_days:
        return np.nan, np.nan
        
    recent = results_df.iloc[-24*window_days:]
    y_true = recent['y_true']
    y_pred = recent['yhat']
    
    # MASE approx denominator (naive forecast on seasonality)
    # d = mean(|y_t - y_{t-24}|)
    d = np.abs(y_true.diff(24)).mean()
    if d == 0 or np.isnan(d):
        d = 1.0
        
    mase = np.abs(y_true - y_pred).mean() / d
    
    # Coverage
    coverage = ((y_true >= recent['lo']) & (y_true <= recent['hi'])).mean() * 100
    
    return mase, coverage

def run_live_simulation(config):
    # Pick ONE country (DE)
    country = 'DE'
    print(f"Starting Live Simulation for {country}...")
    
    orders = load_model_orders()
    if orders is None or country not in orders:
        print("Model orders not found.")
        return

    file_path = f'data/{country}.csv'
    if not os.path.exists(file_path):
        print("Data file not found.")
        return
        
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    series = df['load']
    
    # Simulation settings
    start_history_days = 120 # "Live: start history 120d"
    min_sim_hours = 2000
    
    # Adaptation settings
    refit_window_days = 90
    drift_alpha = 0.1
    drift_percentile_window_days = 30
    
    # Initial setup
    history_end = start_history_days * 24
    if len(series) < history_end + min_sim_hours:
        print("Not enough data.")
        return
        
    # We simulate row by row
    # But for efficiency, we can batch update, but logic must be row-by-row for Z-score/Drift?
    # "Loop each hour: append next row; at 00:00 UTC forecast next 24h; update z-score... check drift"
    
    # Initialize model
    order = orders[country]['order']
    seasonal_order = orders[country]['seasonal_order']
    
    # Initial training
    train_data = series.iloc[:history_end]
    model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, 
                    enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    print("Initial model trained.")
    
    # State tracking
    current_history = train_data.copy()
    residuals = [] # Keep track of residuals for Z-score
    z_scores = []
    ewma_z = 0 # Initial EWMA
    
    # Pre-calculate rolling stats for Z-score (need history)
    # We need to maintain a rolling window for Z-score calculation (336h)
    # Let's keep a buffer of residuals
    residual_buffer = [] 
    
    # Logs
    updates_log = []
    results = []
    
    # Simulation Loop
    sim_start_idx = history_end
    sim_end_idx = min(len(series), sim_start_idx + min_sim_hours)
    
    print(f"Simulating {sim_end_idx - sim_start_idx} hours...")
    
    last_refit_time = series.index[sim_start_idx-1]
    
    for i in range(sim_start_idx, sim_end_idx):
        timestamp = series.index[i]
        true_value = series.iloc[i]
        
        # 1. Append next row (Ingestion)
        # In reality, we observe true_value NOW.
        # But we made a forecast for this timestamp YESTERDAY (at 00:00 UTC of this day).
        # Wait, "at 00:00 UTC forecast next 24h".
        # So at 00:00, we forecast 01:00 to 00:00 next day.
        
        # Let's handle forecasting first if it's 00:00
        if timestamp.hour == 0:
            # Forecast next 24h (timestamp to timestamp+23h)
            # We use current_history (up to i-1)
            
            # Check Adaptation (Daily Schedule)
            # "Rolling SARIMA refit... daily at 00:00"
            # We refit every day at 00:00? Yes.
            # "A) Rolling SARIMA 90d daily"
            
            # Check Drift Trigger first? Or Scheduled?
            # "Daily at 00:00 refit... ALSO refit on drift trigger"
            # If we refit daily, drift trigger might be redundant unless drift happens mid-day?
            # Or maybe "Daily" means we check daily? 
            # Let's assume we refit daily at 00:00 regardless.
            
            update_reason = "scheduled"
            t0 = time.time()
            
            # Refit
            train_start = max(0, len(current_history) - refit_window_days * 24)
            history_window = current_history.iloc[train_start:]
            
            model = SARIMAX(history_window, order=order, seasonal_order=seasonal_order, 
                            enforce_stationarity=False, enforce_invertibility=False)
            try:
                model_fit = model.fit(disp=False)
                duration = time.time() - t0
                updates_log.append({
                    'timestamp': timestamp,
                    'strategy': 'Rolling SARIMA',
                    'reason': update_reason,
                    'duration_s': duration
                })
            except:
                print(f"Refit failed at {timestamp}")
            
            # Forecast 24 steps
            fc = model_fit.get_forecast(steps=24)
            yhat_24 = fc.predicted_mean
            conf_24 = fc.conf_int(alpha=0.2)
            
            # Store forecasts for the coming hours
            # We need to map these forecasts to the timestamps they correspond to
            # yhat_24 index should be correct if model_fit has freq
            
            # We will access these forecasts as we loop through the hours
            current_forecasts = pd.DataFrame({
                'yhat': yhat_24,
                'lo': conf_24.iloc[:, 0],
                'hi': conf_24.iloc[:, 1]
            })
            
        # 2. Retrieve forecast for this hour (made previously)
        # If i is the first step (00:00), we just made the forecast.
        # If i is 15:00, we use the forecast made at 00:00 today.
        
        # Actually, if we start at random hour, we might not have forecast. 
        # But we start at history_end. Let's assume history_end is aligned or we make initial forecast.
        # If timestamp.hour != 0 and we just started, we need a forecast.
        # For simplicity, let's ensure we start simulation at 00:00 or handle startup.
        
        if i == sim_start_idx and timestamp.hour != 0:
            # Make initial forecast for remaining hours of day
            steps = 24 - timestamp.hour
            fc = model_fit.get_forecast(steps=steps)
            current_forecasts = pd.DataFrame({
                'yhat': fc.predicted_mean,
                'lo': fc.conf_int(alpha=0.2).iloc[:, 0],
                'hi': fc.conf_int(alpha=0.2).iloc[:, 1]
            })
            
        # Get forecast for current timestamp
        if timestamp in current_forecasts.index:
            yhat = current_forecasts.loc[timestamp, 'yhat']
            lo = current_forecasts.loc[timestamp, 'lo']
            hi = current_forecasts.loc[timestamp, 'hi']
        else:
            # Should not happen if logic is correct
            yhat, lo, hi = np.nan, np.nan, np.nan
            
        # 3. Compute Residual & Z-score
        resid = true_value - yhat
        residuals.append(resid)
        residual_buffer.append(resid)
        
        # Rolling stats (336h)
        if len(residual_buffer) > 336:
            residual_buffer.pop(0)
            
        if len(residual_buffer) >= 168:
            roll_mean = np.mean(residual_buffer)
            roll_std = np.std(residual_buffer)
            if roll_std == 0: roll_std = 1e-6
            z = (resid - roll_mean) / roll_std
        else:
            z = 0.0
            
        z_scores.append(z)
        
        # 4. Check Drift
        # EWMA(|z|; alpha=0.1)
        abs_z = abs(z)
        ewma_z = 0.1 * abs_z + 0.9 * ewma_z
        
        # Threshold: 95th percentile of |z| over last 30 days (720h)
        # We need history of |z|
        # Let's keep z history
        z_history = z_scores[-720:]
        if len(z_history) > 100:
            z_thresh = np.percentile(np.abs(z_history), 95)
        else:
            z_thresh = 3.0 # Default fallback
            
        drift_triggered = ewma_z > z_thresh
        
        # If drift triggered, adapt immediately (unless we just adapted at 00:00)
        # "ALSO refit on drift trigger"
        if drift_triggered and timestamp.hour != 0: # Avoid double refit at 00:00
            # Refit
            update_reason = "drift"
            t0 = time.time()
            
            # Refit logic
            train_start = max(0, len(current_history) - refit_window_days * 24)
            history_window = current_history.iloc[train_start:]
            # Append current observation first? 
            # "Ingest -> Forecast -> Detect -> Adapt". 
            # We haven't appended current row to history yet in this code (it's in 'series' but not 'current_history' used for model).
            # We should append it before refit?
            # "Append next row... check drift... if triggered, adapt"
            # So yes, include current row.
            
            # Update history with current row
            current_history = pd.concat([current_history, pd.Series([true_value], index=[timestamp])])
            
            # Refit
            model = SARIMAX(history_window, order=order, seasonal_order=seasonal_order, 
                            enforce_stationarity=False, enforce_invertibility=False)
            try:
                model_fit = model.fit(disp=False)
                duration = time.time() - t0
                updates_log.append({
                    'timestamp': timestamp,
                    'strategy': 'Rolling SARIMA',
                    'reason': update_reason,
                    'duration_s': duration
                })
                # Reset EWMA? Usually not, but maybe?
            except:
                pass
                
            # Re-forecast next 24h? 
            # "Update your model so it stays calibrated". 
            # Usually implies we might update future forecasts.
            # But we already made forecasts at 00:00. 
            # If we adapt mid-day, should we update the rest of the day's forecast?
            # Let's assume yes, we update the forecast cone for remaining hours.
            steps = 24 # Forecast next 24h from NOW
            fc = model_fit.get_forecast(steps=steps)
            
            # Update current_forecasts
            new_fc_df = pd.DataFrame({
                'yhat': fc.predicted_mean,
                'lo': fc.conf_int(alpha=0.2).iloc[:, 0],
                'hi': fc.conf_int(alpha=0.2).iloc[:, 1]
            })
            # Merge/Overwrite
            current_forecasts = current_forecasts.combine_first(new_fc_df) # Keep existing if not overwritten? No, overwrite.
            current_forecasts.update(new_fc_df)
            
        else:
            # Just append to history
            current_history = pd.concat([current_history, pd.Series([true_value], index=[timestamp])])
            
            # Update state (filter)
            try:
                model_fit = model_fit.append(pd.Series([true_value], index=[timestamp]), refit=False)
            except:
                pass

        # Store Result
        results.append({
            'timestamp': timestamp,
            'y_true': true_value,
            'yhat': yhat,
            'lo': lo,
            'hi': hi,
            'z_score': z,
            'ewma_z': ewma_z,
            'drift_thresh': z_thresh,
            'drift_flag': drift_triggered
        })
        
        if i % 100 == 0:
            print(f"  Simulated {i - sim_start_idx} hours...")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'outputs/{country}_live_simulation.csv', index=False)
    
    updates_df = pd.DataFrame(updates_log)
    updates_df.to_csv(f'outputs/{country}_online_updates.csv', index=False)
    
    print("Live simulation completed.")

if __name__ == "__main__":
    config = load_config()
    run_live_simulation(config)
