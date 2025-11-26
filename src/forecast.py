import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import yaml
import os
import ast
try:
    from src.metrics import calculate_metrics
except ImportError:
    from metrics import calculate_metrics
import warnings

warnings.filterwarnings("ignore")

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model_orders(filepath='outputs/selected_models.txt'):
    orders = {}
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Run EDA first.")
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Simple parsing of the text file format
    # Country:
    #   Order: (p, d, q)
    #   Seasonal Order: (P, D, Q, s)
    
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

def backtest_country(df, country, orders, config):
    print(f"\nForecasting for {country}...")
    series = df['load']
    
    # 2.1 Splits
    n = len(series)
    train_end = int(n * 0.8)
    dev_end = int(n * 0.9)
    
    train_data = series.iloc[:train_end]
    dev_data = series.iloc[train_end:dev_end]
    test_data = series.iloc[dev_end:]
    
    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
    
    # Model orders
    if country not in orders:
        print(f"  No orders found for {country}, skipping.")
        return
    
    order = orders[country]['order']
    seasonal_order = orders[country]['seasonal_order']
    print(f"  Model: SARIMA{order}x{seasonal_order}")
    
    # Backtest settings
    warmup_days = config['backtest']['warmup_days'] # Not strictly used if we fix Train split, but useful for minimum history
    stride = config['backtest']['stride_hours']
    horizon = config['backtest']['horizon_hours']
    
    # We need to backtest on Dev AND Test
    # Strategy: Expanding origin.
    # For Dev: Start training on Train, predict first 24h of Dev. Then add 24h to Train, predict next 24h.
    # For Test: Continue expanding through Dev into Test.
    
    # Actually, let's do two separate passes or one continuous pass?
    # "Create per-country CSVs: outputs/<CC>_forecasts_dev.csv and outputs/<CC>_forecasts_test.csv"
    
    # Let's define a generic backtest function
    def run_backtest(history_series, target_series, name):
        results = []
        history = history_series.copy()
        
        # Pre-fit model on history?
        # Refitting every step is slow. 
        # Strategy: Fit once on history, then filter/update?
        # Or refit every N steps?
        # User prompt doesn't specify refit frequency for backtest, but "Live" has specific adaptation.
        # For "Backtest", usually we want accuracy. 
        # Let's try to fit initially, and then use `apply` (filter) to update state for new observations without full re-estimation of params, 
        # OR re-estimate every X steps. 
        # Given the constraints and "Day-ahead" nature, let's try to re-estimate every 24h (stride) if possible, 
        # OR just re-filter. Re-filtering is much faster.
        # Let's do: Fit on initial history. For each step, append new observation, update state, forecast.
        # Re-fit parameters every 7 days? Or just once?
        # Let's Fit ONCE on initial history for the Dev set. 
        # Then for Test set, Fit ONCE on Train+Dev.
        
        print(f"  Running backtest for {name}...")
        model = SARIMAX(history, order=order, seasonal_order=seasonal_order, 
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        
        # Iterate through target series with stride
        for i in range(0, len(target_series), stride):
            # Current forecast horizon
            current_target = target_series.iloc[i : i+horizon]
            if len(current_target) < horizon:
                break
            
            # Forecast
            # We need to forecast from the end of current history
            # If i > 0, we need to update the model with the data we just stepped over
            if i > 0:
                new_obs = target_series.iloc[i-stride : i]
                model_fit = model_fit.append(new_obs, refit=False) # refit=False for speed
            
            fc = model_fit.get_forecast(steps=horizon)
            yhat = fc.predicted_mean
            conf_int = fc.conf_int(alpha=0.2) # 80% PI -> alpha=0.2
            
            # Collect results
            for h in range(horizon):
                timestamp = current_target.index[h]
                results.append({
                    'timestamp': timestamp,
                    'y_true': current_target.iloc[h],
                    'yhat': yhat.iloc[h],
                    'lo': conf_int.iloc[h, 0],
                    'hi': conf_int.iloc[h, 1],
                    'horizon': h + 1,
                    'train_end': history.index[-1] if i==0 else target_series.index[i-1] # Approx
                })
                
        return pd.DataFrame(results)

    # Run for Dev
    dev_results = run_backtest(train_data, dev_data, "Dev")
    dev_results.to_csv(f'outputs/{country}_forecasts_dev.csv', index=False)
    
    # Run for Test
    # For Test, history is Train + Dev
    train_dev_data = pd.concat([train_data, dev_data])
    test_results = run_backtest(train_dev_data, test_data, "Test")
    test_results.to_csv(f'outputs/{country}_forecasts_test.csv', index=False)
    
    # --- LSTM Section ---
    print(f"  Training LSTM for {country}...")
    try:
        from src.lstm_model import train_lstm, predict_lstm
    except ImportError:
        from lstm_model import train_lstm, predict_lstm
    
    # Train on Train, Validate on Dev
    lstm_model, mean, std = train_lstm(train_data, dev_data, epochs=20) # Short epochs for demo
    
    # Backtest LSTM on Test set
    # Expanding window for LSTM? Or just rolling?
    # "Backtest: expanding origin"
    # For LSTM, usually we train once and then predict rolling.
    # Let's do rolling prediction on Test set.
    
    print(f"  Running LSTM backtest on Test set...")
    lstm_results = []
    history = train_dev_data.copy()
    target_series = test_data
    stride = config['backtest']['stride_hours']
    horizon = config['backtest']['horizon_hours']
    input_window = 168
    
    for i in range(0, len(target_series), stride):
        current_target = target_series.iloc[i : i+horizon]
        if len(current_target) < horizon:
            break
            
        # Update history
        current_history = pd.concat([history, target_series.iloc[:i]])
        
        yhat = predict_lstm(lstm_model, current_history, mean, std, input_window, horizon)
        
        for h in range(horizon):
            lstm_results.append({
                'timestamp': current_target.index[h],
                'y_true': current_target.iloc[h],
                'yhat': yhat[h],
                'lo': np.nan, # No PI for simple LSTM
                'hi': np.nan,
                'horizon': h + 1
            })
            
    lstm_df = pd.DataFrame(lstm_results)
    lstm_df.to_csv(f'outputs/{country}_forecasts_test_lstm.csv', index=False)
    
    return dev_results, test_results, lstm_df

def evaluate_forecasts(df_results, train_data, country, dataset_name, model_name="SARIMA"):
    # Calculate metrics
    y_true = df_results['y_true']
    y_pred = df_results['yhat']
    lower = df_results['lo']
    upper = df_results['hi']
    
    metrics = calculate_metrics(y_true, y_pred, train_data, lower, upper)
    metrics['Country'] = country
    metrics['Dataset'] = dataset_name
    metrics['Model'] = model_name
    return metrics

def run_forecasting_pipeline(config):
    orders = load_model_orders()
    if orders is None:
        print("Model orders not found. Running EDA first...")
        # Optional: trigger EDA here? Or just return.
        # Let's assume user runs EDA.
        return
    
    countries = config['data']['countries']
    all_metrics = []
    
    for country in countries:
        file_path = f'data/{country}.csv'
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Run backtest
        dev_res, test_res, lstm_res = backtest_country(df, country, orders, config)
        
        # Evaluate SARIMA Dev
        n = len(df)
        train_end = int(n * 0.8)
        train_data = df['load'].iloc[:train_end]
        
        m_dev = evaluate_forecasts(dev_res, train_data, country, "Dev", "SARIMA")
        all_metrics.append(m_dev)
        
        # Evaluate SARIMA Test
        train_dev_data = df['load'].iloc[:int(n * 0.9)]
        m_test = evaluate_forecasts(test_res, train_dev_data, country, "Test", "SARIMA")
        all_metrics.append(m_test)
        
        # Evaluate LSTM Test
        m_lstm = evaluate_forecasts(lstm_res, train_dev_data, country, "Test", "LSTM")
        all_metrics.append(m_lstm)
        
    # Create comparison table
    metrics_df = pd.DataFrame(all_metrics)
    print("\nMetrics Summary:")
    print(metrics_df)
    metrics_df.to_csv('outputs/metrics_summary.csv', index=False)

if __name__ == "__main__":
    config = load_config()
    run_forecasting_pipeline(config)
