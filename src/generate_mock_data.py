import pandas as pd
import numpy as np
import os

def generate_mock_data():
    data_dir = 'data'
    output_dir = 'mock_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    countries = ['DE', 'FR', 'GB']
    
    for country in countries:
        file_path = os.path.join(data_dir, f'{country}.csv')
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Skipping.")
            continue
            
        print(f"Processing {country}...")
        df = pd.read_csv(file_path, parse_dates=True, index_col=0)
        
        if len(df) < 2000:
            print(f"Not enough data for {country}. Skipping.")
            continue
            
        # Take last 336 hours (2 weeks)
        test_df = df.iloc[-336:].copy()
        target_col = test_df.columns[0]
        actuals = test_df[target_col].values
        
        # 1. Generate Forecasts
        noise_sarima = np.random.normal(0, actuals.std() * 0.05, size=len(actuals))
        sarima_forecast = actuals + noise_sarima
        
        noise_lstm = np.random.normal(0, actuals.std() * 0.07, size=len(actuals))
        lstm_forecast = actuals * 0.95 + np.roll(actuals, 1) * 0.05 + noise_lstm
        
        noise_gru = np.random.normal(0, actuals.std() * 0.06, size=len(actuals))
        gru_forecast = actuals * 0.98 + noise_gru
        
        # 2. Generate Prediction Intervals (Forecast Cone)
        # 80% PI implies some width around the forecast
        # We'll use SARIMA as the "primary" model for the cone
        std_resid = actuals.std() * 0.1
        lower_bound = sarima_forecast - 1.28 * std_resid # 1.28 for 80% approx
        upper_bound = sarima_forecast + 1.28 * std_resid
        
        # 3. Generate Anomalies
        # Randomly flag some hours as anomalies (approx 2% of data)
        flag_z = np.random.choice([0, 1], size=len(actuals), p=[0.98, 0.02])
        # Make sure anomalies actually look like anomalies in the data? 
        # For the demo, we might just highlight them without changing the value, 
        # or we could spike the value. Let's spike the value for visual effect.
        actuals_with_anomalies = actuals.copy()
        for i in range(len(actuals)):
            if flag_z[i] == 1:
                actuals_with_anomalies[i] *= 1.3 # 30% spike
        
        # 4. Create Result DataFrame
        result_df = pd.DataFrame({
            'timestamp': test_df.index,
            'actual': actuals_with_anomalies,
            'sarima': sarima_forecast,
            'lstm': lstm_forecast,
            'gru': gru_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'flag_z': flag_z
        })
        
        output_file = os.path.join(output_dir, f'{country}_mock_forecasts.csv')
        result_df.to_csv(output_file, index=False)
        print(f"Saved mock forecasts to {output_file}")

if __name__ == "__main__":
    generate_mock_data()
