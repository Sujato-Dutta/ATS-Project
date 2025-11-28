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
        
        # Ensure we have enough data
        if len(df) < 2000:
            print(f"Not enough data for {country}. Skipping.")
            continue
            
        # Take the last 168 hours (1 week) for the "forecast" demo
        # Actually, let's take a bit more to show a trend, maybe 2 weeks (336 hours)
        test_df = df.iloc[-336:].copy()
        
        # Assuming the column name is the country code or 'load' or similar. 
        # Let's check the first column.
        target_col = test_df.columns[0]
        actuals = test_df[target_col].values
        
        # Generate mock forecasts
        # SARIMA: Good at capturing seasonality, maybe slightly smoother
        noise_sarima = np.random.normal(0, actuals.std() * 0.05, size=len(actuals))
        sarima_forecast = actuals + noise_sarima
        
        # LSTM: Might capture non-linearities but maybe slightly more erratic or smoothed depending on training
        # Let's make it slightly smoothed version of actuals + noise
        noise_lstm = np.random.normal(0, actuals.std() * 0.07, size=len(actuals))
        lstm_forecast = actuals * 0.95 + np.roll(actuals, 1) * 0.05 + noise_lstm # Slight lag/smoothing
        
        # GRU: Similar to LSTM
        noise_gru = np.random.normal(0, actuals.std() * 0.06, size=len(actuals))
        gru_forecast = actuals * 0.98 + noise_gru
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'timestamp': test_df.index,
            'actual': actuals,
            'sarima': sarima_forecast,
            'lstm': lstm_forecast,
            'gru': gru_forecast
        })
        
        output_file = os.path.join(output_dir, f'{country}_mock_forecasts.csv')
        result_df.to_csv(output_file, index=False)
        print(f"Saved mock forecasts to {output_file}")

if __name__ == "__main__":
    generate_mock_data()
