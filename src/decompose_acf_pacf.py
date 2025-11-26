import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import os
import yaml
import warnings

warnings.filterwarnings("ignore")

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def check_stationarity(series, name):
    result = adfuller(series.dropna())
    print(f"  ADF Statistic ({name}): {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    return result[1] < 0.05

def perform_eda_and_selection(config):
    countries = config['data']['countries']
    seasonality = config['modeling']['seasonality']
    os.makedirs('outputs', exist_ok=True)
    
    model_orders = {}

    for country in countries:
        file_path = f'data/{country}.csv'
        if not os.path.exists(file_path):
            print(f"Skipping {country}, file not found.")
            continue
            
        print(f"\nAnalyzing {country}...")
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        series = df['load']
        
        # 1.3 Basic sanity plot (last 14 days)
        last_14d = series.iloc[-14*24:]
        plt.figure(figsize=(12, 6))
        plt.plot(last_14d)
        plt.title(f'{country} Load - Last 14 Days')
        plt.xlabel('Timestamp')
        plt.ylabel('Load')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'outputs/{country}_sanity_14d.png')
        plt.close()
        
        # 1.4 Decomposition & Seasonality
        # i. STL
        # Using a subset for clearer STL visualization if series is too long, 
        # but STL can handle long series. Let's use the last year for STL to be representative but not overwhelming?
        # Or just the whole thing. STL is fast enough.
        stl = STL(series, period=seasonality, robust=True)
        res = stl.fit()
        
        fig = res.plot()
        fig.set_size_inches(12, 10)
        plt.suptitle(f'STL Decomposition - {country}')
        plt.tight_layout()
        plt.savefig(f'outputs/{country}_stl.png')
        plt.close()
        
        # ii. Stationarity and differencing
        # Check original
        is_stationary = check_stationarity(series, "Original")
        d = 0
        D = 0
        
        diff_series = series.copy()
        
        if not is_stationary:
            print("  Series is not stationary. Trying d=1...")
            diff_series = series.diff().dropna()
            if check_stationarity(diff_series, "Differenced (d=1)"):
                d = 1
            else:
                print("  Still not stationary (or maybe seasonal?).")
        
        # Check seasonal differencing
        # Prompt says: "if strong daily seasonality, try seasonal differencing D=1 with s=24"
        # We can look at the seasonal component range vs residual, or just try D=1.
        # Let's apply D=1 if we suspect seasonality (which we do for power data).
        print("  Applying seasonal differencing D=1 (s=24)...")
        diff_seasonal = diff_series.diff(seasonality).dropna()
        D = 1
        
        # iii. ACF/PACF
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        plot_acf(diff_seasonal, ax=ax[0], lags=48, title=f'ACF - {country} (d={d}, D={D})')
        plot_pacf(diff_seasonal, ax=ax[1], lags=48, title=f'PACF - {country} (d={d}, D={D})')
        plt.tight_layout()
        plt.savefig(f'outputs/{country}_acf_pacf.png')
        plt.close()
        
        # iv. Information criteria (AIC/BIC) - Grid Search
        print("  Running SARIMA Grid Search (BIC)...")
        # Prompt: (p,q) in {0,1,2}, d in {0,1}; (P,Q) in {0,1}, D in {0,1}, s=24
        # We already estimated d and D, but let's let auto_arima confirm or search within the small grid.
        # Actually, prompt says "search a small SARIMA grid".
        # Let's use auto_arima with the specific constraints.
        
        # To save time and memory, use a smaller representative subset
        # Last 30 days (720 hours) should be enough for determining seasonality/trend orders
        train_subset = series.iloc[-24*30:] 
        
        model = pm.auto_arima(train_subset,
                              start_p=0, max_p=2,
                              start_q=0, max_q=2,
                              d=None, max_d=1, 
                              start_P=0, max_P=1,
                              start_Q=0, max_Q=1,
                              D=None, max_D=1, 
                              m=seasonality,
                              seasonal=True,
                              stepwise=False, # Grid search
                              n_jobs=1, # Avoid MemoryError
                              information_criterion='bic',
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True)
        
        print(f"  Best Model for {country}: {model.order} {model.seasonal_order} BIC={model.bic()}")
        model_orders[country] = {
            'order': model.order,
            'seasonal_order': model.seasonal_order,
            'bic': model.bic()
        }

    # v. Document chosen orders
    with open('outputs/selected_models.txt', 'w') as f:
        for country, info in model_orders.items():
            f.write(f"{country}:\n")
            f.write(f"  Order: {info['order']}\n")
            f.write(f"  Seasonal Order: {info['seasonal_order']}\n")
            f.write(f"  BIC: {info['bic']}\n\n")
    print("Model selection completed. Results saved to outputs/selected_models.txt")

if __name__ == "__main__":
    config = load_config()
    perform_eda_and_selection(config)
