import pandas as pd
import yaml
import os
from datetime import datetime, timedelta

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_country_data(df, country, history_years):
    print(f"Processing {country}...")
    
    # 1.1 Select columns and rename
    # Look for load, wind, solar
    cols_map = {}
    
    # Load
    load_col = f'{country}_load_actual_entsoe_transparency'
    if load_col not in df.columns:
        # Fallback for GB or others
        if country == 'GB':
             possible = ['GB_GBN_load_actual_entsoe_transparency', 'GB_UKM_load_actual_entsoe_transparency']
             for p in possible:
                 if p in df.columns:
                     load_col = p
                     break
        else:
             load_col = f'{country}_load_actual_entsoe_power_statistics'
    
    if load_col in df.columns:
        cols_map[load_col] = 'load'
    else:
        print(f"  Warning: No load column found for {country}")
        return None

    # Wind (Optional but requested)
    wind_col = f'{country}_wind_generation_actual'
    if wind_col in df.columns:
        cols_map[wind_col] = 'wind'
        
    # Solar (Optional but requested)
    solar_col = f'{country}_solar_generation_actual'
    if solar_col in df.columns:
        cols_map[solar_col] = 'solar'
        
    # Create tidy DF
    df_country = df[list(cols_map.keys())].copy()
    df_country.rename(columns=cols_map, inplace=True)
    df_country.index.name = 'timestamp' # Rename index to timestamp
    
    # Drop rows with missing load
    df_country.dropna(subset=['load'], inplace=True)
    
    # Sort by timestamp (already sorted by index usually, but ensure)
    df_country.sort_index(inplace=True)
    
    # Filter for last N years
    end_date = df_country.index.max()
    start_date = end_date - timedelta(days=365 * history_years)
    df_country = df_country[start_date:end_date]
    
    print(f"  Shape: {df_country.shape}, Time range: {df_country.index.min()} to {df_country.index.max()}")
    
    return df_country

def download_and_load_data(config):
    url = config['data']['url']
    countries = config['data']['countries']
    history_years = config['data']['history_years']
    
    local_file = 'data/opsd_time_series.csv'
    os.makedirs('data', exist_ok=True)
    
    if not os.path.exists(local_file):
        print(f"Downloading data from {url}...")
        df = pd.read_csv(url, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
        df.to_csv(local_file)
    else:
        print(f"Loading local data from {local_file}...")
        df = pd.read_csv(local_file, parse_dates=['utc_timestamp'], index_col='utc_timestamp')
        
    # 1.2 One tidy DataFrame per country
    for country in countries:
        df_country = process_country_data(df, country, history_years)
        if df_country is not None:
            out_file = f'data/{country}.csv'
            df_country.to_csv(out_file)
            print(f"  Saved {out_file}")

if __name__ == "__main__":
    config = load_config()
    download_and_load_data(config)
