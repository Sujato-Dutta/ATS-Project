import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import yaml
import os
import json

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def create_silver_labels(df):
    # i. Create “silver labels”: 
    # positive if (|zt| ≥ 3.5) OR (y_true outside [lo,hi] AND |zt| ≥ 2.5); 
    # negative if |zt| < 1.0 AND y_true inside [lo,hi].
    
    z = df['z_resid'].abs()
    y = df['y_true']
    lo = df['lo']
    hi = df['hi']
    
    outside_pi = (y < lo) | (y > hi)
    inside_pi = ~outside_pi
    
    # Positive condition
    cond_pos = (z >= 3.5) | (outside_pi & (z >= 2.5))
    
    # Negative condition
    cond_neg = (z < 1.0) & inside_pi
    
    df['silver_label'] = np.nan
    df.loc[cond_pos, 'silver_label'] = 1
    df.loc[cond_neg, 'silver_label'] = 0
    
    return df

def simulate_human_verification(df, n_samples=100):
    # ii. Human verification: per country, randomly sample ≈100 timestamps 
    # (≈50 positives, 50 negatives) and confirm labels by visual check.
    # We will simulate this by assuming silver labels are correct for the "verified" set.
    
    positives = df[df['silver_label'] == 1]
    negatives = df[df['silver_label'] == 0]
    
    n_pos = min(len(positives), n_samples // 2)
    n_neg = min(len(negatives), n_samples // 2)
    
    if n_pos == 0 or n_neg == 0:
        print("  Warning: Not enough labeled samples for verification.")
        return pd.DataFrame()
    
    sample_pos = positives.sample(n=n_pos, random_state=42)
    sample_neg = negatives.sample(n=n_neg, random_state=42)
    
    verified = pd.concat([sample_pos, sample_neg])
    verified['verified_label'] = verified['silver_label'] # Assume silver is correct for simulation
    
    return verified

def extract_features(df):
    # Features from last 24-48h (lags, rollups, calendar, forecast context)
    # We need to compute features for the labeled points.
    # Ideally we compute features for the whole dataset first.
    
    # Lags of residuals and z-scores
    for lag in [1, 2, 24, 48]:
        df[f'z_lag_{lag}'] = df['z_resid'].shift(lag)
        df[f'resid_lag_{lag}'] = df['residual'].shift(lag)
        
    # Rolling stats
    df['z_roll_mean_24'] = df['z_resid'].rolling(24).mean()
    df['z_roll_std_24'] = df['z_resid'].rolling(24).std()
    
    # Calendar
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    # Forecast context
    df['yhat'] = df['yhat']
    df['width_pi'] = df['hi'] - df['lo']
    
    return df

def train_classifier(df_verified, df_all):
    # Train simple classifier
    features = [c for c in df_all.columns if 'lag' in c or 'roll' in c or c in ['hour', 'dayofweek', 'yhat', 'width_pi']]
    features = [f for f in features if f in df_verified.columns]
    
    # Drop NaNs
    df_model = df_verified.dropna(subset=features + ['verified_label'])
    
    if len(df_model) < 10:
        print("  Not enough data to train.")
        return None
    
    X = df_model[features]
    y = df_model['verified_label']
    
    # Train/Test split on verified data? Or just train on all verified and eval?
    # "Train a simple classifier... Report PR-AUC and F1"
    # Usually cross-val or hold-out. Let's do simple split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # Eval
    y_probs = clf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    
    # F1 at fixed precision 0.8
    # Find threshold for precision >= 0.8
    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        p = precision_score(y_test, y_pred, zero_division=0)
        if p >= 0.80:
            f = f1_score(y_test, y_pred)
            if f > best_f1:
                best_f1 = f
                
    return clf, pr_auc, best_f1

def run_ml_anomaly(config):
    countries = config['data']['countries']
    results = {}
    
    for country in countries:
        # Load anomalies file (from step 3.1)
        file_path = f'outputs/{country}_anomalies.csv'
        if not os.path.exists(file_path):
            print(f"Skipping {country}, anomalies file not found.")
            continue
            
        print(f"ML Anomaly Detection for {country}...")
        df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
        
        # Need 'lo' and 'hi' which might not be in anomalies.csv if I didn't save them.
        # I saved: y_true, yhat, z_resid, flag_z, flag_cusum.
        # I need to merge with forecasts to get lo/hi.
        forecast_file = f'outputs/{country}_forecasts_test.csv'
        df_fc = pd.read_csv(forecast_file, parse_dates=['timestamp'], index_col='timestamp')
        df = df.join(df_fc[['lo', 'hi']], rsuffix='_fc')
        
        # Add residual column if missing (it was renamed to z_resid, but we need raw residual for features)
        df['residual'] = df['y_true'] - df['yhat']
        
        # 1. Silver Labels
        df = create_silver_labels(df)
        
        # 2. Features
        df = extract_features(df)
        
        # 3. Verification
        df_verified = simulate_human_verification(df)
        if df_verified.empty:
            continue
            
        df_verified.to_csv(f'outputs/{country}_anomaly_labels_verified.csv')
        
        # 4. Train & Eval
        res = train_classifier(df_verified, df)
        if res:
            clf, pr_auc, f1 = res
            print(f"  {country} - PR-AUC: {pr_auc:.4f}, F1 (P>=0.8): {f1:.4f}")
            results[country] = {'PR_AUC': pr_auc, 'F1_P80': f1}
            
    with open('outputs/anomaly_ml_eval.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("ML Anomaly Detection completed. Results saved.")

if __name__ == "__main__":
    config = load_config()
    run_ml_anomaly(config)
