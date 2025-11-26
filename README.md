# OPSD PowerDesk Assignment: Day-Ahead Load Forecasting

## Overview
This project implements a robust day-ahead electric load forecasting system for **Germany (DE)**, **France (FR)**, and **Great Britain (GB)** using Open Power System Data (OPSD). The system is designed to handle the full lifecycle of a forecasting product: from strict data ingestion and exploratory analysis to advanced modeling (SARIMA, LSTM), anomaly detection, and simulated live deployment with online adaptation.

## Key Features

### 1. Data Pipeline
- **Strict Ingestion**: Automated downloading and preprocessing of OPSD time series data.
- **Cleaning**: Handling of missing values, timestamp alignment, and column standardization.
- **Tidy Storage**: Processed data saved as per-country CSVs for efficient access.

### 2. Exploratory Data Analysis (EDA)
- **Decomposition**: STL decomposition (Seasonal-Trend-Loess) to isolate seasonal and trend components.
- **Stationarity**: ADF tests and differencing strategies ($d=1, D=1$).
- **Correlation**: ACF/PACF analysis to determine lag structures.
- **Model Selection**: Automated grid search for SARIMA orders using AIC/BIC information criteria.

### 3. Forecasting Models
- **SARIMA**: Seasonal AutoRegressive Integrated Moving Average with exogenous variables capability.
- **LSTM**: Long Short-Term Memory neural networks for capturing complex non-linear dependencies (168h input $\to$ 24h output).
- **Backtesting**: Expanding window validation with separate Dev and Test sets.
- **Metrics**: MASE, sMAPE, RMSE, MAPE, and 80% Prediction Interval Coverage.

### 4. Anomaly Detection
- **Statistical**: Residual Z-score detection (Rolling window = 336h) and CUSUM for drift monitoring.
- **Machine Learning**: Random Forest classifier trained on "silver labels" (derived from statistical thresholds) to verify anomalies, reporting PR-AUC and F1 scores.

### 5. Live Simulation & Adaptation
- **Simulation**: Replays historical data hour-by-hour to mimic a live production environment.
- **Online Adaptation**: Implements a **Rolling SARIMA** strategy that refits daily (00:00 UTC) and triggers immediate updates upon detecting concept drift (EWMA of Z-scores).

### 6. Interactive Dashboard
- **Streamlit UI**: A comprehensive dashboard to visualize:
    - Live forecast cones vs. actuals.
    - Detected anomalies and residual heatmaps.
    - Real-time KPIs (Rolling 7-day MASE, Coverage).
    - Simulation update logs.

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

### Step 1: Data Preparation
Download and process the raw OPSD data:
```bash
python src/load_opsd.py
```

### Step 2: EDA & Model Selection
Generate analysis plots and determine optimal SARIMA parameters:
```bash
python src/decompose_acf_pacf.py
```

### Step 3: Forecasting & Evaluation
Train models, run backtests, and generate performance metrics:
```bash
python src/forecast.py
```

### Step 4: Anomaly Detection
Run statistical detection and train the ML anomaly classifier:
```bash
python src/anomaly.py
python src/anomaly_ml.py
```

### Step 5: Live Simulation
Execute the live simulation loop (default: Germany, 2000+ hours):
```bash
python src/live_loop.py
```

### Step 6: Launch Dashboard
Start the interactive dashboard to view results:
```bash
streamlit run src/dashboard.py
```

## Project Structure

```
ATS Project/
├── data/                   # Processed data files (DE.csv, FR.csv, GB.csv)
├── outputs/                # Generated plots, forecasts, logs, and metrics
├── src/
│   ├── load_opsd.py        # Data ingestion script
│   ├── decompose_acf_pacf.py # EDA and model selection
│   ├── forecast.py         # SARIMA and LSTM forecasting pipeline
│   ├── lstm_model.py       # LSTM model definition (PyTorch)
│   ├── metrics.py          # Evaluation metrics implementation
│   ├── anomaly.py          # Statistical anomaly detection
│   ├── anomaly_ml.py       # ML-based anomaly classification
│   ├── live_loop.py        # Live simulation and adaptation logic
│   └── dashboard.py        # Streamlit dashboard
├── config.yaml             # Configuration parameters
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Configuration
Modify `config.yaml` to adjust parameters such as:
- **Countries**: List of countries to process.
- **Backtesting**: Split ratios and window sizes.
- **Anomaly**: Z-score thresholds and windows.
- **Simulation**: Adaptation strategies and drift sensitivity.
