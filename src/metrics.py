import numpy as np

def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=24):
    n = len(y_train)
    d = np.abs(np.diff(y_train, n=seasonality)).sum() / (n - seasonality)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

def coverage_80(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper)) * 100

def calculate_metrics(y_true, y_pred, y_train=None, lower=None, upper=None, seasonality=24):
    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'sMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred)
    }
    
    if y_train is not None:
        metrics['MASE'] = mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality)
        
    if lower is not None and upper is not None:
        metrics['Coverage_80'] = coverage_80(y_true, lower, upper)
        
    return metrics
