import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_window=168, output_window=24):
        self.series = torch.FloatTensor(series).unsqueeze(1) # [L, 1]
        self.input_window = input_window
        self.output_window = output_window
        
    def __len__(self):
        return len(self.series) - self.input_window - self.output_window + 1
    
    def __getitem__(self, idx):
        x = self.series[idx : idx + self.input_window]
        y = self.series[idx + self.input_window : idx + self.input_window + self.output_window]
        return x, y

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=24):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: [batch, input_window, input_size]
        out, _ = self.lstm(x)
        # Take last time step output
        last_out = out[:, -1, :]
        # Predict all 24 steps at once (Direct Multi-horizon)
        prediction = self.fc(last_out)
        return prediction.unsqueeze(2) # [batch, output_window, 1] if we want to match shape, but fc output is [batch, 24]

def train_lstm(train_series, val_series, input_window=168, output_window=24, epochs=10, batch_size=32, lr=0.001):
    # Normalize
    mean = train_series.mean()
    std = train_series.std()
    train_norm = (train_series - mean) / std
    val_norm = (val_series - mean) / std
    
    train_dataset = TimeSeriesDataset(train_norm.values, input_window, output_window)
    val_dataset = TimeSeriesDataset(val_norm.values, input_window, output_window)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMForecaster(output_size=output_window).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"  Training LSTM on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            # Output from FC is [batch, 24], y is [batch, 24, 1]
            target = y.squeeze(-1)
            # Ensure shapes match
            if output.shape != target.shape:
                print(f"Shape mismatch: output {output.shape}, target {target.shape}")
                # Try to reshape target?
                target = target.view(output.shape)
                
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                target = y.squeeze(-1)
                if output.shape != target.shape:
                    target = target.view(output.shape)
                loss = criterion(output, target)
                val_loss += loss.item()
                
        if (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f}")
            
    return model, mean, std

def predict_lstm(model, history_series, mean, std, input_window=168, output_window=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Prepare input
    last_window = history_series.iloc[-input_window:].values
    last_window_norm = (last_window - mean) / std
    x = torch.FloatTensor(last_window_norm).unsqueeze(0).unsqueeze(2).to(device) # [1, 168, 1]
    
    with torch.no_grad():
        out_norm = model(x) # [1, 24]
        
    out_norm = out_norm.cpu().numpy().flatten()
    out_denorm = out_norm * std + mean
    return out_denorm
