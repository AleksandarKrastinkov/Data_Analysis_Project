import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# Create output folder for images
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Helper function to save figures to the output directory
def save_figure(filename):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
# File paths
timestamps_file = 'energy_comsumption_timestamps.csv'
readings_file = 'energy_comsumption_readings.csv'

# Load the data
timestamps_df = pd.read_csv(timestamps_file)
readings_df = pd.read_csv(readings_file)

# Display the first few rows of each dataframe
print("Timestamps data:")
print(timestamps_df.head())
print("\nReadings data:")
print(readings_df.head())

# Check for missing values
print("\nMissing values in timestamps data:", timestamps_df.isnull().sum().sum())
print("Missing values in readings data:", readings_df.isnull().sum().sum())

# Data cleaning and preparation
# Merge the two dataframes on timestamp_id and reading_id
merged_df = pd.merge(
    timestamps_df, 
    readings_df, 
    left_on='timestamp_id', 
    right_on='reading_id', 
    how='inner'
)

# Create datetime column by combining date and time
merged_df['datetime'] = pd.to_datetime(merged_df['date'] + ' ' + merged_df['time'])

# Filter out invalid measurements (less than 0.5 or NaN)
merged_df_clean = merged_df.dropna(subset=['consumption'])
merged_df_clean = merged_df_clean[merged_df_clean['consumption'] >= 0.5]

print(f"\nOriginal data points: {len(merged_df)}")
print(f"Clean data points: {len(merged_df_clean)}")

# Sort by datetime
merged_df_clean = merged_df_clean.sort_values('datetime')

# Set datetime as index
time_series = merged_df_clean.set_index('datetime')['consumption']

# Resample to regular intervals (2 hours)
resampled_ts = time_series.resample('2h').mean()  # Using 2h instead of 2H to avoid warning

# Fill missing values using forward fill (or interpolation)
resampled_ts = resampled_ts.interpolate(method='linear')

# Plot the original time series
plt.figure(figsize=(15, 6))
plt.plot(time_series, label='Original data')
plt.plot(resampled_ts, label='Resampled data (2H)', alpha=0.7)
plt.title('Energy Consumption Time Series')
plt.xlabel('Date')
plt.ylabel('Consumption (KWH)')
plt.legend()
save_figure('energy_consumption_ts.png')

# Analyze for seasonality and trends
# Decompose into trend, seasonal, and residual components if data covers enough time
if len(resampled_ts) > 24*7:  # At least a week of 2-hourly data
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(resampled_ts, model='additive', period=12)  # 12 periods = 24 hours
        
        plt.figure(figsize=(15, 12))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('Observed')
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('Trend')
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('Seasonality')
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('Residuals')
        plt.tight_layout()
        save_figure('seasonal_decomposition.png')
    except:
        print("Couldn't perform seasonal decomposition, may need more data points.")

# Prepare data for LSTM
# Normalize the data
data_values = resampled_ts.values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data_values)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Define sequence length (window size)
seq_length = 24  # 24 time steps (2 days with 2-hour intervals)

# Create sequences
X, y = create_sequences(normalized_data, seq_length)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Split into train and test sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Define the LSTM model
class EnergyConsumptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        # Take only the last output and apply linear layer
        x = self.linear(x[:, -1, :])
        return x

# Initialize model and optimizer
model = EnergyConsumptionModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

# Training
n_epochs = 500  # Reduced from 2000 for quicker execution
losses = []

print("\nTraining the LSTM model...")
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        # Reshape y_batch to match y_pred dimensions
        y_batch = y_batch.squeeze(2)  # Remove the last dimension to match y_pred
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    if epoch % 100 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        # Reshape y_train to match y_pred dimensions
        y_train_reshaped = y_train.squeeze(2)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train_reshaped))
        y_pred = model(X_test)
        # Reshape y_test to match y_pred dimensions
        y_test_reshaped = y_test.squeeze(2)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test_reshaped))
    print(f"Epoch {epoch}: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}")

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('MSE Loss')
save_figure('training_loss.png')

# Evaluate the model
model.eval()
with torch.no_grad():
    # Make predictions
    train_predict = model(X_train).detach().numpy()
    test_predict = model(X_test).detach().numpy()
    
    # Reshape for inverse transform (scaler expects 2D arrays)
    train_predict = train_predict.reshape(-1, 1)
    test_predict = test_predict.reshape(-1, 1)
    y_train_np = y_train.detach().numpy().reshape(-1, 1)
    y_test_np = y_test.detach().numpy().reshape(-1, 1)
    
    # Inverse transform
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train_np)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test_np)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_predict))
    test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_predict))
    
    print(f"\nFinal Train RMSE: {train_rmse:.4f}")
    print(f"Final Test RMSE: {test_rmse:.4f}")

# Visualize predictions
# Create a timeline for plotting
train_timeline = resampled_ts.index[seq_length:seq_length + len(train_predict)]
test_timeline = resampled_ts.index[seq_length + len(train_predict):seq_length + len(train_predict) + len(test_predict)]

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(train_timeline, y_train_inv, label='Train Actual')
plt.plot(train_timeline, train_predict, label='Train Predicted')
plt.plot(test_timeline, y_test_inv, label='Test Actual')
plt.plot(test_timeline, test_predict, label='Test Predicted')
plt.title('Energy Consumption Prediction')
plt.xlabel('Date')
plt.ylabel('Consumption (KWH)')
plt.legend()
save_figure('energy_prediction_results.png')

# Save the model
model_path = os.path.join(output_dir, 'energy_prediction_model.pth')
torch.save(model.state_dict(), model_path)
print(f"\nModel saved to {model_path}")

# BONUS SECTION: Alternative Models
print("\n--- BONUS: Testing additional models ---")

# Convert data for scikit-learn models
X_train_2d = X_train.reshape(X_train.shape[0], -1).numpy()
X_test_2d = X_test.reshape(X_test.shape[0], -1).numpy()
y_train_1d = y_train.reshape(-1).numpy()
y_test_1d = y_test.reshape(-1).numpy()

# 1. Random Forest
try:
    from sklearn.ensemble import RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_2d, y_train_1d)
    
    rf_train_pred = rf_model.predict(X_train_2d).reshape(-1, 1)
    rf_test_pred = rf_model.predict(X_test_2d).reshape(-1, 1)
    
    # Inverse transform predictions
    rf_train_pred = scaler.inverse_transform(rf_train_pred)
    rf_test_pred = scaler.inverse_transform(rf_test_pred)
    
    # Calculate RMSE
    rf_train_rmse = np.sqrt(mean_squared_error(y_train_inv, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test_inv, rf_test_pred))
    
    print("\nRandom Forest Results:")
    print(f"Train RMSE: {rf_train_rmse:.4f}")
    print(f"Test RMSE: {rf_test_rmse:.4f}")
    
    # Plot Random Forest results
    plt.figure(figsize=(15, 6))
    plt.plot(test_timeline, y_test_inv, label='Actual')
    plt.plot(test_timeline, test_predict, label='LSTM Predictions')
    plt.plot(test_timeline, rf_test_pred, label='Random Forest Predictions')
    plt.title('Energy Consumption Prediction Comparison - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Consumption (KWH)')
    plt.legend()
    save_figure('model_comparison_rf.png')
except Exception as e:
    print(f"Could not run Random Forest model: {str(e)}")

# 2. XGBoost
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train_2d, y_train_1d)
    
    xgb_train_pred = xgb_model.predict(X_train_2d).reshape(-1, 1)
    xgb_test_pred = xgb_model.predict(X_test_2d).reshape(-1, 1)
    
    # Inverse transform predictions
    xgb_train_pred = scaler.inverse_transform(xgb_train_pred)
    xgb_test_pred = scaler.inverse_transform(xgb_test_pred)
    
    # Calculate RMSE
    xgb_train_rmse = np.sqrt(mean_squared_error(y_train_inv, xgb_train_pred))
    xgb_test_rmse = np.sqrt(mean_squared_error(y_test_inv, xgb_test_pred))
    
    print("\nXGBoost Results:")
    print(f"Train RMSE: {xgb_train_rmse:.4f}")
    print(f"Test RMSE: {xgb_test_rmse:.4f}")
    
    # Plot XGBoost results
    plt.figure(figsize=(15, 6))
    plt.plot(test_timeline, y_test_inv, label='Actual')
    plt.plot(test_timeline, test_predict, label='LSTM Predictions')
    plt.plot(test_timeline, xgb_test_pred, label='XGBoost Predictions')
    plt.title('Energy Consumption Prediction Comparison - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Consumption (KWH)')
    plt.legend()
    save_figure('model_comparison_xgb.png')
except Exception as e:
    print(f"Could not run XGBoost model: {str(e)}")

# 3. ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings("ignore")
    
    # Prepare a simplified version for ARIMA (no sequencing needed)
    # Use non-normalized data for ARIMA
    train_data = resampled_ts.iloc[:int(len(resampled_ts)*0.8)]
    test_data = resampled_ts.iloc[int(len(resampled_ts)*0.8):]
    
    # Fit ARIMA model
    arima_model = ARIMA(train_data, order=(5,1,0))
    arima_fit = arima_model.fit()
    
    # Forecast
    arima_forecast = arima_fit.forecast(steps=len(test_data))
    
    # Calculate RMSE
    arima_rmse = np.sqrt(mean_squared_error(test_data, arima_forecast))
    
    print("\nARIMA Results:")
    print(f"Test RMSE: {arima_rmse:.4f}")
    
    # Plot ARIMA results
    plt.figure(figsize=(15, 6))
    plt.plot(test_data.index, test_data, label='Actual')
    plt.plot(test_data.index, arima_forecast, label='ARIMA Predictions')
    plt.title('Energy Consumption Prediction with ARIMA - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Consumption (KWH)')
    plt.legend()
    save_figure('model_comparison_arima.png')
    
    # Combined model comparison
    plt.figure(figsize=(15, 6))
    plt.plot(test_timeline, y_test_inv, label='Actual')
    plt.plot(test_timeline, test_predict, label='LSTM')
    
    # Try to align ARIMA predictions with test_timeline
    if len(test_timeline) == len(arima_forecast):
        plt.plot(test_timeline, arima_forecast, label='ARIMA')
    
    try:
        plt.plot(test_timeline, rf_test_pred, label='Random Forest')
    except:
        pass
    
    try:
        plt.plot(test_timeline, xgb_test_pred, label='XGBoost')
    except:
        pass
    
    plt.title('Model Comparison - Test Set')
    plt.xlabel('Date')
    plt.ylabel('Consumption (KWH)')
    plt.legend()
    save_figure('model_comparison_all.png')
except Exception as e:
    print(f"Could not run ARIMA model: {str(e)}")

print(f"\nAll output images have been saved to the '{output_dir}' folder.") 