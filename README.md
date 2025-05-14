# Energy Consumption Prediction

This project analyzes and predicts energy consumption from a heating device using time series data. The data comes from Smart sensors with timestamps and energy consumption readings.

## Project Overview

1. **Data Cleaning**: Removing invalid readings (< 0.5 KWH or empty values)
2. **Time Series Analysis**: Examining seasonality and trends in energy consumption
3. **Feature Engineering**: Creating regular time intervals and time windows for prediction
4. **Model Implementation**: Using LSTM neural network for primary prediction
5. **Model Evaluation**: Comparing actual vs predicted values
6. **Bonus Models**: Comparing LSTM performance with Random Forest, XGBoost, and ARIMA models

## Requirements

The required packages are listed in `requirements.txt`. Install them using:

```
pip install -r requirements.txt
```

## Running the Code

To run the analysis and prediction:

```
python energy_prediction.py
```

## Output

The script generates several visualizations:
- `energy_consumption_ts.png`: Original and resampled time series
- `seasonal_decomposition.png`: Trend, seasonality, and residual components
- `training_loss.png`: LSTM training loss
- `energy_prediction_results.png`: LSTM prediction results
- Model comparison charts (if additional models are installed)

## Data Files

The script expects two CSV files:
- `energy_comsumption_timestamps.csv`: Contains timestamp_id, date, and time columns
- `energy_comsumption_readings.csv`: Contains reading_id and consumption columns

## Model

The primary model is an LSTM neural network for time series forecasting. The trained model is saved as `energy_prediction_model.pth`. 