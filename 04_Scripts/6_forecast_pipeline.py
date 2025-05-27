import pandas as pd
import joblib
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load data
train_df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Processed_dataset/train_data.csv", parse_dates=["Date"])
test_df = pd.read_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/Processed_dataset/test_data.csv", parse_dates=["Date"])

# Use store 1 only for simplicity
train_store = train_df[train_df["Store"] == 1].sort_values("Date")
test_store = test_df[test_df["Store"] == 1].sort_values("Date")

# -----------------------------------
# ARIMA MODEL
print("Training ARIMA...")
# Use Weekly_Sales as time series
arima_train = train_store.set_index("Date")["Weekly_Sales"]

# Train ARIMA (order can be tuned)
arima_model = ARIMA(arima_train, order=(5,1,0))
arima_model = arima_model.fit()

# Predict on test dates
arima_preds = arima_model.predict(start=test_store["Date"].min(), end=test_store["Date"].max())

# Align predictions length
arima_preds = arima_preds[:len(test_store)]

# Save ARIMA model
joblib.dump(arima_model,"R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/05_Models/arima_model.pkl")

# -----------------------------------
# PROPHET MODEL
print("Training Prophet...")
prophet_train = train_store[["Date", "Weekly_Sales"]].rename(columns={"Date":"ds", "Weekly_Sales":"y"})

prophet_model = Prophet()
prophet_model.fit(prophet_train)

future = test_store[["Date"]].rename(columns={"Date":"ds"})
prophet_forecast = prophet_model.predict(future)

prophet_preds = prophet_forecast["yhat"].values

# Save Prophet model
joblib.dump(prophet_model,"R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/05_Models/prophet_model.pkl")

# -----------------------------------
# LSTM MODEL
print("Training LSTM...")
# Prepare data: use last 10 days as features to predict next day sales

def create_lstm_dataset(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

sales_series = train_store.sort_values("Date")["Weekly_Sales"].values
X_train, y_train = create_lstm_dataset(sales_series, n_steps=10)

# Reshape for LSTM: [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(10,1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, verbose=0)

# Prepare test input sequences
test_sales_series = np.concatenate([sales_series[-10:], test_store["Weekly_Sales"].values[:-1]])
X_test = []
for i in range(len(test_store)):
    seq = test_sales_series[i:i+10]
    X_test.append(seq)
X_test = np.array(X_test).reshape((len(test_store),10,1))

# Predict test
lstm_preds = model.predict(X_test).flatten()

# Save LSTM model
model.save("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/05_Models/lstm_model.h5")

# -----------------------------------
# EVALUATE MODELS (RMSE)
def rmse(actual, pred):
    return mean_squared_error(actual, pred, squared=False)

arima_rmse = rmse(test_store["Weekly_Sales"], arima_preds)
prophet_rmse = rmse(test_store["Weekly_Sales"], prophet_preds)
lstm_rmse = rmse(test_store["Weekly_Sales"], lstm_preds)

print(f"ARIMA RMSE: {arima_rmse:.2f}")
print(f"Prophet RMSE: {prophet_rmse:.2f}")
print(f"LSTM RMSE: {lstm_rmse:.2f}")

# -----------------------------------
# Save predictions to CSV
results = test_store.copy()
results["ARIMA_Prediction"] = arima_preds
results["Prophet_Prediction"] = prophet_preds
results["LSTM_Prediction"] = lstm_preds
results.to_csv("R:/Projects/1_Data_Science & ML_Projects/04_Sales and Demand Forecasting for Retail Chains or Predictive Sales/02_Data/model_predictions.csv", index=False)

print("Forecast pipeline completed and predictions saved.")
