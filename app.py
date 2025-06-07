import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from prophet import Prophet
import pickle
from sklearn.metrics import mean_squared_error

# Paths
processed_path = "data/processed/"
models_path = "models/"

# Load data
df = pd.read_csv(processed_path + "model_predictions.csv", parse_dates=["Date"])
stores = sorted(df["Store"].unique())

st.title("Retail Sales Forecasting Dashboard 📊")

# Sidebar
store_selected = st.sidebar.selectbox("Select Store", stores)
model_selected = st.sidebar.selectbox("Select Forecasting Model", ["ARIMA", "Prophet", "LSTM"])

# Filter for selected store
df_store = df[df["Store"] == store_selected]

# Actual vs predicted
st.subheader(f"Store {store_selected} - {model_selected} Forecast")

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(df_store["Date"], df_store["Weekly_Sales"], label="Actual", color="black")

if model_selected == "ARIMA":
    preds = df_store["ARIMA_Prediction"]
elif model_selected == "Prophet":
    preds = df_store["Prophet_Prediction"]
else:
    preds = df_store["LSTM_Prediction"]

ax.plot(df_store["Date"], preds, label="Predicted", color="orange")
ax.set_xlabel("Date")
ax.set_ylabel("Weekly Sales")
ax.legend()
st.pyplot(fig)

# Metrics
rmse = mean_squared_error(df_store["Weekly_Sales"], preds, squared=False)
st.metric(label=f"{model_selected} RMSE", value=f"{rmse:.2f}")
