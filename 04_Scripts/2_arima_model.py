import joblib
from statsmodels.tsa.arima.model import ARIMA

model = None

def train_arima(train_df):
    global model
    ts = train_df.set_index("Date")["Weekly_Sales"]
    model = ARIMA(ts, order=(5,1,0)).fit()

def predict_arima(test_df):
    start_date = test_df["Date"].min()
    end_date = test_df["Date"].max()
    preds = model.predict(start=start_date, end=end_date)
    # Align length with test
    return preds[:len(test_df)]

def save_model(filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    global model
    model = joblib.load(filepath)