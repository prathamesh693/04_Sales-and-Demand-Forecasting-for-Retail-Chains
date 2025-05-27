from prophet import Prophet
import pickle

model = None

def train_prophet(train_df):
    global model
    df = train_df[["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    model = Prophet()
    model.fit(df)

def predict_prophet(test_df):
    future = test_df[["Date"]].rename(columns={"Date": "ds"})
    forecast = model.predict(future)
    return forecast["yhat"].values

def save_model(filepath):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath):
    global model
    with open(filepath, "rb") as f:
        model = pickle.load(f)