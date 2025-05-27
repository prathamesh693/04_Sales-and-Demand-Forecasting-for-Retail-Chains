import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

model = None

def create_dataset(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)

def train_lstm(train_df):
    global model
    sales = train_df.sort_values("Date")["Weekly_Sales"].values
    X, y = create_dataset(sales)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

def predict_lstm(test_df, train_df, n_steps=10):
    sales_train = train_df.sort_values("Date")["Weekly_Sales"].values
    sales_test = test_df.sort_values("Date")["Weekly_Sales"].values

    # Prepare input sequences by combining last train steps + test steps
    input_seq = np.concatenate([sales_train[-n_steps:], sales_test[:-1]])
    X_test = []
    for i in range(len(test_df)):
        X_test.append(input_seq[i:i+n_steps])
    X_test = np.array(X_test).reshape((len(test_df), n_steps, 1))

    preds = model.predict(X_test)
    return preds.flatten()

def save_model(filepath):
    model.save(filepath)

def load_model_file(filepath):
    global model
    model = load_model(filepath)