from sklearn.metrics import mean_squared_error
import joblib

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

def save_model(model, filepath):
    # For sklearn/statsmodels or joblib-compatible models
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)