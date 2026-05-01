import numpy as np
import pandas as pd
import joblib
import os

from xgboost import XGBRegressor

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

from backend.features import create_features

import warnings
warnings.filterwarnings("ignore")

WINDOW = 14

# ================= MODEL DIR =================
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ================= CLEAN =================
def clean_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset="date", keep="last")
    return df


# ================= XGBOOST =================
def train_xgboost(df):

    df = create_features(clean_data(df))

    X = df.drop(columns=["date", "sales"])
    y = np.log1p(df["sales"])

    split = int(len(X) * 0.85)

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=30,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model


# ================= LSTM =================
def train_lstm(df):

    df = create_features(clean_data(df))
    
    feature_cols = [c for c in df.columns if c != "date"]
    data_values = df[feature_cols].values
    sales_idx = feature_cols.index("sales")

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    data_scaled = scaler_X.fit_transform(data_values)
    
    sales_values = df["sales"].values.reshape(-1, 1)
    scaler_y.fit(sales_values)

    if len(data_scaled) <= WINDOW:
        raise ValueError("Not enough data for LSTM")

    X, y = [], []

    for i in range(len(data_scaled) - WINDOW):
        X.append(data_scaled[i:i + WINDOW])
        y.append(data_scaled[i + WINDOW, sales_idx])

    X = np.array(X)
    y = np.array(y)
    
    num_features = X.shape[2]

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(WINDOW, num_features)),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mae")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X, y,
        epochs=40,
        batch_size=8,
        validation_split=0.1,
        verbose=0,
        callbacks=[early_stop]
    )
    
    scalers = {
        "X": scaler_X,
        "y": scaler_y,
        "features": feature_cols
    }

    return model, scalers


# ================= PREDICT =================
def predict_xgboost(model, df):

    df = create_features(clean_data(df))
    X = df.drop(columns=["date", "sales"])

    preds = model.predict(X)

    return np.expm1(preds)


def predict_lstm(model, scalers, df):

    df = create_features(clean_data(df))
    
    scaler_X = scalers["X"]
    scaler_y = scalers["y"]
    feature_cols = scalers["features"]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    data_values = df[feature_cols].values
    data_scaled = scaler_X.transform(data_values)

    if len(data_scaled) <= WINDOW:
        return np.array([])

    X = []

    for i in range(len(data_scaled) - WINDOW):
        X.append(data_scaled[i:i + WINDOW])

    X = np.array(X)

    preds = model.predict(X, verbose=0)
    preds = scaler_y.inverse_transform(preds)

    return preds.flatten()


# ================= HYBRID =================
def hybrid_forecast(xgb_preds, lgb_preds=None, lstm_preds=None):

    if lstm_preds is None or len(lstm_preds) == 0:
        return np.array(xgb_preds)

    length = min(len(xgb_preds), len(lstm_preds))

    xgb_preds = np.array(xgb_preds[-length:])
    lstm_preds = np.array(lstm_preds[-length:])

    return 0.5 * xgb_preds + 0.5 * lstm_preds


# ================= EVALUATION =================
def evaluate_models(actual, xgb_preds, lgb_preds=None, lstm_preds=None, hybrid_preds=None):

    def compute(y_true, y_pred):

        if y_pred is None or len(y_pred) == 0:
            return {"MAE": None, "RMSE": None, "MAPE": None}

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        length = min(len(y_true), len(y_pred))
        y_true = y_true[-length:]
        y_pred = y_pred[-length:]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        safe = np.where(y_true == 0, 1e-6, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / safe)) * 100

        return {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "MAPE": round(float(mape), 2)
        }

    return {
        "XGBoost": compute(actual, xgb_preds),
        "LSTM": compute(actual, lstm_preds),
        "Hybrid": compute(actual, hybrid_preds)
    }


# ================= SAVE =================
def save_xgboost_model(model):
    joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))

def save_lstm_model(model):
    model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

def save_scaler(scaler):
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))


# ================= LOAD =================
def load_xgboost_model():
    return joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))

def load_lstm_model():
    return load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))

def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))