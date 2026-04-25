import traceback
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.database import get_db
from backend.schemas import SalesData
from backend.validation import validate_sales_data
from backend.anomaly import detect_anomaly
from backend.features import create_features

from backend.models import (
    train_xgboost,
    train_lstm,
    predict_xgboost,
    predict_lstm,
    hybrid_forecast,
    evaluate_models,
    save_xgboost_model,
    save_lstm_model,
    save_scaler,
    load_xgboost_model,
    load_lstm_model,
    load_scaler
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI()
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

# ---------------- GLOBAL MODELS ----------------
xgb_model = None
lstm_model = None
lstm_scaler = None


# ---------------- HOME ----------------
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def frontend_styles():
    return FileResponse(FRONTEND_DIR / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def frontend_script():
    return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


@app.get("/api/health")
def api_health():
    return {"message": "API connected to database"}


@app.get("/api/status")
def api_status():
    return {
        "xgboost_loaded": xgb_model is not None,
        "lstm_loaded": lstm_model is not None,
        "scaler_loaded": lstm_scaler is not None
    }


# ---------------- INSERT ----------------
@app.post("/add-sales/")
def add_sales(data: SalesData, db: Session = Depends(get_db)):
    try:
        validate_sales_data(data)
        anomaly_flag = detect_anomaly(data.sales)

        query = text("""
            INSERT INTO sales_data (date, sales, promotion, stock, holiday)
            VALUES (:date, :sales, :promotion, :stock, :holiday)
        """)

        db.execute(query, {
            "date": data.date,
            "sales": data.sales,
            "promotion": data.promotion,
            "stock": data.stock,
            "holiday": data.holiday
        })

        db.commit()

        return {
            "message": "Sales data inserted successfully",
            "anomaly": anomaly_flag
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- DATA FETCH ----------------
def get_clean_df(db):
    result = db.execute(text("SELECT date, sales FROM sales_data"))
    rows = result.fetchall()

    df = pd.DataFrame(rows, columns=["date", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


# ---------------- FEATURE ----------------
@app.get("/generate-features/")
def generate_features(db: Session = Depends(get_db)):
    try:
        df = get_clean_df(db)
        df = create_features(df)
        return df.tail(10).to_dict(orient="records")

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- TRAIN XGBOOST ----------------
@app.get("/train-xgboost/")
def train_xgboost_model(db: Session = Depends(get_db)):
    global xgb_model

    try:
        df = get_clean_df(db)

        if len(df) < 20:
            return {"error": "Not enough data for XGBoost"}

        xgb_model = train_xgboost(df)

        save_xgboost_model(xgb_model)

        return {"message": "XGBoost Model trained and saved"}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- TRAIN LSTM ----------------
@app.get("/train-lstm/")
def train_lstm_model(db: Session = Depends(get_db)):
    global lstm_model, lstm_scaler

    try:
        df = get_clean_df(db)

        if len(df) < 30:
            return {"error": "Not enough data for LSTM"}

        lstm_model, lstm_scaler = train_lstm(df)

        save_lstm_model(lstm_model)
        save_scaler(lstm_scaler)

        return {"message": "LSTM model trained and saved"}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- HYBRID ----------------
@app.get("/hybrid-forecast/")
def hybrid_forecast_api(db: Session = Depends(get_db)):

    global xgb_model, lstm_model, lstm_scaler

    try:
        if xgb_model is None or lstm_model is None:
            return {"error": "Train both models first"}

        df = get_clean_df(db)

        if len(df) < 20:
            return {"error": "Not enough data"}

        features_df = create_features(df).dropna()

        if len(features_df) == 0:
            return {"error": "Feature dataframe empty"}

        # ALIGN
        df_aligned = df.loc[features_df.index].reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)

        # PREDICTIONS
        xgb_preds = predict_xgboost(xgb_model, features_df)
        lstm_preds = predict_lstm(lstm_model, lstm_scaler, df_aligned)

        if len(xgb_preds) == 0 or len(lstm_preds) == 0:
            return {"error": "Prediction failed (empty output)"}

        hybrid_preds = hybrid_forecast(xgb_preds, None, lstm_preds)

        result = [round(float(p), 2) for p in hybrid_preds[-5:]]

        return {
            "hybrid_prediction": result,
            "count": len(result)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}


# ---------------- EVALUATION ----------------
@app.get("/evaluate-models/")
def evaluate_models_api(db: Session = Depends(get_db)):

    global xgb_model, lstm_model, lstm_scaler

    try:
        if xgb_model is None or lstm_model is None:
            return {"error": "Train both models first"}

        df = get_clean_df(db)

        if len(df) < 40:
            return {"error": "Not enough data"}

        features_df = create_features(df).dropna()

        if len(features_df) == 0:
            return {"error": "Feature dataframe empty"}

        df_aligned = df.loc[features_df.index].reset_index(drop=True)
        features_df = features_df.reset_index(drop=True)

        xgb_preds = predict_xgboost(xgb_model, features_df)
        lstm_preds = predict_lstm(lstm_model, lstm_scaler, df_aligned)

        hybrid_preds = hybrid_forecast(xgb_preds, None, lstm_preds)

        actual = df_aligned["sales"].values

        results = evaluate_models(
            actual,
            xgb_preds,
            None,
            lstm_preds,
            hybrid_preds
        )

        return results

    except Exception as e:
        return {"error": str(e)}


# ---------------- LOAD MODELS ON STARTUP ----------------
@app.on_event("startup")
def load_models():
    global xgb_model, lstm_model, lstm_scaler

    try:
        xgb_model = load_xgboost_model()
        lstm_model = load_lstm_model()
        lstm_scaler = load_scaler()

        print("Models loaded successfully")

    except:
        print("No saved models found")
