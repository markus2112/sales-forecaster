import traceback
from pathlib import Path
import pandas as pd
import io

from fastapi import FastAPI, Depends, UploadFile, File
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

# ================= PATH =================
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

print("FRONTEND PATH =", FRONTEND_DIR)

app = FastAPI()

# ================= STATIC FILES =================
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

# ================= GLOBAL MODELS =================
xgb_model = None
lstm_model = None
lstm_scaler = None


# ================= HOME =================
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/styles.css", include_in_schema=False)
def frontend_styles():
    return FileResponse(
        FRONTEND_DIR / "styles.css",
        media_type="text/css"
    )


@app.get("/app.js", include_in_schema=False)
def frontend_script():
    return FileResponse(
        FRONTEND_DIR / "app.js",
        media_type="application/javascript"
    )


# ================= API HEALTH =================
@app.get("/api/health")
def api_health():
    return {
        "message": "API connected successfully"
    }


@app.get("/api/status")
def api_status():
    return {
        "xgboost_loaded": xgb_model is not None,
        "lstm_loaded": lstm_model is not None,
        "scaler_loaded": lstm_scaler is not None
    }


# ================= CREATE TABLE =================
@app.get("/create-table/")
def create_table(db: Session = Depends(get_db)):
    try:
        query = text("""
            CREATE TABLE IF NOT EXISTS sales_data (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                sales FLOAT NOT NULL,
                promotion BOOLEAN NOT NULL,
                stock INTEGER NOT NULL,
                holiday BOOLEAN NOT NULL
            )
        """)

        db.execute(query)
        db.commit()

        return {
            "message": "Table created successfully"
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= ADD SALES =================
@app.post("/add-sales/")
def add_sales(data: SalesData, db: Session = Depends(get_db)):
    try:
        validate_sales_data(data)

        anomaly_flag = detect_anomaly(data.sales)

        query = text("""
            INSERT INTO sales_data
            (date, sales, promotion, stock, holiday)
            VALUES
            (:date, :sales, :promotion, :stock, :holiday)
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
        return {
            "error": str(e)
        }


# ================= EXCEL UPLOAD =================
@app.post("/upload-excel/")
def upload_excel(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)

        elif filename.endswith(".xlsx"):
            contents = file.file.read()
            df = pd.read_excel(io.BytesIO(contents))

        else:
            return {
                "error": "Only CSV and XLSX files are allowed"
            }

        required_columns = [
            "date",
            "sales",
            "promotion",
            "stock",
            "holiday"
        ]

        for col in required_columns:
            if col not in df.columns:
                return {
                    "error": f"Missing column: {col}"
                }

        inserted_count = 0

        for _, row in df.iterrows():
            query = text("""
                INSERT INTO sales_data
                (date, sales, promotion, stock, holiday)
                VALUES
                (:date, :sales, :promotion, :stock, :holiday)
            """)

            db.execute(query, {
                "date": pd.to_datetime(row["date"]).date(),
                "sales": float(row["sales"]),
                "promotion": bool(row["promotion"]),
                "stock": int(row["stock"]),
                "holiday": bool(row["holiday"])
            })

            inserted_count += 1

        db.commit()

        return {
            "message": f"{inserted_count} rows uploaded successfully"
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= FETCH DATA =================
def get_clean_df(db):
    try:
        result = db.execute(
            text("""
                SELECT
                    date,
                    sales,
                    promotion,
                    stock,
                    holiday
                FROM sales_data
                ORDER BY date
            """)
        )

        rows = result.fetchall()

        df = pd.DataFrame(
            rows,
            columns=[
                "date",
                "sales",
                "promotion",
                "stock",
                "holiday"
            ]
        )

        if df.empty:
            return pd.DataFrame(columns=[
                "date",
                "sales",
                "promotion",
                "stock",
                "holiday"
            ])

        df["date"] = pd.to_datetime(df["date"])
        df["sales"] = df["sales"].astype(float)
        df["promotion"] = df["promotion"].astype(bool)
        df["stock"] = df["stock"].astype(int)
        df["holiday"] = df["holiday"].astype(bool)

        df = df.sort_values("date").reset_index(drop=True)

        return df

    except Exception:
        print(traceback.format_exc())

        return pd.DataFrame(columns=[
            "date",
            "sales",
            "promotion",
            "stock",
            "holiday"
        ])

# ================= GENERATE FEATURES =================
@app.get("/generate-features/")
def generate_features_api(db: Session = Depends(get_db)):
    try:
        df = get_clean_df(db)

        if df.empty:
            return {
                "error": "No sales data found"
            }

        feature_df = create_features(df)

        return feature_df.tail(10).to_dict(
            orient="records"
        )

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= TRAIN XGBOOST =================
@app.get("/train-xgboost/")
def train_xgboost_model(db: Session = Depends(get_db)):
    global xgb_model

    try:
        df = get_clean_df(db)

        if len(df) < 20:
            return {
                "error": "Minimum 20 rows required for XGBoost"
            }

        xgb_model = train_xgboost(df)
        save_xgboost_model(xgb_model)

        return {
            "message": "XGBoost model trained successfully"
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= TRAIN LSTM =================
@app.get("/train-lstm/")
def train_lstm_model(db: Session = Depends(get_db)):
    global lstm_model, lstm_scaler

    try:
        df = get_clean_df(db)

        if len(df) < 30:
            return {
                "error": "Minimum 30 rows required for LSTM"
            }

        lstm_model, lstm_scaler = train_lstm(df)

        save_lstm_model(lstm_model)
        save_scaler(lstm_scaler)

        return {
            "message": "LSTM model trained successfully"
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= HYBRID FORECAST =================
@app.get("/hybrid-forecast/")
def hybrid_forecast_api(db: Session = Depends(get_db)):
    global xgb_model, lstm_model, lstm_scaler

    try:
        if xgb_model is None or lstm_model is None:
            return {
                "error": "Train both models first"
            }

        df = get_clean_df(db)

        if len(df) < 20:
            return {
                "error": "Not enough data"
            }

        features_df = create_features(df).dropna()

        if len(features_df) == 0:
            return {
                "error": "Feature dataframe empty"
            }

        xgb_preds = predict_xgboost(
            xgb_model,
            features_df
        )

        lstm_preds = predict_lstm(
            lstm_model,
            lstm_scaler,
            df
        )

        hybrid_preds = hybrid_forecast(
            xgb_preds,
            None,
            lstm_preds
        )

        result = [
            round(float(x), 2)
            for x in hybrid_preds[-5:]
        ]

        return {
            "hybrid_prediction": result,
            "count": len(result)
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= EVALUATE MODELS =================
@app.get("/evaluate-models/")
def evaluate_models_api(db: Session = Depends(get_db)):
    global xgb_model, lstm_model, lstm_scaler

    try:
        if xgb_model is None or lstm_model is None:
            return {
                "error": "Train both models first"
            }

        df = get_clean_df(db)

        if len(df) < 40:
            return {
                "error": "Minimum 40 rows required"
            }

        features_df = create_features(df).dropna()

        if len(features_df) == 0:
            return {
                "error": "Feature dataframe empty"
            }

        xgb_preds = predict_xgboost(
            xgb_model,
            features_df
        )

        lstm_preds = predict_lstm(
            lstm_model,
            lstm_scaler,
            df
        )

        hybrid_preds = hybrid_forecast(
            xgb_preds,
            None,
            lstm_preds
        )

        actual = df["sales"].values

        results = evaluate_models(
            actual,
            xgb_preds,
            None,
            lstm_preds,
            hybrid_preds
        )

        return results

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }
# ================= LOAD MODELS =================
@app.on_event("startup")
def load_models():
    global xgb_model, lstm_model, lstm_scaler

    try:
        xgb_model = load_xgboost_model()
        lstm_model = load_lstm_model()
        lstm_scaler = load_scaler()

        print("Saved models loaded successfully")

    except:
        print("No saved models found")