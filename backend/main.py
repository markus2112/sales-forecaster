import traceback
from pathlib import Path
import pandas as pd
import io

from fastapi import FastAPI, Depends, UploadFile, File, Response
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

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(content=b"", media_type="image/x-icon")



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


# ================= NULL VALUE DETECTION =================
@app.post("/detect-nulls/")
def detect_nulls(file: UploadFile = File(...)):
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

        null_counts = df.isnull().sum().to_dict()
        total_rows = len(df)
        total_nulls = int(sum(null_counts.values()))
        clean_null_counts = {k: int(v) for k, v in null_counts.items()}

        return {
            "total_rows": total_rows,
            "total_nulls": total_nulls,
            "null_breakdown": clean_null_counts,
            "message": "Null value detection complete"
        }

    except Exception as e:
        print(traceback.format_exc())
        return {
            "error": str(e)
        }


# ================= ERROR DETECTION =================
@app.post("/detect-errors/")
def detect_errors(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith(".xlsx"):
            contents = file.file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            return {"error": "Only CSV and XLSX files are allowed"}

        errors = []
        
        # 1. Check Required Columns
        required_columns = ["date", "sales", "promotion", "stock", "holiday"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            errors.append({"type": "Missing Columns", "detail": f"Missing: {', '.join(missing_cols)}"})
            return {
                "total_rows": len(df),
                "error_count": len(errors),
                "errors": errors,
                "message": "Critical schema errors found"
            }

        # 2. Check Duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            errors.append({"type": "Duplicate Rows", "detail": f"Found {int(duplicate_count)} duplicate row(s)"})

        # 3. Check Negative Sales
        try:
            negative_sales = (pd.to_numeric(df["sales"], errors='coerce') < 0).sum()
            if negative_sales > 0:
                errors.append({"type": "Negative Sales", "detail": f"Found {int(negative_sales)} row(s) with negative sales"})
        except:
            errors.append({"type": "Data Type Error", "detail": "Sales column contains invalid numeric data"})

        # 4. Check Negative Stock
        try:
            negative_stock = (pd.to_numeric(df["stock"], errors='coerce') < 0).sum()
            if negative_stock > 0:
                errors.append({"type": "Negative Stock", "detail": f"Found {int(negative_stock)} row(s) with negative stock"})
        except:
            errors.append({"type": "Data Type Error", "detail": "Stock column contains invalid numeric data"})

        # 5. Check Invalid Dates
        try:
            invalid_dates = df["date"].isnull().sum()
            parsed_dates = pd.to_datetime(df["date"], errors='coerce')
            unparseable = parsed_dates.isnull().sum() - invalid_dates
            if unparseable > 0:
                errors.append({"type": "Invalid Dates", "detail": f"Found {int(unparseable)} row(s) with unparseable dates"})
        except:
            errors.append({"type": "Data Type Error", "detail": "Date column contains invalid formats"})

        return {
            "total_rows": int(len(df)),
            "error_count": len(errors),
            "errors": errors,
            "message": "Error detection complete"
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}

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

        future_preds = []
        df_future = df.copy()

        for step in range(5):
            last_row = df_future.iloc[-1]
            next_date = last_row["date"] + pd.Timedelta(days=1)
            
            # Create a new row
            new_row = pd.DataFrame([{
                "date": next_date,
                "sales": last_row["sales"],  # Dummy value
                "promotion": last_row["promotion"],
                "stock": last_row["stock"],
                "holiday": last_row["holiday"]
            }])
            
            df_future = pd.concat([df_future, new_row], ignore_index=True)
            
            # Predict XGBoost
            xgb_preds = predict_xgboost(xgb_model, df_future)
            p_xgb = float(xgb_preds[-1])
            
            # Predict LSTM
            lstm_preds = predict_lstm(lstm_model, lstm_scaler, df_future)
            p_lstm = float(lstm_preds[-1]) if len(lstm_preds) > 0 else p_xgb
            
            # Hybrid
            p_hybrid = 0.7 * p_xgb + 0.3 * p_lstm
            future_preds.append(round(p_hybrid, 2))
            
            # Update dummy sales with prediction
            df_future.at[len(df_future)-1, "sales"] = p_hybrid

        return {
            "hybrid_prediction": future_preds,
            "count": len(future_preds)
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
            df
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

# ================= AI STRATEGIC INSIGHTS =================
@app.get("/generate-insights/")
def generate_insights_api(db: Session = Depends(get_db)):
    try:
        df = get_clean_df(db)

        if len(df) < 14:
            return {"error": "Need at least 14 days of data to generate insights"}

        insights = []

        # 1. Promotional Impact
        promo_sales = df[df["promotion"] == True]["sales"].mean()
        non_promo_sales = df[df["promotion"] == False]["sales"].mean()
        
        if pd.notna(promo_sales) and pd.notna(non_promo_sales) and non_promo_sales > 0:
            if promo_sales > non_promo_sales * 1.15:
                increase = round(((promo_sales - non_promo_sales) / non_promo_sales) * 100)
                insights.append({
                    "type": "positive",
                    "title": "Promotions are Working",
                    "text": f"Promotional days see a {increase}% increase in average sales. Consider increasing marketing spend on strategic campaigns.",
                    "icon": "bx-trending-up"
                })
            elif promo_sales < non_promo_sales * 1.05:
                insights.append({
                    "type": "warning",
                    "title": "Weak Promo Impact",
                    "text": "Recent promotions haven't significantly boosted sales. Review your campaign targeting or offer value.",
                    "icon": "bx-target-lock"
                })

        # 2. Inventory Health
        min_stock = df.tail(7)["stock"].min()
        if min_stock < 20:
            insights.append({
                "type": "danger",
                "title": "Critical Stockout Risk",
                "text": f"Inventory levels dropped to {min_stock} units recently. Increase buffer stock to prevent lost sales.",
                "icon": "bx-error"
            })
        elif min_stock > 100:
            insights.append({
                "type": "positive",
                "title": "Healthy Inventory",
                "text": "Buffer stock is strong, preventing potential out-of-stock lost revenue.",
                "icon": "bx-check-shield"
            })

        # 3. Trend Momentum
        recent_avg = df.tail(7)["sales"].mean()
        past_avg = df.iloc[-14:-7]["sales"].mean()
        
        if pd.notna(recent_avg) and pd.notna(past_avg) and past_avg > 0:
            if recent_avg > past_avg * 1.1:
                insights.append({
                    "type": "positive",
                    "title": "Sales Surging",
                    "text": "7-day sales average is up significantly compared to the previous week. Capitalize on this momentum.",
                    "icon": "bx-line-chart"
                })
            elif recent_avg < past_avg * 0.9:
                insights.append({
                    "type": "warning",
                    "title": "Dropping Momentum",
                    "text": "7-day average sales have dropped. Immediate promotional intervention is recommended.",
                    "icon": "bx-trending-down"
                })

        # 4. Weekend vs Weekday
        df["is_weekend"] = df["date"].dt.dayofweek >= 5
        weekend_sales = df[df["is_weekend"]]["sales"].mean()
        weekday_sales = df[~df["is_weekend"]]["sales"].mean()
        
        if pd.notna(weekend_sales) and pd.notna(weekday_sales) and weekday_sales > 0:
            if weekend_sales > weekday_sales * 1.2:
                insights.append({
                    "type": "info",
                    "title": "Weekend Dominance",
                    "text": "Sales spike significantly on weekends. Align ad-spend and staff schedules accordingly.",
                    "icon": "bx-calendar-star"
                })

        if not insights:
            insights.append({
                "type": "info",
                "title": "Stable Baseline",
                "text": "Sales metrics are stable with no extreme anomalies detected. Keep monitoring the dashboard.",
                "icon": "bx-info-circle"
            })

        return {"insights": insights}

    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e)}
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