import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.database import SessionLocal
from backend.main import get_clean_df
from backend.models import train_xgboost, train_lstm, predict_xgboost, predict_lstm, evaluate_models, hybrid_forecast

def main():
    db = SessionLocal()
    df = get_clean_df(db)
    print(f"Data shape: {df.shape}")
    
    print("Training XGBoost...")
    xgb_model = train_xgboost(df)
    
    print("Training LSTM...")
    lstm_model, lstm_scaler = train_lstm(df)
    
    print("Evaluating XGBoost...")
    xgb_preds = predict_xgboost(xgb_model, df)
    
    print("Evaluating LSTM...")
    lstm_preds = predict_lstm(lstm_model, lstm_scaler, df)
    
    hybrid_70_30 = hybrid_forecast(xgb_preds, None, lstm_preds)
    
    actual = df["sales"].values
    
    res = evaluate_models(actual, xgb_preds, None, lstm_preds, hybrid_70_30)
    print("Evaluation Results:")
    print(res)

if __name__ == "__main__":
    main()
