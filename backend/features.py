import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # ---------------- SORT & DATE ----------------
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # ---------------- LAG FEATURES ----------------
    for lag in [1, 2, 3, 4, 5, 6, 7, 14]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # ---------------- ROLLING FEATURES (NO LEAKAGE) ----------------
    shifted = df["sales"].shift(1)

    df["rolling_mean_3"] = shifted.rolling(3).mean()
    df["rolling_mean_7"] = shifted.rolling(7).mean()
    df["rolling_mean_14"] = shifted.rolling(14).mean()

    df["rolling_std_3"] = shifted.rolling(3).std()
    df["rolling_std_7"] = shifted.rolling(7).std()

    df["rolling_median_7"] = shifted.rolling(7).median()

    # ---------------- ADVANCED FEATURES ----------------
    df["momentum_3"] = df["lag_1"] - df["lag_4"]
    df["momentum_7"] = df["lag_1"] - df["lag_7"]

    df["pct_change_1"] = (df["lag_1"] - df["lag_2"]) / (df["lag_2"] + 1e-6)

    df["spike_score"] = (
        (df["lag_1"] - df["rolling_median_7"]) /
        (df["rolling_std_7"] + 1e-6)
    )

    # ---------------- DATE FEATURES ----------------
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # ---------------- CYCLICAL FEATURES ----------------
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---------------- CLEAN ----------------
    df = df.dropna().reset_index(drop=True)

    return df