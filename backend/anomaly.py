import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ---------------- GLOBAL MODEL ----------------
model = IsolationForest(
    contamination=0.05,
    random_state=42,
    n_estimators=200
)

scaler = RobustScaler()

# ---------------- INITIAL BASELINE ----------------
BASE_SALES = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130]).reshape(-1, 1)

BASE_SCALED = scaler.fit_transform(BASE_SALES)
model.fit(BASE_SCALED)


# ---------------- ANOMALY FUNCTION ----------------
def detect_anomaly(sales_value, historical_data=None):

    try:
        # -------- retrain with real data if available --------
        if historical_data is not None and len(historical_data) > 20:

            data = np.array(historical_data).reshape(-1, 1)

            data_scaled = scaler.fit_transform(data)

            model.fit(data_scaled)

        # -------- scale input --------
        value_scaled = scaler.transform(np.array([[sales_value]]))

        prediction = model.predict(value_scaled)

        return bool(prediction[0] == -1)

    except Exception:
        return False
    