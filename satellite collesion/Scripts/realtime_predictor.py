"""
🌍 Phase 10A — Real-Time Collision Risk Predictor
Author: Harsh Nema
Description:
Loads trained model + continuously monitors satellite trajectory data
to predict collision probabilities and suggest safe maneuvers.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# 📂 PATH CONFIGURATION
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODELS_DIR, "collision_predictor.pkl")
LIVE_DATA = os.path.join(DATA_DIR, "live_satellite_feed.csv")

# ------------------------------------------------------
# ⚙️ LOAD MODEL
# ------------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("❌ Model file not found. Run Phase 9 first.")
    model = joblib.load(MODEL_FILE)
    print("✅ Loaded trained model successfully.")
    return model

# ------------------------------------------------------
# 🧮 FEATURE PREPROCESSOR (same as Phase 9)
# ------------------------------------------------------
def preprocess(df):
    df["pos_mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    df["radial_speed"] = (
        (df["x"]*df["vx"] + df["y"]*df["vy"] + df["z"]*df["vz"]) /
        df["pos_mag"].replace(0, np.nan)
    ).fillna(0.0)
    df["delta_km"] = pd.to_numeric(df.get("Delta", df.get("delta_km", df["pos_mag"])), errors="coerce").fillna(df["pos_mag"])
    df["deldot"] = pd.to_numeric(df.get("Deldot", df.get("deldot", 0.0)), errors="coerce").fillna(0.0)

    df["hour"] = pd.to_datetime(df["UTC_Time"], errors="coerce").dt.hour.fillna(0)
    df["dayofyear"] = pd.to_datetime(df["UTC_Time"], errors="coerce").dt.dayofyear.fillna(0)

    features = ["pos_mag","speed","radial_speed","x","y","z","vx","vy","vz","delta_km","deldot","hour","dayofyear"]
    return df[features].fillna(0.0)

# ------------------------------------------------------
# 🚀 MANEUVER SUGGESTION LOGIC
# ------------------------------------------------------
def suggest_maneuver(row, prob):
    """
    Suggests simple orbital maneuvers based on risk probability.
    """
    if prob > 0.8:
        return "⚠️ Immediate radial boost (+50 m/s) & minor inclination change (2°)"
    elif prob > 0.5:
        return "🟡 Adjust altitude +20 km & re-sync orbital phase"
    elif prob > 0.2:
        return "🟢 Monitor — small attitude adjustment recommended"
    else:
        return "✅ Stable orbit — no action needed"

# ------------------------------------------------------
# 🔁 REAL-TIME MONITOR
# ------------------------------------------------------
def main():
    model = load_model()
    print("\n🛰️ Starting real-time collision risk monitoring...\n")

    # continuous monitoring loop (simulate live telemetry)
    while True:
        if os.path.exists(LIVE_DATA):
            df = pd.read_csv(LIVE_DATA)
            X = preprocess(df)
            proba = model.predict_proba(X)[:,1]
            df["collision_probability"] = proba
            df["maneuver_plan"] = [suggest_maneuver(r, p) for r, p in zip(df.to_dict("records"), proba)]

            print("\n📡 Latest Predictions:")
            print(df[["UTC_Time", "collision_probability", "maneuver_plan"]].tail(3))
        else:
            print("⚠️ Waiting for live_satellite_feed.csv ...")

        time.sleep(5)  # refresh every 5 seconds

if __name__ == "__main__":
    main()
