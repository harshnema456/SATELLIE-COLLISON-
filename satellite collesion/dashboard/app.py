"""
🚀 Phase 10B — Real-Time Collision Monitoring Dashboard
Author: Harsh Nema.
"""

import os
import time
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ✅ Must be first Streamlit command
st.set_page_config(page_title="AI Collision Risk Dashboard", layout="wide", page_icon="🛰️")

# ------------------------------------------------------
# 📂 PATH CONFIGURATION
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILE = os.path.join(MODELS_DIR, "collision_predictor.pkl")
LIVE_FILE = os.path.join(DATA_DIR, "live_satellite_feed.csv")

# ------------------------------------------------------
# 🧠 LOAD MODEL
# ------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.error("❌ Model not found! Please run Phase 9 first.")
        st.stop()
    model = joblib.load(MODEL_FILE)
    return model

model = load_model()

# ------------------------------------------------------
# 🧮 FEATURE PREPROCESSOR
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
# 🚀 MANEUVER SUGGESTION
# ------------------------------------------------------
def suggest_maneuver(prob):
    if prob > 0.8:
        return "⚠️ Immediate radial boost (+50 m/s) & inclination change (2°)"
    elif prob > 0.5:
        return "🟡 Adjust altitude +20 km & re-sync orbital phase"
    elif prob > 0.2:
        return "🟢 Monitor — minor attitude correction"
    else:
        return "✅ Stable orbit — no action needed"

# ------------------------------------------------------
# 🛰️ DASHBOARD UI
# ------------------------------------------------------
st.title("🛰️ Real-Time Satellite Collision Risk Dashboard")
st.markdown("### Phase 10B — Powered by Harsh Nema’s Space AI System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

# ------------------------------------------------------
# 🔁 LIVE DATA STREAMING
# ------------------------------------------------------
def load_live_data():
    if not os.path.exists(LIVE_FILE):
        st.warning("⚠️ Waiting for live_satellite_feed.csv ...")
        return pd.DataFrame()
    return pd.read_csv(LIVE_FILE)

refresh_rate = st.sidebar.slider("🔁 Refresh every (seconds):", 2, 20, 5)

# ------------------------------------------------------
# 🚀 LIVE UPDATER
# ------------------------------------------------------
placeholder = st.empty()
while True:
    df = load_live_data()
    if not df.empty:
        X = preprocess(df)
        proba = model.predict_proba(X)[:, 1]
        df["collision_probability"] = proba
        df["maneuver_plan"] = [suggest_maneuver(p) for p in proba]

        latest = df.tail(1).iloc[0]
        risk = latest["collision_probability"]
        status_color = "red" if risk > 0.7 else "orange" if risk > 0.4 else "green"

        with placeholder.container():
            with col2:
                st.markdown("### 🔍 Latest Prediction")
                st.metric("Collision Probability", f"{risk*100:.2f}%", delta=None)
                st.markdown(f"**Suggested Maneuver:** {latest['maneuver_plan']}")

            with col1:
                st.markdown("### 🌐 3D Orbit Visualization")
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=df["x"], y=df["y"], z=df["z"],
                    mode="lines+markers",
                    line=dict(color="cyan", width=3),
                    marker=dict(size=3, color=df["collision_probability"], colorscale="RdYlGn_r")
                ))
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    scene=dict(
                        xaxis_title="X (km)",
                        yaxis_title="Y (km)",
                        zaxis_title="Z (km)"
                    ),
                    height=600,
                    paper_bgcolor="#0e1117",
                    scene_camera=dict(eye=dict(x=1.8, y=1.8, z=0.9))
                )

                # ✅ FIX: add a unique key for each render
                st.plotly_chart(fig, use_container_width=True, key=f"orbit_chart_{time.time()}")

            st.markdown("### 🧾 Prediction History")
            st.dataframe(df[["UTC_Time", "collision_probability", "maneuver_plan"]].tail(10))

    else:
        st.info("Waiting for live data stream...")

    time.sleep(refresh_rate)
