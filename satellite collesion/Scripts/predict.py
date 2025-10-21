"""
Simple inference helper for saved collision predictor.
Usage:
    python scripts/predict.py --file data/processed_dataset.csv
    or
    python scripts/predict.py --single "x,y,z,vx,vy,vz,delta_km,deldot"
"""

import os
import sys
import argparse
import pandas as pd
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "collision_predictor.pkl")

def predict_file(path):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(path)
    # minimal feature engineering to match trainer:
    # trainer expects pos_mag, speed, radial_speed, x,y,z,vx,vy,vz,delta_km,deldot,hour,dayofyear
    # We'll compute same fields here:
    for c in ["x","y","z","vx","vy","vz"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    df["pos_mag"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df["speed"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
    df["radial_speed"] = (df["x"]*df["vx"] + df["y"]*df["vy"] + df["z"]*df["vz"]) / df["pos_mag"].replace(0, np.nan)
    df["radial_speed"] = df["radial_speed"].fillna(0.0)
    df["delta_km"] = pd.to_numeric(df.get("delta_km", df.get("Delta", df.get("Range_km", df.get("delta", df.get("range", df["pos_mag"]))))), errors="coerce").fillna(df["pos_mag"])
    df["deldot"] = pd.to_numeric(df.get("deldot", df.get("Deldot", 0.0)), errors="coerce").fillna(0.0)
    df["time"] = pd.to_datetime(df.get("time", df.get("UTC_Time", df.get("Date_UT", pd.NaT))), errors="coerce")
    df["hour"] = df["time"].dt.hour.fillna(0).astype(int)
    df["dayofyear"] = df["time"].dt.dayofyear.fillna(0).astype(int)

    feature_cols = ["pos_mag","speed","radial_speed","x","y","z","vx","vy","vz","delta_km","deldot","hour","dayofyear"]
    X = df[feature_cols].fillna(0.0)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model, "predict_proba") else None

    df["collision_pred"] = preds
    if probs is not None:
        df["collision_proba"] = probs

    out = os.path.join(BASE_DIR, "data", "predictions_output.csv")
    df.to_csv(out, index=False)
    print(f"[i] Predictions saved to {out}")

def predict_single(s):
    vals = [float(x) for x in s.split(",")]
    # expecting at least x,y,z,vx,vy,vz,delta,deldot (8 values). If less, pad.
    while len(vals) < 8:
        vals.append(0.0)
    x,y,z,vx,vy,vz,delta_km,deldot = vals[:8]
    pos_mag = np.sqrt(x*x + y*y + z*z)
    speed = np.sqrt(vx*vx + vy*vy + vz*vz)
    radial_speed = (x*vx + y*vy + z*vz) / (pos_mag if pos_mag!=0 else 1.0)
    hour = 0
    dayofyear = 0
    X = [[pos_mag,speed,radial_speed,x,y,z,vx,vy,vz,delta_km,deldot,hour,dayofyear]]
    model = joblib.load(MODEL_PATH)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0,1] if hasattr(model, "predict_proba") else None
    print("prediction:", pred, "proba:", proba)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="CSV file to predict on")
    parser.add_argument("--single", help="single sample as comma-separated numbers")
    args = parser.parse_args()
    if args.file:
        predict_file(args.file)
    elif args.single:
        predict_single(args.single)
    else:
        print("Specify --file <csv> or --single \"x,y,z,vx,vy,vz,delta,deldot\"")
