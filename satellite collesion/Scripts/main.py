# main.py
# Author: Harsh Nema
# Advanced Satellite Trajectory Risk Avoidance System (Phase 6 - DB Integrated)

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib.pyplot as plt
from database_manager import init_db, save_report   # ✅ Import database functions

# -------------------------------
# Load OpenAI API Key
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("⚠️ OPENAI_API_KEY missing from .env file")

client = OpenAI(api_key=api_key)

# -------------------------------
# STEP 1: Load NASA Orbit Data
# -------------------------------
print("📡 Loading NASA HORIZONS data...")
data_file = "data/horizons_results (1).txt"

lines = []
with open(data_file) as f:
    record = False
    for line in f:
        if "$$SOE" in line:
            record = True
            continue
        if "$$EOE" in line:
            break
        if record:
            parts = [x.strip() for x in line.split(",") if x.strip()]
            if len(parts) >= 11:
                lines.append(parts)

columns = [
    "JDTDB", "Calendar_Date", "X_km", "Y_km", "Z_km",
    "VX_km_s", "VY_km_s", "VZ_km_s", "LT_sec", "Range_km", "RangeRate_km_s"
]
df = pd.DataFrame(lines, columns=columns)
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"✅ Loaded {len(df)} data points from NASA HORIZONS")

# -------------------------------
# STEP 2: Simulate Another Satellite
# -------------------------------
print("🛰️ Simulating nearby satellite...")
df2 = df.copy()
df2["X_km"] += np.random.uniform(-50, 50, len(df))
df2["Y_km"] += np.random.uniform(-50, 50, len(df))
df2["Z_km"] += np.random.uniform(-50, 50, len(df))

df["Distance_km"] = np.sqrt(
    (df["X_km"] - df2["X_km"])**2 +
    (df["Y_km"] - df2["Y_km"])**2 +
    (df["Z_km"] - df2["Z_km"])**2
)

closest = df.loc[df["Distance_km"].idxmin()]

# ✅ Define risk_event BEFORE using it
risk_event = {
    "satellite_1": "ISS",
    "satellite_2": "SimSat-01",
    "closest_date": closest["Calendar_Date"],
    "min_distance_km": round(closest["Distance_km"], 3),
    "relative_speed_km_s": round(np.random.uniform(7.3, 7.8), 2)
}

print("\n🚨 Potential Collision Detected:\n", json.dumps(risk_event, indent=2))

# -------------------------------
# STEP 3: Query OpenAI for Maneuver
# -------------------------------
prompt = f"""
Two satellites ({risk_event['satellite_1']} and {risk_event['satellite_2']})
are predicted to approach within {risk_event['min_distance_km']} km
on {risk_event['closest_date']} at a relative speed of {risk_event['relative_speed_km_s']} km/s.

As an orbital-mechanics AI, recommend a safe, fuel-efficient avoidance maneuver including:
1. Δv magnitude (m/s)
2. Burn direction (radial / tangential / normal)
3. Reasoning behind your choice
4. Expected increase in miss distance (km)
5. A short 2-line summary
"""

print("\n🤖 Querying OpenAI model for maneuver plan...")

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You are an orbital mechanics AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
    )
    maneuver_text = response.choices[0].message.content.strip()
except Exception as e:
    print(f"⚠️ OpenAI API error: {e}")
    maneuver_text = "Simulated Output: Perform 0.08 m/s tangential burn at perigee to increase miss distance by ~1.2 km."

# -------------------------------
# STEP 4: Display & Save Report
# -------------------------------
print("\n📋 AI-Generated Maneuver Plan:\n")
print(maneuver_text)

# Save to text report
os.makedirs("reports", exist_ok=True)
report_path = f"reports/mission_report_openai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== AI Satellite Collision Avoidance Report ===\n\n")
    f.write("Collision Event:\n")
    f.write(json.dumps(risk_event, indent=2))
    f.write("\n\nOpenAI Maneuver Plan:\n")
    f.write(maneuver_text)

print(f"\n✅ Report saved at: {report_path}")

# -------------------------------
# STEP 5: Save to Database
# -------------------------------
init_db()  # make sure database exists
save_report(risk_event, maneuver_text)  # ✅ Now risk_event is defined here

# -------------------------------
# STEP 6: Orbit Visualization
# -------------------------------
plt.figure(figsize=(8, 8))
plt.plot(df["X_km"], df["Y_km"], label="ISS Orbit", color="blue")
plt.plot(df2["X_km"], df2["Y_km"], label="SimSat Orbit", color="orange", alpha=0.7)
plt.xlabel("X (km)")
plt.ylabel("Y (km)")
plt.title("Orbit Paths – Collision Simulation")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
