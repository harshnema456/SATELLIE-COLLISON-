# llm_decision.py
# Author: Harsh Nema
# Purpose: LLM module for AI-based collision avoidance maneuver planning

import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load API key (if available)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", None)

# ------------------------
# Step 1: Load Closest Approach Data
# ------------------------
# We’ll reuse the same logic as risk_detection.py
data_file = "../data/horizons_results (1).txt"

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

# For demonstration: we’ll simulate a detected risk
risk_event = {
    "satellite_1": "ISS",
    "satellite_2": "SimSat-01",
    "closest_date": "2024-10-25 00:00",
    "min_distance_km": 0.62,
    "relative_speed_km_s": 7.5
}

print("🚨 Potential Collision Event Detected:")
print(json.dumps(risk_event, indent=2))

# ------------------------
# Step 2: Prepare Prompt for the LLM
# ------------------------
prompt = f"""
Two satellites are predicted to approach within {risk_event['min_distance_km']} km 
on {risk_event['closest_date']} at a relative speed of {risk_event['relative_speed_km_s']} km/s.

Suggest a safe, fuel-efficient avoidance maneuver in the following format:
1. Recommended Δv (m/s)
2. Direction (radial, tangential, or normal)
3. Justification for the choice
4. Expected change in miss distance
"""

# ------------------------
# Step 3: Send to LLM (or simulate)
# ------------------------
response_text = ""

if API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        print(f"⚠️ API Error: {e}")
else:
    # Simulated LLM output (for offline use)
    response_text = """
1. Recommended Δv: 0.07 m/s
2. Direction: Tangential (along orbital velocity vector)
3. Justification: A small tangential burn at perigee will increase orbital energy,
   raising the miss distance while minimizing fuel use.
4. Expected miss distance: ~1.2 km after 2 orbits.
"""

print("\n🤖 AI-Generated Maneuver Suggestion:\n")
print(response_text)
