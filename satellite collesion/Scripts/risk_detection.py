# risk_detection.py
# Author: Harsh Nema
# Purpose: Detect minimum distance between two satellite orbits

import pandas as pd
import numpy as np

# Load NASA ISS Orbit
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
df1 = pd.DataFrame(lines, columns=columns)

# Convert to numeric
for col in df1.columns[2:]:
    df1[col] = pd.to_numeric(df1[col], errors="coerce")

# Simulate another satellite nearby
df2 = df1.copy()
df2["X_km"] += np.random.uniform(-50, 50, len(df1))
df2["Y_km"] += np.random.uniform(-50, 50, len(df1))
df2["Z_km"] += np.random.uniform(-50, 50, len(df1))

# Calculate distance between each time point
df1["Distance_km"] = np.sqrt(
    (df1["X_km"] - df2["X_km"])**2 +
    (df1["Y_km"] - df2["Y_km"])**2 +
    (df1["Z_km"] - df2["Z_km"])**2
)

# Find closest approach
closest = df1.loc[df1["Distance_km"].idxmin()]

print("🚨 Closest Approach Detected:")
print(f"📅 Date/Time: {closest['Calendar_Date']}")
print(f"📏 Distance: {closest['Distance_km']:.2f} km")
