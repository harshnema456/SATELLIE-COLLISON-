# orbit_visualization.py
# Author: Harsh Nema
# Purpose: Visualize NASA HORIZONS orbital data (ISS)

import pandas as pd
import matplotlib.pyplot as plt

# Load your NASA Horizons data file
data_file = "../data/horizons_results (1).txt"

# Parse the data between $$SOE and $$EOE
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
            # split by comma or spaces
            parts = [x.strip() for x in line.split(",") if x.strip()]
            if len(parts) >= 11:
                lines.append(parts)

# Convert to DataFrame
columns = [
    "JDTDB", "Calendar_Date", "X_km", "Y_km", "Z_km",
    "VX_km_s", "VY_km_s", "VZ_km_s", "LT_sec", "Range_km", "RangeRate_km_s"
]
df = pd.DataFrame(lines, columns=columns)

# Convert to numeric values
for col in df.columns[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"✅ Loaded {len(df)} data points from NASA HORIZONS")

# Plot Orbit (X vs Y)
plt.figure(figsize=(8, 8))
plt.plot(df["X_km"], df["Y_km"], color='blue', label="ISS Orbit Path")
plt.xlabel("X (km)")
plt.ylabel("Y (km)")
plt.title("ISS Orbit Projection - NASA HORIZONS Data")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
