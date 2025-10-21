import sys
import json
import base64
import requests
import re
import pandas as pd
import os

# ==========================================
# NASA Horizons API Configuration
# ==========================================
BASE_URL = 'https://ssd.jpl.nasa.gov/api/horizons.api'
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# ==========================================
# Function 1: Fetch SPK (Trajectory) File
# ==========================================
def fetch_spk_file(spkid, start_time='2030-01-01', stop_time='2031-01-01'):
    """
    Fetch a binary SPK trajectory file from NASA Horizons API.
    SPK contains high-precision orbital vectors.
    """
    print(f"📡 Fetching SPK file for SPK-ID: {spkid} ...")

    # Build the API URL
    url = (
        f"{BASE_URL}?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
        f"&COMMAND='DES%3D{spkid}%3B'&START_TIME='{start_time}'&STOP_TIME='{stop_time}'"
    )

    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Failed to connect: HTTP {response.status_code}")
        sys.exit(1)

    try:
        data = json.loads(response.text)
    except ValueError:
        print("❌ Unable to decode JSON from response.")
        sys.exit(1)

    # Check for valid SPK
    if "spk" in data:
        filename = f"{data.get('spk_file_id', spkid)}.bsp"
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(data["spk"]))
        print(f"✅ SPK trajectory file saved: {filepath}")
    else:
        print("⚠️ SPK file not generated. Response:")
        print(data.get("result", response.text))
        sys.exit(1)

# ==========================================
# Function 2: Fetch OBSERVER Data (Readable Table)
# ==========================================
def fetch_observer_data(command="'499'", start="'2006-01-01'", stop="'2006-01-20'", step="'1 d'"):
    """
    Fetch observational ephemeris data (RA, DEC, magnitude, range, etc.).
    """
    print("🔭 Fetching OBSERVER data (RA/DEC, brightness, range)...")

    params = {
        "format": "text",
        "COMMAND": command,
        "OBJ_DATA": "YES",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "OBSERVER",
        "CENTER": "'500@399'",
        "START_TIME": start,
        "STOP_TIME": stop,
        "STEP_SIZE": step,
        "QUANTITIES": "'1,9,20,23,24,29'",
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        print(f"❌ Failed to fetch OBSERVER data: HTTP {response.status_code}")
        sys.exit(1)

    result = response.text

    # Save raw text
    raw_path = os.path.join(DATA_DIR, "observer_raw.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(result)

    # Extract $$SOE/$$EOE data
    match = re.search(r"\$\$SOE(.*?)\$\$EOE", result, re.DOTALL)
    if not match:
        print("⚠️ No $$SOE/$$EOE data block found in output.")
        return

    block = match.group(1).strip()
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    records = []
    for line in lines:
        # Split each line by 2 or more spaces
        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 3:
            records.append(parts)

    if not records:
        print("⚠️ No numeric data rows found in block.")
        return

    # ✅ FIX: Dynamically detect how many columns NASA returned
    col_count = len(records[0])

    # Default header list (long enough to handle all cases)
    default_cols = [
        "Date_UT", "RA", "DEC", "APmag", "S_brt",
        "Delta", "Deldot", "Alpha", "SunLon", "Extra"
    ]

    # Take only as many headers as needed
    cols = default_cols[:col_count]

    # Create dataframe dynamically with correct number of headers
    df = pd.DataFrame(records, columns=cols)

    csv_path = os.path.join(DATA_DIR, "observer_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Observer data saved: {csv_path}")
    print("📄 Raw text file saved:", raw_path)

# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("🚀 NASA Horizons Unified Fetcher (SPK + Observer Data)\n")

    # If user didn’t pass a SPK-ID, show usage help
    if len(sys.argv) == 1:
        print("Usage Example:")
        print("  python scripts/nasa_fetcher.py 2000001     # (SPK-ID for 1 Ceres)")
        print("  python scripts/nasa_fetcher.py 25544       # (SPK-ID for ISS)")
        print("\nYou can change COMMAND in code to '499' for Mars or other objects.")
        sys.exit(2)

    spkid = sys.argv[1]

    # 1️⃣ Fetch SPK trajectory file
    fetch_spk_file(spkid)

    # 2️⃣ Fetch readable observational data
    fetch_observer_data()

    print("\n✅ All NASA JPL Horizons data successfully fetched & saved!")
