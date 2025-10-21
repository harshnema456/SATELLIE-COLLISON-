# database_manager.py
# Author: Harsh Nema
# Purpose: Handle SQLite database for mission events and AI reports

import sqlite3
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/mission_data.db")

def init_db():
    """Create database & table if not exists"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS mission_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            satellite_1 TEXT,
            satellite_2 TEXT,
            min_distance REAL,
            relative_speed REAL,
            maneuver_plan TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_report(event, maneuver_text):
    """Insert a new report row"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO mission_reports 
        (date, satellite_1, satellite_2, min_distance, relative_speed, maneuver_plan)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        event["satellite_1"],
        event["satellite_2"],
        event["min_distance_km"],
        event["relative_speed_km_s"],
        maneuver_text
    ))
    conn.commit()
    conn.close()
    print("💾 Report saved successfully to mission_data.db!")

def get_all_reports():
    """Fetch all stored reports"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM mission_reports ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return rows
