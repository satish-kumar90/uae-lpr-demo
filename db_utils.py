# db_utils.py
import sqlite3
import json

DB_PATH = "results.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            final_number TEXT,
            area TEXT,
            number TEXT,
            others TEXT,
            angle REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def insert_result(image_name, final_number, area, number, others, angle):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO results (image_name, final_number, area, number, others, angle)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        image_name,
        final_number,
        area,
        number,
        json.dumps(others),  # Save as string
        angle
    ))
    conn.commit()
    conn.close()

# Ensure table exists on import
init_db()
