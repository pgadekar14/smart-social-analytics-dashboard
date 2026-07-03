"""
db_helper.py
------------
Lightweight SQLite backend for logging and retrieving prediction history.
Uses Python's built-in sqlite3 module - no extra install required.

This gives the project a real RDBMS component (Create, Read, Delete operations)
to report on for the "Backend / RDBMS Tools" section of your project report.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/predictions.db"


def init_db():
    """Create the predictions table if it doesn't already exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            confidence REAL,
            source TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(text, sentiment, confidence, source="Live Prediction"):
    """Insert a new prediction record (Create)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (text, sentiment, confidence, source, created_at) VALUES (?, ?, ?, ?, ?)",
        (text, sentiment, float(confidence), source, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def get_history(limit=200):
    """Fetch recent prediction records, newest first (Read)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, text, sentiment, confidence, source, created_at "
        "FROM predictions ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def clear_history():
    """Delete all prediction records (Delete)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


def count_records():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM predictions")
    n = cur.fetchone()[0]
    conn.close()
    return n
