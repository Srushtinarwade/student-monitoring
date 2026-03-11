"""
SQLite database logic for persistent session history.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
import config

DB_PATH: Path = config.LOG_DIR / "sessions.db"

def init_db() -> None:
    """Initialize the SQLite database with the sessions table."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                username TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                distraction_count INTEGER NOT NULL,
                phone_usage_seconds INTEGER NOT NULL,
                focus_score REAL NOT NULL
            )
            """
        )

def save_session(start_time: float, duration: float, distractions: int, phone_secs: float, focus_score: float) -> None:
    """Save a completed session to the database."""
    start_dt_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sessions (
                start_time, username, duration_seconds, 
                distraction_count, phone_usage_seconds, focus_score
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (start_dt_str, config.USERNAME, int(duration), distractions, int(phone_secs), focus_score)
        )

def get_recent_sessions(limit: int = 50) -> list[dict]:
    """Retrieve the most recent sessions."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]
