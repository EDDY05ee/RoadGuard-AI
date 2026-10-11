import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "roadguard.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        # Create the table if it does not exist.
        connection.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_code TEXT UNIQUE,
                location TEXT NOT NULL DEFAULT 'Unknown',
                hazard TEXT NOT NULL DEFAULT 'Other',
                severity TEXT NOT NULL DEFAULT 'Medium',
                description TEXT DEFAULT '',
                risk_score INTEGER NOT NULL DEFAULT 30,
                priority TEXT NOT NULL DEFAULT 'MEDIUM',
                suggested_action TEXT NOT NULL DEFAULT 'Inspect and schedule maintenance.',
                status TEXT NOT NULL DEFAULT 'Pending',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Check columns in an existing database.
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(reports)"
            ).fetchall()
        }

        additions = {
            "report_code": "TEXT",
            "location": "TEXT NOT NULL DEFAULT 'Unknown'",
            "hazard": "TEXT NOT NULL DEFAULT 'Other'",
            "severity": "TEXT NOT NULL DEFAULT 'Medium'",
            "description": "TEXT DEFAULT ''",
            "risk_score": "INTEGER NOT NULL DEFAULT 30",
            "priority": "TEXT NOT NULL DEFAULT 'MEDIUM'",
            "suggested_action": (
                "TEXT NOT NULL DEFAULT 'Inspect and schedule maintenance.'"
            ),
            "status": "TEXT NOT NULL DEFAULT 'Pending'",
            "created_at": "TIMESTAMP"
        }

        for column, definition in additions.items():
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE reports ADD COLUMN {column} {definition}"
                )

        # Fill missing timestamps and report codes in older records.
        rows = connection.execute(
            "SELECT * FROM reports"
        ).fetchall()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for row in rows:
            if "created_at" in row.keys() and not row["created_at"]:
                connection.execute(
                    "UPDATE reports SET created_at = ? WHERE id = ?",
                    (now, row["id"])
                )

            if "report_code" in row.keys() and not row["report_code"]:
                connection.execute(
                    "UPDATE reports SET report_code = ? WHERE id = ?",
                    (f"RG-OLD-{row['id']}", row["id"])
                )


def save_report(report):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO reports (
                report_code,
                location,
                hazard,
                severity,
                description,
                risk_score,
                priority,
                suggested_action,
                status,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report["report_code"],
            report["location"],
            report["hazard"],
            report["severity"],
            report.get("description", ""),
            report["risk_score"],
            report["priority"],
            report["suggested_action"],
            report.get("status", "Pending"),
            created_at
        ))


def get_reports():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT * FROM reports ORDER BY id DESC
        """).fetchall()

        return [dict(row) for row in rows]


def update_status(report_code, new_status):
    with get_connection() as connection:
        connection.execute("""
            UPDATE reports
            SET status = ?
            WHERE report_code = ?
        """, (new_status, str(report_code)))