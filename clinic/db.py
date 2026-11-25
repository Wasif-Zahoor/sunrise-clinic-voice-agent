"""SQLite schema, synthetic seed data and helpers.

Usage:  python -m clinic.db --init | --reset | --show
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config
from .security import hash_last4, mask_email, mask_phone

SEED_PATH = config.PROJECT_ROOT / "data" / "seed.json"

TABLES = (
    "audit_logs", "email_outbox", "callback_messages", "requests",
    "available_slots", "appointments", "medications", "patients",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    first_name TEXT NOT NULL, last_name TEXT NOT NULL, dob TEXT NOT NULL,
    phone TEXT, email TEXT,
    ssn_last4_hash TEXT,              -- keyed hash of last 4 only; full SSN is never stored
    insurance_provider TEXT, member_id TEXT, self_pay INTEGER DEFAULT 0,
    lab_status TEXT, balance_due REAL DEFAULT 0,
    is_new INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS medications (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    name TEXT NOT NULL, refills_remaining INTEGER NOT NULL, last_filled TEXT, pharmacy TEXT
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL REFERENCES patients(patient_id),
    doctor TEXT NOT NULL, starts_at TEXT NOT NULL, reason TEXT,
    status TEXT NOT NULL DEFAULT 'scheduled', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS available_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT, doctor TEXT NOT NULL, starts_at TEXT NOT NULL,
    UNIQUE (doctor, starts_at)
);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,         -- patient id, new-patient id, or 'UNVERIFIED'
    request_type TEXT NOT NULL, created_at TEXT NOT NULL, summary TEXT NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS callback_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id INTEGER REFERENCES requests(id),
    patient_id TEXT NOT NULL, caller_name TEXT NOT NULL, callback_number TEXT NOT NULL,
    reason TEXT NOT NULL, created_at TEXT NOT NULL, followup_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS email_outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT, to_email TEXT NOT NULL, subject TEXT NOT NULL,
    body TEXT NOT NULL, created_at TEXT NOT NULL, appointment_id INTEGER
);
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, call_id TEXT, patient_id TEXT,
    action TEXT NOT NULL, outcome TEXT NOT NULL, detail TEXT
);
"""


def get_db_path() -> Path:
    return Path(os.getenv("DB_PATH") or config.PROJECT_ROOT / "data" / "clinic.db")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path) if path else get_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, timeout=10, isolation_level=None)  # autocommit; explicit transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def to_starts_at(date_str: str, time_str: str) -> str:
    """('2026-12-01', '9:30 AM') -> '2026-12-01T09:30'"""
    t = datetime.strptime(time_str.strip().upper(), "%I:%M %p")
    return f"{date_str}T{t.strftime('%H:%M')}"


def _seed(conn: sqlite3.Connection) -> None:
    data = json.loads(SEED_PATH.read_text(encoding="utf8"))
    with transaction(conn):
        for p in data["patients"]:
            conn.execute(
                "INSERT INTO patients (patient_id, first_name, last_name, dob, phone, ssn_last4_hash,"
                " lab_status, balance_due, is_new, created_at) VALUES (?,?,?,?,?,?,?,?,0,?)",
                (p["patient_id"], p["first_name"], p["last_name"], p["dob"], p["phone"],
                 hash_last4(p["ssn_last4"]), p["lab_results_status"], p["balance_due"], now_iso()),
            )
            for m in p["medications"]:
                conn.execute(
                    "INSERT INTO medications (patient_id, name, refills_remaining, last_filled, pharmacy)"
                    " VALUES (?,?,?,?,?)",
                    (p["patient_id"], m["name"], m["refills_remaining"], m["last_filled"], m["pharmacy"]),
                )
            if p.get("next_appointment"):
                when, doctor = p["next_appointment"].split(" with ")
                d, t = when.split(" ", 1)
                conn.execute(
                    "INSERT INTO appointments (patient_id, doctor, starts_at, reason, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (p["patient_id"], doctor.strip(), to_starts_at(d, t), "follow-up", now_iso()),
                )
        for block in data["available_slots"]:
            for t in block["times"]:
                conn.execute(
                    "INSERT INTO available_slots (doctor, starts_at) VALUES (?,?)",
                    (block["doctor"], to_starts_at(block["date"], t)),
                )


def init_db(path: str | Path | None = None, reset: bool = False) -> Path:
    """Create tables and load synthetic data if empty. reset=True wipes everything first."""
    conn = connect(path)
    try:
        if reset:
            for t in TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0:
            _seed(conn)
    finally:
        conn.close()
    return Path(path) if path else get_db_path()


def show(path: str | Path | None = None) -> None:
    """Print saved requests, callbacks, outbox and audit rows (contact details masked)."""
    conn = connect(path)

    def rows(sql: str):
        return conn.execute(sql).fetchall()

    print(f"DB: {path or get_db_path()}")
    print(f"\nSlots left: {rows('SELECT COUNT(*) c FROM available_slots')[0]['c']}")
    print("\n== requests")
    for r in rows("SELECT * FROM requests ORDER BY id"):
        print(f"  #{r['id']} {r['created_at']} {r['patient_id']:<10} {r['request_type']:<16} {r['status']:<26} {r['summary']}")
    print("\n== callback_messages")
    for r in rows("SELECT * FROM callback_messages ORDER BY id"):
        print(f"  #{r['id']} {r['patient_id']:<10} {r['caller_name']} {mask_phone(r['callback_number'])} follow-up {r['followup_date']}: {r['reason']}")
    print("\n== appointments")
    for r in rows("SELECT * FROM appointments ORDER BY id"):
        print(f"  #{r['id']} {r['patient_id']:<10} {r['doctor']} {r['starts_at']} {r['status']} ({r['reason']})")
    print("\n== email_outbox (simulated, never sent)")
    for r in rows("SELECT * FROM email_outbox ORDER BY id"):
        print(f"  #{r['id']} to {mask_email(r['to_email'])}: {r['subject']}")
    print("\n== audit_logs (last 25)")
    for r in rows("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 25")[::-1]:
        print(f"  {r['ts']} call={r['call_id']} patient={r['patient_id']} {r['action']} -> {r['outcome']} {r['detail'] or ''}")
    conn.close()


if __name__ == "__main__":
    config.load_env()
    arg = sys.argv[1] if len(sys.argv) > 1 else "--init"
    if arg == "--show":
        show()
    elif arg in ("--init", "--reset"):
        print("Database ready:", init_db(reset=(arg == "--reset")))
    else:
        raise SystemExit("usage: python -m clinic.db [--init | --reset | --show]")
