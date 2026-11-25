"""Masking, SSN-last-4 hashing and the audit log. No PHI is written by this module."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import sqlite3
from datetime import datetime, timezone

audit_logger = logging.getLogger("clinic.audit")

_DEV_KEY = "dev-only-insecure-key-set-SSN_HASH_KEY-in-env"


def mask_dob(dob: str | None) -> str:
    """'1985-03-14' -> '****-**-14'"""
    return f"****-**-{dob[-2:]}" if dob and len(dob) >= 2 else "****-**-**"


def mask_ssn(_value: object = None) -> str:
    return "***-**-****"


def mask_phone(phone: str | None) -> str:
    digits = "".join(c for c in (phone or "") if c.isdigit())
    return f"***-***-**{digits[-2:]}" if len(digits) >= 2 else "***-***-****"


def mask_email(email: str | None) -> str:
    if not email or "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    return f"{local[:1]}***@{domain}"


def mask_name(name: str | None) -> str:
    return " ".join(p[:1].upper() + "." for p in (name or "").split()) or "?"


def _key() -> bytes:
    return (os.getenv("SSN_HASH_KEY") or _DEV_KEY).encode()


def hash_last4(last4: str) -> str:
    """Keyed hash of the last 4 SSN digits. Only 10,000 values exist, so this is a
    prototype safeguard, not strong protection (see README)."""
    return hmac.new(_key(), last4.encode(), hashlib.sha256).hexdigest()


def last4_matches(last4: str, stored_hash: str | None) -> bool:
    return bool(stored_hash) and hmac.compare_digest(hash_last4(last4), stored_hash)


def audit(
    conn: sqlite3.Connection,
    call_id: str,
    action: str,
    outcome: str,
    patient_id: str | None = None,
    detail: str = "",
) -> None:
    """Record who/what/when. `detail` must never contain PHI, DOB, SSN digits or free text."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO audit_logs (ts, call_id, patient_id, action, outcome, detail) VALUES (?,?,?,?,?,?)",
        (ts, call_id, patient_id, action, outcome, detail),
    )
    audit_logger.info("audit call=%s patient=%s action=%s outcome=%s", call_id, patient_id, action, outcome)
