"""Shared fixtures. Every test gets its own throwaway SQLite database seeded with the
synthetic data, and a fixed clinic clock, so tests never touch data/clinic.db or real time."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinic import config, db  # noqa: E402
from clinic.clock import ET, Clock  # noqa: E402
from clinic.state import CallState  # noqa: E402

config.load_env()

OPEN_AT = datetime(2026, 11, 30, 10, 0, tzinfo=ET)          # Monday 10:00 - open
CLOSED_EVENING = datetime(2026, 11, 30, 20, 30, tzinfo=ET)  # Monday 20:30 - closed
CLOSED_SATURDAY = datetime(2026, 11, 28, 11, 0, tzinfo=ET)  # Saturday - closed

# Synthetic callers from data/seed.json: (name, dob, spoken last-4)
SARAH = ("Sarah Johnson", "1985-03-14", "3344")
MICHAEL = ("Michael Brown", "1972-11-08", "1020")
MARIA = ("Maria Garcia", "1990-07-22", "9081")


@pytest.fixture
def db_path(tmp_path) -> str:
    p = tmp_path / "test.db"
    db.init_db(p)
    return str(p)


@pytest.fixture
def make_state(db_path):
    def _make(at: datetime = OPEN_AT) -> CallState:
        return CallState(db_path=db_path, clock=Clock(at))

    return _make


@pytest.fixture
def state(make_state) -> CallState:
    return make_state()


@pytest.fixture
def conn(db_path):
    c = db.connect(db_path)
    yield c
    c.close()


# Live LLM tests spend ~1.6K tokens per request against Groq's free-tier cap of 8K tokens/minute,
# so space them out.
import time as _time  # noqa: E402

_last_live = {"t": 0.0}


@pytest.fixture
def pace():
    wait = 50 - (_time.time() - _last_live["t"])
    if _last_live["t"] and wait > 0:
        _time.sleep(wait)
    yield
    _last_live["t"] = _time.time()


def require_real_llm_reply(text: str) -> str:
    """Live tests judge what the LLM SAID. If the agent fell back to its outage apology (quota, 503, timeout),
    nothing was judged, so skip instead of passing vacuously."""
    from agent import LLM_FALLBACK_SAY
    if LLM_FALLBACK_SAY.lower() in (text or "").lower():
        pytest.skip("LLM unavailable (quota or outage): no real reply to judge")
    return text
