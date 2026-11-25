"""Clinic clock: US Eastern Time (DST-aware) with a configurable override for testing.

Set CLINIC_NOW_OVERRIDE to e.g. "2026-12-01T20:30" (Eastern wall-clock time) to simulate
a call at that moment; leave it unset to use the real current time.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN_HOUR = 9
CLOSE_HOUR = 17
HOURS_TEXT = "Monday to Friday, 9 AM to 5 PM Eastern Time"


def parse_override(value: str) -> datetime:
    """Naive values are interpreted as Eastern wall-clock time."""
    dt = datetime.fromisoformat(value.strip())
    return dt.replace(tzinfo=ET) if dt.tzinfo is None else dt.astimezone(ET)


def is_open_at(dt: datetime) -> bool:
    et = dt.astimezone(ET)
    return et.weekday() < 5 and OPEN_HOUR <= et.hour < CLOSE_HOUR


def next_business_day(d: date) -> date:
    """First weekday strictly after `d`."""
    d = d + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


class Clock:
    def __init__(self, override: datetime | None = None) -> None:
        self._override = override.astimezone(ET) if override else None

    @classmethod
    def from_env(cls) -> "Clock":
        raw = os.getenv("CLINIC_NOW_OVERRIDE", "").strip()
        return cls(parse_override(raw) if raw else None)

    def set(self, dt: datetime | None) -> None:
        self._override = dt.astimezone(ET) if dt else None

    def now(self) -> datetime:
        return self._override or datetime.now(ET)

    def is_open(self) -> bool:
        return is_open_at(self.now())

    def next_business_day(self) -> date:
        return next_business_day(self.now().date())

    @property
    def simulated(self) -> bool:
        return self._override is not None
