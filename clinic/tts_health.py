"""Which TTS providers just failed, remembered across calls (and restarts) for a short while.

Why: a provider whose network path is bad (measured here: Deepgram timing out for many seconds) costs the NEXT call its first sentences
too, because every call used to find out again by waiting for a deadline: in a real call the greeting had 5 s and then 9 s of silence
before the local voice spoke. A provider that was given up on is now skipped by the next calls for a few minutes, then tried again.

Counts nothing and stores no text: only provider names, a reason such as "APITimeoutError" and a time. The file is
data/tts_health.json (git-ignored; TTS_HEALTH_FILE overrides; TTS_HEALTH=off disables it). Every failure to read or write it is ignored:
this must never break a call.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Callable

from . import config

logger = logging.getLogger("clinic.tts")

COOLDOWN_TRANSIENT = 600.0        # seconds a provider that timed out / errored twice in a row is skipped by later calls
COOLDOWN_PERMANENT = 3600.0       # seconds for a bad key, no credit or an exhausted daily limit (401 / 402 / 403 / 429)


def health_path() -> Path:
    raw = (os.getenv("TTS_HEALTH_FILE") or "data/tts_health.json").strip()
    p = Path(raw)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def enabled() -> bool:
    return (os.getenv("TTS_HEALTH") or "on").strip().lower() not in ("off", "0", "false", "no")


class ProviderHealth:
    def __init__(self, path: Path | str | None = None, now: Callable[[], float] = time.time) -> None:
        self.path = Path(path) if path else health_path()
        self._now = now

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass

    def cooling(self) -> dict[str, dict]:
        """{provider: {'until': epoch, 'why': text, 'minutes_left': n}} for providers still in their cooldown."""
        out = {}
        for name, v in self._load().items():
            try:
                left = float(v["until"]) - self._now()
            except (KeyError, TypeError, ValueError):
                continue
            if left > 0:
                out[name] = {"until": float(v["until"]), "why": str(v.get("why", "")), "minutes_left": max(1, round(left / 60))}
        return out

    def mark_down(self, name: str, seconds: float, why: str) -> None:
        data = self._load()
        data[name] = {"until": self._now() + seconds, "why": why[:80]}
        self._save(data)

    def mark_ok(self, name: str) -> None:
        data = self._load()
        if name in data:                                  # only touch the file when something has to change
            del data[name]
            self._save(data)
