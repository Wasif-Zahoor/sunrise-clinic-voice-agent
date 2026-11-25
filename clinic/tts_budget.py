"""How many characters of speech each TTS provider has been asked to speak: per call, per day, and in total.

Deepgram's credit is one-time, so the running total matters more than a daily limit. Counts only: no text is stored or logged.
The counter lives in a small JSON file (data/tts_usage.json, git-ignored; TTS_USAGE_FILE overrides). Each call runs in its own
process, so the file is re-read before every update; two simultaneous calls could in theory lose a few characters.
Counts are characters REQUESTED (what the provider is billed on when the request succeeds), so they are an upper bound.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Callable

from . import config

logger = logging.getLogger("clinic.tts")

DEFAULT_WARN_TOTAL = 100_000          # characters per metered provider; override with TTS_CHAR_WARN
FREE_PROVIDERS = {"local"}            # no cost and no quota, never warned about


def usage_path() -> Path:
    raw = (os.getenv("TTS_USAGE_FILE") or "data/tts_usage.json").strip()
    p = Path(raw)
    return p if p.is_absolute() else config.PROJECT_ROOT / p


def warn_threshold() -> int:
    try:
        return int(os.getenv("TTS_CHAR_WARN") or DEFAULT_WARN_TOTAL)
    except ValueError:
        return DEFAULT_WARN_TOTAL


class CharBudget:
    def __init__(self, call_id: str = "", path: Path | str | None = None, warn_total: int | None = None,
                 today: Callable[[], date] = date.today) -> None:
        self.call_id = call_id
        self.path = Path(path) if path else usage_path()
        self.warn_total = warn_total if warn_total is not None else warn_threshold()
        self._today = today
        self.call_chars: dict[str, int] = {}
        self._warned: set[str] = set()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:                                      # a read-only disk must not break the call
            logger.warning("TTS usage file not saved (%s)", type(e).__name__)

    def note(self, provider: str, chars: int) -> dict[str, int]:
        """Record `chars` requested from `provider`; returns {'call', 'today', 'total'} and warns once at the threshold."""
        day = self._today().isoformat()
        data = self._load()
        entry = data.setdefault(provider, {"total": 0, "days": {}})
        entry["total"] = int(entry.get("total", 0)) + chars
        entry["days"][day] = int(entry["days"].get(day, 0)) + chars
        self._save(data)
        self.call_chars[provider] = self.call_chars.get(provider, 0) + chars
        counts = {"call": self.call_chars[provider], "today": entry["days"][day], "total": entry["total"]}
        logger.info("TTS usage provider=%s chunk=%d call=%d today=%d total=%d", provider, chars, counts["call"],
                    counts["today"], counts["total"])
        if provider not in FREE_PROVIDERS and self.warn_total > 0 and counts["total"] >= self.warn_total \
                and provider not in self._warned:
            self._warned.add(provider)
            logger.warning("TTS provider %s: %d characters used in total (warning threshold %d, set TTS_CHAR_WARN). "
                           "Check how much credit is left.", provider, counts["total"], self.warn_total)
        return counts

    def summary(self) -> dict[str, dict]:
        """{provider: {'total': n, 'today': n}} from the file (for the report command)."""
        day = self._today().isoformat()
        return {p: {"total": int(v.get("total", 0)), "today": int(v.get("days", {}).get(day, 0))}
                for p, v in self._load().items()}


if __name__ == "__main__":        # python -m clinic.tts_budget : how many characters each provider has been asked to speak
    config.load_env()
    rows = CharBudget().summary()
    print(f"TTS characters ({usage_path()}); warning threshold {warn_threshold()} per paid provider")
    for name, v in sorted(rows.items()) or [("(nothing recorded yet)", {"total": 0, "today": 0})]:
        print(f"  {name:12s} today {v['today']:>8}   total {v['total']:>9}")
