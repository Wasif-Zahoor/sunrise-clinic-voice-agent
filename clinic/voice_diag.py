"""Diagnostics for choppy voice: when does each piece of speech audio arrive, would the listener hear a gap, and was this program's own
event loop blocked at the time? Logs timings and counts only: never the spoken text.

Playback is modelled the way a listener experiences it: it starts when the first audio arrives and runs at normal speed; whenever
the audio received so far has run out before the next piece arrives, that wait is a break ("underrun")."""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from collections import deque

logger = logging.getLogger("clinic.tts")

UNDERRUN_WARN = 0.4          # seconds of silence inside one chunk that is reported as a break
LAG_WARN = 0.25              # an event-loop stall this long is logged on its own
HEARTBEAT = 0.05


class LoopMonitor:
    """Samples how late the event loop wakes up. A busy or blocked loop delays audio no matter how fast the provider is."""

    def __init__(self) -> None:
        self._lags: deque[tuple[float, float]] = deque(maxlen=400)       # (when, lateness) for anything over 50 ms
        self._task: asyncio.Task | None = None
        self.worst = 0.0

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._beat())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()

    async def _beat(self) -> None:
        while True:
            before = time.perf_counter()
            await asyncio.sleep(HEARTBEAT)
            late = time.perf_counter() - before - HEARTBEAT
            if late > 0.05:
                self._lags.append((time.perf_counter(), late))
                self.worst = max(self.worst, late)
                if late >= LAG_WARN:
                    logger.warning("VOICE loop lag %.2fs: this program's event loop was blocked or starved (CPU busy?)", late)

    def worst_since(self, t: float) -> float:
        return max((lag for at, lag in self._lags if at >= t), default=0.0)


class ChunkTrace:
    """Arrival times of one chunk of speech audio."""

    def __init__(self, provider: str, chars: int, idle: float | None) -> None:
        self.provider, self.chars, self.idle = provider, chars, idle
        self.started = time.perf_counter()
        self.arrivals: list[tuple[float, float]] = []                   # (seconds since start, audio seconds received so far)
        self.audio = 0.0

    def frame(self, seconds: float) -> None:
        self.audio += seconds
        self.arrivals.append((time.perf_counter() - self.started, self.audio))

    def result(self, lag: float = 0.0, held: bool = False) -> dict:
        """`held`: the whole clip was kept until complete before playing, so the network gaps were never heard (they are
        reported as `hidden`, and `underrun` is 0)."""
        if not self.arrivals:
            return {"provider": self.provider, "chars": self.chars, "frames": 0, "first_audio": 0.0, "done": 0.0, "audio": 0.0,
                    "max_gap": 0.0, "gap_at": 0.0, "underrun": 0.0, "hidden": 0.0, "held": held, "idle": self.idle, "lag": lag}
        first = self.arrivals[0][0]
        max_gap, gap_at, under, runs_out, got, prev = 0.0, 0.0, 0.0, first, 0.0, None
        for at, total in self.arrivals:
            if prev is not None and at - prev > max_gap:
                max_gap, gap_at = at - prev, got                        # the longest wait between two pieces, and how much audio preceded it
            if at > runs_out:                                           # the listener had run out of audio before this arrived
                under += at - runs_out
                runs_out = at
            runs_out += total - got
            got, prev = total, at
        return {"provider": self.provider, "chars": self.chars, "frames": len(self.arrivals), "first_audio": first,
                "done": self.arrivals[-1][0], "audio": self.audio, "max_gap": max_gap, "gap_at": gap_at,
                "underrun": 0.0 if held else under, "hidden": under if held else 0.0, "held": held,
                "idle": self.idle, "lag": lag}


def describe(r: dict) -> str:
    idle = "n/a" if r["idle"] is None else f"{r['idle']:.0f}s"
    heard = r["done"] if r.get("held") else r["first_audio"]                 # when the caller starts hearing it
    return (f"provider={r['provider']} chars={r['chars']} first_audio={r['first_audio']:.2f}s synthesis={r['done']:.2f}s "
            f"audio={r['audio']:.2f}s frames={r['frames']} max_gap={r['max_gap']:.2f}s@{r['gap_at']:.1f}s "
            f"held={'yes' if r.get('held') else 'no'} heard_after={heard:.2f}s underrun={r['underrun']:.2f}s "
            f"hidden_by_hold={r.get('hidden', 0.0):.2f}s idle_before={idle} loop_lag={r['lag']:.2f}s")


def verdict(r: dict) -> str:
    """One plain sentence on who is to blame for a break."""
    if r["lag"] >= max(0.3, 0.5 * r["max_gap"]):
        return "this program's event loop was blocked at the same time (CPU or another task), not only the provider"
    return "the provider or the network paused while sending audio; this program's event loop was not blocked"


def report_chunk(r: dict) -> None:
    logger.info("TTS chunk %s", describe(r))
    if r["underrun"] >= UNDERRUN_WARN:
        logger.warning("VOICE BREAK: the audio ran dry for %.1fs in one %s chunk (largest gap %.1fs, after %.1fs of audio, "
                       "idle %s before this chunk): %s", r["underrun"], r["provider"], r["max_gap"], r["gap_at"],
                       "n/a" if r["idle"] is None else f"{r['idle']:.0f}s", verdict(r))


def summary(chunks: list[dict]) -> str | None:
    if not chunks:
        return None
    first = [c["first_audio"] for c in chunks if c["frames"]]
    breaks = [c for c in chunks if c["underrun"] >= UNDERRUN_WARN]
    by: dict[str, int] = {}
    for c in chunks:
        by[c["provider"]] = by.get(c["provider"], 0) + 1
    idle_breaks = sum(1 for c in breaks if (c["idle"] or 0) >= 10)
    return (f"VOICE SUMMARY chunks={len(chunks)} by_provider={by} breaks={len(breaks)} "
            f"break_seconds={sum(c['underrun'] for c in chunks):.1f} hidden_by_hold_seconds={sum(c.get('hidden', 0.0) for c in chunks):.1f} "
            f"worst_gap={max(c['max_gap'] for c in chunks):.1f}s "
            f"breaks_after_10s_idle={idle_breaks} first_audio_median={statistics.median(first) if first else 0:.2f}s "
            f"first_audio_max={max(first) if first else 0:.2f}s worst_loop_lag={max(c['lag'] for c in chunks):.2f}s")
