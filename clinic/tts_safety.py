"""Groq Orpheus TTS hardening: fewer requests, paced requests, no session-killing failures.

Why (measured, see README): the free tier allows 10 TTS requests/minute and 100/day, and LiveKit
makes one request per sentence. A 429 that exhausts retries counts as an unrecoverable error, and
three in a row close the whole session.

* GroupedSentenceTokenizer - merges sentences into chunks of at most 190 characters (Orpheus rejects
  more than 200), so a typical two-sentence reply is ONE request instead of two.
* RateWindow - sliding-window pacer: at most N requests per 60 s, delaying only when the window is full.
* SafeGroqTTS - honours Retry-After once, never retries a daily-quota 429, and on failure plays a
  short silence instead of raising, so the conversation state and tool actions stay intact.
  Nothing here logs the spoken text (it can contain patient information).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import replace
from collections.abc import Awaitable, Callable

import aiohttp
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents import APIConnectionError, APIError, APIStatusError, APITimeoutError, tokenize, tts, utils
from livekit.plugins import groq
from livekit.plugins.groq.tts import SAMPLE_RATE

logger = logging.getLogger("clinic.tts")

MAX_CHARS = 190          # Orpheus rejects more than 200 characters per request
FIRST_CHUNK_CHARS = 90   # the first chunk of a reply is kept short: the caller waits for all of it before hearing anything
SILENCE_SECONDS = 0.4


# ---- sentence grouping -------------------------------------------------------------------------
def _split_long(text: str, limit: int) -> list[str]:
    out = []
    while len(text) > limit:
        cut = max(text.rfind(", ", 0, limit), text.rfind("; ", 0, limit), text.rfind(": ", 0, limit))
        if cut < limit * 0.4:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(text[: cut + 1].strip())
        text = text[cut + 1:].strip()
    if text:
        out.append(text)
    return out


def group_sentences(sentences: list[str], max_chars: int = MAX_CHARS, first_max: int | None = None) -> list[str]:
    """Greedily merge sentences into chunks of at most max_chars; split any single longer sentence. The FIRST chunk is merged only
    up to first_max (when given), so that what the caller waits for before hearing anything stays short."""
    out: list[str] = []
    cur = ""
    for sentence in (s.strip() for s in sentences if s and s.strip()):
        for piece in _split_long(sentence, max_chars):
            cap = max_chars if out or first_max is None else first_max
            if not cur:
                cur = piece
            elif len(cur) + 1 + len(piece) <= cap:
                cur = f"{cur} {piece}"
            else:
                out.append(cur)
                cur = piece
    if cur:
        out.append(cur)
    return out


class GroupedSentenceTokenizer(tokenize.SentenceTokenizer):
    def __init__(self, max_chars: int = MAX_CHARS, base: tokenize.SentenceTokenizer | None = None,
                 first_max: int | None = FIRST_CHUNK_CHARS) -> None:
        self._max, self._first_max = max_chars, first_max
        self._base = base or tokenize.blingfire.SentenceTokenizer(retain_format=True)

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return group_sentences(self._base.tokenize(text, language=language), self._max, self._first_max)

    def stream(self, *, language: str | None = None) -> tokenize.SentenceStream:
        return _GroupedStream(self, language)


class _GroupedStream(tokenize.SentenceStream):
    """Collects the reply text and emits grouped chunks at end of input (or when it gets long)."""

    def __init__(self, owner: GroupedSentenceTokenizer, language: str | None) -> None:
        super().__init__()
        self._owner, self._lang = owner, language
        self._buf = ""
        self._segment_id = utils.shortuuid()

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        self._buf += text
        if len(self._buf) > 2 * self._owner._max:        # long monologue: do not wait for the end
            self._emit(keep_last=True)

    def flush(self) -> None:
        self._emit()

    def end_input(self) -> None:
        self._emit()
        self._do_close()

    async def aclose(self) -> None:
        self._do_close()

    def _emit(self, keep_last: bool = False) -> None:
        if self._event_ch.closed or not self._buf.strip():
            return
        groups = self._owner.tokenize(self._buf, language=self._lang)
        keep = groups.pop() if keep_last and len(groups) > 1 else ""
        for g in groups:
            self._event_ch.send_nowait(tokenize.TokenData(segment_id=self._segment_id, token=g))
        self._buf = keep


# ---- request pacing ----------------------------------------------------------------------------
class RateWindow:
    """At most `max_requests` per `per` seconds. Delays only when the window is full."""

    def __init__(self, max_requests: int, per: float = 60.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.max_requests, self.per = max_requests, per
        self._clock, self._sleep = clock, sleep
        self._stamps: deque[float] = deque()
        self._lock: asyncio.Lock | None = None

    async def acquire(self, max_wait: float) -> bool:
        """True once a slot is free; False if that would take longer than max_wait seconds."""
        if self.max_requests <= 0:
            return True
        self._lock = self._lock or asyncio.Lock()
        async with self._lock:
            while True:
                now = self._clock()
                while self._stamps and now - self._stamps[0] >= self.per:
                    self._stamps.popleft()
                if len(self._stamps) < self.max_requests:
                    self._stamps.append(now)
                    return True
                wait = self._stamps[0] + self.per - now + 0.05
                if wait > max_wait:
                    return False
                await self._sleep(wait)


_limiters: dict[int, RateWindow] = {}


def get_limiter(max_rpm: int) -> RateWindow:
    """One shared limiter per process: all calls share the same Groq quota."""
    return _limiters.setdefault(max_rpm, RateWindow(max_rpm))


# ---- the TTS ----------------------------------------------------------------------------------------
def _float(value: str | None) -> float:
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


_LIMIT_RE = re.compile(r"\((?P<kind>[A-Z]+)\): Limit (?P<limit>\d+), Used (?P<used>\d+)")
DAILY_CHARS_WARN = 0.8        # warn when this process has spoken 80% of the free daily allowance


def describe_limit(body: str) -> str:
    """'tokens per day (TPD): limit 3600, used 3592' - from Groq's 429 message, without the organisation id."""
    m = _LIMIT_RE.search(body or "")
    return f"{m['kind']}: limit {m['limit']}, used {m['used']}" if m else "limit reached"


def _is_daily_limit(body: str, retry_after: float) -> bool:
    low = body.lower()
    return "per day" in low or "(rpd)" in low or "(tpd)" in low or retry_after > 120


class SafeGroqTTS(groq.TTS):
    def __init__(self, *, limiter: RateWindow | None = None, max_wait: float = 12.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.limiter = limiter or RateWindow(0)
        self.max_wait = max_wait
        self.stats = {"requests": 0, "degraded": 0, "retry_after_waits": 0, "remaining_requests": None,
                      "chars_spoken": 0}
        self._low_budget_warned = False
        self._chars_warned = False
        self.daily_char_allowance = 3600       # Groq free tier: tokens per day (about one token per character)
        self.blocked_until = 0.0               # after a daily-limit 429 do not ask again until this time
        self.now = time.monotonic              # replaceable in tests

    def synthesize(self, text: str, *, conn_options=DEFAULT_API_CONNECT_OPTIONS) -> tts.ChunkedStream:
        return _SafeStream(tts=self, input_text=text, conn_options=conn_options)

    def note_spoken(self, chars: int) -> None:
        """Count characters actually sent for speech (never the text itself) and warn before the daily limit."""
        self.stats["chars_spoken"] += chars
        used = self.stats["chars_spoken"]
        logger.info("TTS chunk: %d characters (%d spoken by this agent process so far)", chars, used)
        if not self._chars_warned and used >= self.daily_char_allowance * DAILY_CHARS_WARN:
            self._chars_warned = True
            logger.warning("Groq TTS: about %d of the ~%d characters/day free allowance used by this process. When it runs "
                           "out the agent keeps working but goes silent.", used, self.daily_char_allowance)

    def note_budget(self, headers) -> None:
        remaining = headers.get("x-ratelimit-remaining-requests")
        if remaining is None or not str(remaining).isdigit():
            return
        self.stats["remaining_requests"] = int(remaining)
        if int(remaining) <= 10 and not self._low_budget_warned:
            self._low_budget_warned = True
            logger.warning("Groq TTS request budget is low: %s requests left (resets in %s)", remaining,
                           headers.get("x-ratelimit-reset-requests", "?"))


class _SafeStream(tts.ChunkedStream):
    def __init__(self, *, tts: SafeGroqTTS, input_text: str, conn_options) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._owner = tts
        self._opts = replace(tts._opts)          # snapshot of model / voice / key for this request

    def _degrade(self, emitter: tts.AudioEmitter, why: str, quiet: bool = False) -> None:
        """Skip speaking this chunk: play brief silence so the session carries on unharmed."""
        self._owner.stats["degraded"] += 1
        (logger.info if quiet else logger.error)(
            "TTS unavailable (%s); skipping this speech chunk. The conversation continues without audio "
            "for it. Degraded chunks so far: %d.", why, self._owner.stats["degraded"])
        emitter.initialize(request_id=utils.shortuuid(), sample_rate=SAMPLE_RATE, num_channels=1,
                           mime_type="audio/pcm")
        emitter.push(bytes(int(SAMPLE_RATE * SILENCE_SECONDS) * 2))
        emitter.flush()

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        owner = self._owner
        o = self._opts
        request_id = utils.shortuuid()
        if owner.now() < owner.blocked_until:                 # the daily limit was hit recently: no point asking again
            return self._degrade(output_emitter, f"daily limit still in effect (~{int(owner.blocked_until - owner.now())} s left)",
                                 quiet=True)
        if not await owner.limiter.acquire(owner.max_wait):
            return self._degrade(output_emitter, "local request pacing window is full")
        for attempt in range(2):
            try:
                async with owner._ensure_session().post(
                    f"{o.base_url}/audio/speech",
                    headers={"Authorization": f"Bearer {o.api_key}", "Content-Type": "application/json"},
                    json={"model": o.model, "voice": o.voice, "input": self._input_text, "response_format": "wav"},
                    timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
                ) as resp:
                    owner.stats["requests"] += 1
                    owner.note_budget(resp.headers)
                    if resp.status == 429:
                        body = await resp.text()
                        wait = _float(resp.headers.get("retry-after"))
                        if _is_daily_limit(body, wait):
                            owner.blocked_until = owner.now() + min(max(wait, 60.0), 3600.0)
                            return self._degrade(output_emitter, f"Groq daily limit reached (HTTP 429; {describe_limit(body)}; "
                                                                 f"retry in about {int(wait // 60)} min)")
                        if attempt == 1 or wait > owner.max_wait:
                            return self._degrade(output_emitter, "Groq per-minute limit (HTTP 429)")
                        owner.stats["retry_after_waits"] += 1
                        await asyncio.sleep(wait + 0.5)       # honour Retry-After, once
                        continue
                    if resp.status >= 400:
                        raise APIStatusError(message=f"Groq TTS HTTP {resp.status}", status_code=resp.status,
                                             request_id=request_id, body=None)
                    if not resp.content_type.startswith("audio"):
                        raise APIError(message="Groq returned non-audio data", body=await resp.text())
                    output_emitter.initialize(request_id=request_id, sample_rate=SAMPLE_RATE, num_channels=1,
                                              mime_type="audio/wav")
                    async for data, _ in resp.content.iter_chunks():
                        output_emitter.push(data)
                    output_emitter.flush()
                    owner.note_spoken(len(self._input_text))
                    return
            except asyncio.TimeoutError:
                raise APITimeoutError() from None
            except APIError:
                raise
            except aiohttp.ClientError as e:
                raise APIConnectionError() from e
