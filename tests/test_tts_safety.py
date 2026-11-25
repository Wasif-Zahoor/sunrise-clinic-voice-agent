"""TTS hardening: sentence grouping, request pacing, 429 handling and graceful degradation.
The real SafeGroqTTS class is exercised against a local fake Groq server (no quota is used)."""

from __future__ import annotations

import asyncio
import io
import wave

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from livekit.agents import APIStatusError, tokenize

from clinic.tts_safety import (MAX_CHARS, GroupedSentenceTokenizer, RateWindow, SafeGroqTTS, get_limiter,
                               group_sentences)


# ===== sentence grouping ==========================================================================
def test_group_merges_short_sentences_up_to_the_cap():
    out = group_sentences(["Thank you for calling.", "This call is not being recorded.", "How can I help?"])
    assert out == ["Thank you for calling. This call is not being recorded. How can I help?"]


def test_group_never_exceeds_the_orpheus_limit():
    sentences = [("Word " * 12).strip() + "." for _ in range(8)]          # 8 sentences of ~60 chars
    out = group_sentences(sentences)
    assert len(out) >= 3 and all(len(c) <= MAX_CHARS for c in out) and MAX_CHARS < 200
    assert " ".join(out).replace("  ", " ") == " ".join(sentences)         # nothing lost, order kept


def test_a_single_overlong_sentence_is_split_at_a_natural_break():
    long = ("This is a very long sentence that keeps going, and going, with many clauses; "
            "it would be rejected by the speech service because it is far more than two hundred characters in total, "
            "so it must be broken up safely")
    out = group_sentences([long])
    assert len(out) >= 2 and all(len(c) <= MAX_CHARS for c in out)
    assert " ".join(out).split() == long.split()


@pytest.mark.parametrize("text,expected_requests", [
    # The first chunk of a reply is kept short (FIRST_CHUNK_CHARS), because the caller waits for all of it before hearing anything;
    # the rest of the reply is grouped into as few requests as before.
    ("Thank you for calling Sunrise Family Clinic. This call is not being recorded. I'm the clinic's voice "
     "receptionist. How can I help you today?", 2),
    ("Metformin 500 mg to be sent to CVS Main St. Is that right?", 1),
    ("Your next appointment is Thursday, December 3rd at 10:30 AM with Dr. Patel. Is there anything else?", 2),
])
def test_typical_replies_become_a_few_requests_instead_of_one_per_sentence(text, expected_requests):
    plain = tokenize.blingfire.SentenceTokenizer(retain_format=True).tokenize(text)
    grouped = GroupedSentenceTokenizer().tokenize(text)
    assert len(plain) >= 2 and len(grouped) == expected_requests
    assert len(grouped[0]) <= 90 or len(grouped) == 1


def test_grouped_stream_emits_chunks_at_end_of_input():
    async def run(chunks):
        s = GroupedSentenceTokenizer().stream()
        for c in chunks:
            s.push_text(c)
        s.end_input()
        return [ev.token async for ev in s]

    out = asyncio.run(run(["Hello there. ", "I can ", "help you. ", "What do you need?"]))
    assert out == ["Hello there. I can help you. What do you need?"]
    long = asyncio.run(run(["Sentence number one is here. " * 12]))
    assert len(long) >= 2 and all(len(c) <= MAX_CHARS for c in long)


# ===== pacing =========================================================================================
class FakeTime:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def now(self):
        return self.t

    async def sleep(self, s):
        self.slept.append(s)
        self.t += s


@pytest.mark.asyncio
async def test_rate_window_allows_a_burst_then_delays_only_the_excess():
    ft = FakeTime()
    w = RateWindow(3, per=10, clock=ft.now, sleep=ft.sleep)
    for _ in range(3):
        assert await w.acquire(max_wait=30)
    assert ft.slept == []                                   # the first three go straight through
    assert await w.acquire(max_wait=30)
    assert len(ft.slept) == 1 and 9.9 < ft.slept[0] < 10.2  # the fourth waits for the oldest to expire


@pytest.mark.asyncio
async def test_rate_window_gives_up_instead_of_blocking_for_too_long():
    ft = FakeTime()
    w = RateWindow(1, per=60, clock=ft.now, sleep=ft.sleep)
    assert await w.acquire(max_wait=5)
    assert not await w.acquire(max_wait=5) and ft.slept == []


@pytest.mark.asyncio
async def test_rate_window_zero_disables_pacing_and_limiters_are_shared():
    assert await RateWindow(0).acquire(0)
    assert get_limiter(9) is get_limiter(9) and get_limiter(9) is not get_limiter(5)


# ===== SafeGroqTTS against a fake Groq =================================================================
def _wav(seconds=0.1, rate=48000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(bytes(int(rate * seconds) * 2))
    return buf.getvalue()


class FakeGroq:
    """Plays back a script of responses: ('ok',) | ('429', retry_after, message) | ('500',)."""

    def __init__(self, script):
        self.script, self.calls, self.bodies = list(script), 0, []

    async def handler(self, request: web.Request) -> web.Response:
        self.calls += 1
        self.bodies.append(await request.json())
        kind = self.script.pop(0) if self.script else ("ok",)
        if kind[0] == "429":
            return web.json_response({"error": {"message": kind[2]}}, status=429, headers={"retry-after": str(kind[1])})
        if kind[0] == "500":
            return web.json_response({"error": {"message": "boom"}}, status=500)
        return web.Response(body=_wav(), content_type="audio/wav",
                            headers={"x-ratelimit-remaining-requests": "7", "x-ratelimit-reset-requests": "1h"})


async def _speak(script, *, limiter=None, max_wait=12.0, text="Hello there."):
    fake = FakeGroq(script)
    app = web.Application()
    app.router.add_post("/audio/speech", fake.handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        tts = SafeGroqTTS(base_url=f"http://127.0.0.1:{server.port}", api_key="test-key-not-real",
                          http_session=session, limiter=limiter, max_wait=max_wait)
        frames, error = [], None
        try:
            async with tts.synthesize(text) as stream:
                async for ev in stream:
                    frames.append(ev.frame)
        except Exception as e:                                  # noqa: BLE001
            error = e
        return fake, tts, frames, error


MINUTE = "Rate limit reached for model `canopylabs/orpheus-v1-english` on requests per minute (RPM): Limit 10"
DAILY = "Rate limit reached for model `canopylabs/orpheus-v1-english` on requests per day (RPD): Limit 100"


@pytest.mark.asyncio
async def test_normal_request_returns_audio_and_tracks_the_remaining_budget():
    fake, tts, frames, error = await _speak([("ok",)])
    assert error is None and frames and fake.calls == 1
    assert fake.bodies[0]["input"] == "Hello there." and fake.bodies[0]["response_format"] == "wav"
    assert tts.stats["requests"] == 1 and tts.stats["remaining_requests"] == 7 and tts.stats["degraded"] == 0


@pytest.mark.asyncio
async def test_per_minute_429_honours_retry_after_once_then_succeeds():
    fake, tts, frames, error = await _speak([("429", 0, MINUTE), ("ok",)])
    assert error is None and frames and fake.calls == 2
    assert tts.stats["retry_after_waits"] == 1 and tts.stats["degraded"] == 0


@pytest.mark.asyncio
async def test_persistent_429_degrades_to_silence_instead_of_raising():
    fake, tts, frames, error = await _speak([("429", 0, MINUTE), ("429", 0, MINUTE)])
    assert error is None and frames                         # silence, not an exception
    assert fake.calls == 2 and tts.stats["degraded"] == 1  # no hammering: exactly one retry


@pytest.mark.asyncio
async def test_daily_quota_429_is_never_retried_and_never_fatal():
    fake, tts, frames, error = await _speak([("429", 3600, DAILY)])
    assert error is None and frames and fake.calls == 1
    assert tts.stats["degraded"] == 1 and tts.stats["retry_after_waits"] == 0


@pytest.mark.asyncio
async def test_a_long_retry_after_is_not_waited_out():
    fake, tts, frames, error = await _speak([("429", 45, MINUTE)], max_wait=12)
    assert error is None and frames and fake.calls == 1 and tts.stats["degraded"] == 1


@pytest.mark.asyncio
async def test_full_local_window_skips_the_request_entirely():
    ft = FakeTime()
    w = RateWindow(1, per=60, clock=ft.now, sleep=ft.sleep)
    await w.acquire(1)                                      # window already used up
    fake, tts, frames, error = await _speak([("ok",)], limiter=w, max_wait=2)
    assert error is None and frames and fake.calls == 0 and tts.stats["degraded"] == 1


@pytest.mark.asyncio
async def test_other_server_errors_still_raise_so_real_bugs_are_not_hidden():
    _, _, frames, error = await _speak([("500",)] * 10)        # LiveKit retries a few times, then gives up
    assert not frames and error is not None and "500" in str(error)


@pytest.mark.asyncio
async def test_spoken_text_is_never_logged(caplog):
    caplog.set_level("DEBUG")
    await _speak([("429", 3600, DAILY)], text="Sarah Johnson, your balance is forty-five dollars.")
    assert "Sarah" not in caplog.text and "balance" not in caplog.text and "degraded" in caplog.text.lower()


# ===== wiring ============================================================================================
@pytest.mark.asyncio
async def test_session_uses_grouped_paced_orpheus_and_keeps_phase1_settings(state, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    monkeypatch.setenv("TTS_MAX_RPM", "7")
    monkeypatch.setenv("TTS_PROVIDER_ORDER", "groq")          # the Groq-only chain keeps the older safeguards
    import agent
    s = agent.build_session(state=state)
    assert s.tts.model == "canopylabs/orpheus-v1-english"
    groq_tts = s.tts._wrapped_tts.providers[0].tts
    assert isinstance(groq_tts, SafeGroqTTS) and isinstance(s.tts._sentence_tokenizer, GroupedSentenceTokenizer)
    assert groq_tts.limiter.max_requests == 7
    assert s.options.preemptive_generation["enabled"] is False and s.options.interruption["mode"] == "vad"


# ===== the daily character limit (tokens per day) ============================================================
TPD = ("Rate limit reached for model `canopylabs/orpheus-v1-english` in organization `org_SECRET123` service tier "
       "`on_demand` on tokens per day (TPD): Limit 3600, Used 3592, Requested 139. Please try again in 52m24s.")


def test_the_daily_limit_message_is_summarised_without_the_organisation_id():
    from clinic.tts_safety import describe_limit
    assert describe_limit(TPD) == "TPD: limit 3600, used 3592"
    assert "org_" not in describe_limit(TPD)
    assert describe_limit("something else") == "limit reached"


async def _speak_twice(script):
    """Two chunks in a row against the same fake server; returns (calls, stats, clock_state)."""
    fake = FakeGroq(script)
    app = web.Application()
    app.router.add_post("/audio/speech", fake.handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        tts = SafeGroqTTS(base_url=f"http://127.0.0.1:{server.port}", api_key="test-key-not-real", http_session=session)
        clock = {"t": 1000.0}
        tts.now = lambda: clock["t"]
        results = []
        for advance in (0, 10, 4000):                   # second chunk 10 s later, third one over an hour later
            clock["t"] += advance
            async with tts.synthesize("Hello there.") as stream:
                frames = [ev.frame async for ev in stream]
            results.append((fake.calls, bool(frames)))
        return results, tts


@pytest.mark.asyncio
async def test_after_the_daily_limit_the_agent_stops_asking_until_it_resets():
    results, tts = await _speak_twice([("429", 3144, TPD), ("ok",)])
    assert results[0] == (1, True)              # first chunk: one request, rejected, silence played
    assert results[1] == (1, True)              # 10 s later: NO second request (cooldown), still silent but alive
    assert results[2] == (2, True)              # over an hour later: asks again and succeeds
    assert tts.stats["degraded"] == 2 and tts.stats["chars_spoken"] == len("Hello there.")


@pytest.mark.asyncio
async def test_characters_spoken_are_counted_and_a_warning_comes_before_the_limit(caplog):
    caplog.set_level("INFO")
    fake = FakeGroq([("ok",)] * 10)
    app = web.Application()
    app.router.add_post("/audio/speech", fake.handler)
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        tts = SafeGroqTTS(base_url=f"http://127.0.0.1:{server.port}", api_key="test-key-not-real", http_session=session)
        tts.daily_char_allowance = 100
        for _ in range(3):                                          # 3 x 40 characters = 120 > 80% of 100
            async with tts.synthesize("x" * 40) as stream:
                [ev async for ev in stream]
    assert tts.stats["chars_spoken"] == 120
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "free allowance" in r.getMessage()]
    assert len(warnings) == 1                                       # warned once, not on every chunk
    assert "xxxx" not in caplog.text                                # the spoken text is never logged
