"""Round 10 (third live call, 2026-11-25): the voice kept breaking. Measured from the caller's PC, Deepgram's audio arrived in bursts with
pauses of 12-30 s (one clip in three) while the control hosts answered 10x faster. The fix keeps each network clip until it is complete
and drops it BEFORE playing anything if it misses its deadline. Also: filler words used up the refill tries. Offline."""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from livekit.agents import tts

from clinic import refill_flow, voice_diag
from clinic import tts_chain as tc
from clinic.tts_budget import CharBudget
from clinic.tts_safety import GroupedSentenceTokenizer, group_sentences
from test_round9 import Call, verify_as
from test_tts_chain import SETTINGS, FakeTTS, audible, piper_files, speak  # noqa: F401  (piper_files is a fixture)


# ---- a provider that sends audio in bursts ------------------------------------------------------------------------------------------
class BurstTTS(tts.TTS):
    """Pushes 0.1 s of a distinctive sample value, waits `pause` seconds, pushes another 0.1 s."""

    def __init__(self, pause: float, value: bytes = b"\x20\x00") -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=24000, num_channels=1)
        self.pause, self.value, self.calls = pause, value, 0

    @property
    def model(self) -> str:
        return "burst"

    def synthesize(self, text, *, conn_options=None):
        return _BurstStream(tts=self, input_text=text, conn_options=conn_options or tc.DEFAULT_API_CONNECT_OPTIONS)


class _BurstStream(tts.ChunkedStream):
    async def _run(self, out):
        owner = self._tts
        owner.calls += 1
        out.initialize(request_id="r", sample_rate=24000, num_channels=1, mime_type="audio/pcm")
        out.push(owner.value * 2400)
        await asyncio.sleep(owner.pause)
        out.push(owner.value * 2400)
        out.flush()


def chain_of(tmp_path, *providers: tc.Provider, hold: bool = True) -> tc.TtsChain:
    chain = tc.TtsChain(list(providers), CharBudget("test", path=tmp_path / "usage.json"))
    chain.hold = hold
    return chain


def pcm(frames) -> bytes:
    return b"".join(bytes(f.data) for f in frames)


# ---- the clip is held until complete ---------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_clip_with_a_pause_in_it_is_played_whole_and_the_pause_is_reported_as_hidden(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    chain = chain_of(tmp_path, tc.Provider("deepgram", BurstTTS(0.3), hold=True))
    out = pcm(await speak(chain))
    assert out.count(b"\x20\x00") == 4800                                   # both pieces arrived, nothing lost
    r = chain.chunks[-1]
    assert r["held"] and r["underrun"] == 0.0 and r["hidden"] >= 0.15
    assert "held=yes" in caplog.text and "VOICE BREAK" not in caplog.text


@pytest.mark.asyncio
async def test_without_the_hold_the_same_pause_is_a_break_and_it_is_reported(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    chain = chain_of(tmp_path, tc.Provider("deepgram", BurstTTS(0.6), hold=False))
    await speak(chain)
    r = chain.chunks[-1]
    assert not r["held"] and r["underrun"] >= 0.4 and "VOICE BREAK" in caplog.text


@pytest.mark.asyncio
async def test_the_off_switch_plays_audio_as_it_arrives(tmp_path):
    chain = chain_of(tmp_path, tc.Provider("deepgram", BurstTTS(0.6), hold=True), hold=False)
    await speak(chain)
    assert not chain.chunks[-1]["held"]


@pytest.mark.asyncio
async def test_a_clip_that_misses_its_deadline_is_dropped_before_anything_plays_and_the_next_provider_speaks(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "HOLD_BASE", 0.3)
    monkeypatch.setattr(tc, "HOLD_PER_AUDIO_SECOND", 0.0)
    slow, local = BurstTTS(5.0, b"\x20\x00"), FakeTTS("local", ["ok"])
    chain = chain_of(tmp_path, tc.Provider("deepgram", slow, hold=True), tc.Provider("local", local))
    started = time.perf_counter()
    frames = await speak(chain)
    assert time.perf_counter() - started < 2.0, "must not wait for the stalled clip"
    assert audible(frames) and b"\x20\x00" not in pcm(frames), "not one sample of the stalled clip may be heard"
    assert chain.active == "local" and chain.strikes["deepgram"] == 1 and "deepgram" not in chain.dead
    assert local.calls == 1 and chain.stats["degraded"] == 0


@pytest.mark.asyncio
async def test_two_deadline_misses_in_a_row_give_the_provider_up_for_the_call(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "HOLD_BASE", 0.2)
    monkeypatch.setattr(tc, "HOLD_PER_AUDIO_SECOND", 0.0)
    slow, local = BurstTTS(5.0), FakeTTS("local", ["ok"])
    chain = chain_of(tmp_path, tc.Provider("deepgram", slow, hold=True), tc.Provider("local", local))
    await speak(chain, "one")
    await speak(chain, "two")
    assert "deepgram" in chain.dead
    asked = slow.calls
    await speak(chain, "three")
    assert slow.calls == asked                                              # not asked again


@pytest.mark.asyncio
async def test_the_deadline_grows_with_the_length_of_the_speech(tmp_path):
    chain = chain_of(tmp_path, tc.Provider("x", FakeTTS("x", ["ok"])))
    short, long = (tc._ChainStream(tts=chain, input_text=t, conn_options=tc.DEFAULT_API_CONNECT_OPTIONS)._deadline()
                   for t in ("Yes.", "x" * 171))
    assert 3.0 < short < 4.0 and 8.0 < long < 10.0                          # a healthy 171-character greeting takes about 6 s


def test_network_providers_are_held_and_the_local_voice_is_not(piper_files):
    env = {"DEEPGRAM_API_KEY": "dg-test-not-real", "PIPER_MODEL": str(piper_files),
           "CLOUDFLARE_API_TOKEN": "cf-test-not-real", "CLOUDFLARE_ACCOUNT_ID": "acct"}
    assert [(p.name, p.hold) for p in tc.build_providers(SETTINGS, env)] == [("deepgram", True), ("cloudflare", True), ("local", False)]


def test_the_hold_can_be_turned_off_from_the_environment(tmp_path):
    for value, expect in (("0", False), ("1", True), ("", True)):
        chain = tc.build_chain(SETTINGS, "t", {"DEEPGRAM_API_KEY": "dg-test-not-real", "TTS_HOLD": value},
                               CharBudget("t", path=tmp_path / "u.json"))
        assert chain.hold is expect


def test_default_order_puts_cloudflare_between_deepgram_and_the_local_voice():
    assert tc.provider_order({}) == ["deepgram", "cloudflare", "local"]


# ---- Cloudflare speaks with the same voice -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env,speaker", [
    ({}, "thalia"), ({"DEEPGRAM_TTS_MODEL": "aura-2-andromeda-en"}, "andromeda"),
    ({"DEEPGRAM_TTS_MODEL": "aura-2-andromeda-en", "CLOUDFLARE_TTS_SPEAKER": " Luna "}, "luna"), ({"DEEPGRAM_TTS_MODEL": "other"}, "thalia")])
def test_the_cloudflare_speaker_follows_the_deepgram_voice(env, speaker):
    assert tc.cloudflare_speaker(env) == speaker


@pytest.mark.asyncio
async def test_the_cloudflare_request_names_the_speaker(tmp_path):
    seen = {}

    async def handler(request: web.Request) -> web.Response:
        seen["body"] = await request.json()
        return web.Response(body=b"\x20\x00" * 4800, content_type="audio/mpeg")

    app = web.Application()
    app.router.add_post("/accounts/ACCT/ai/run/{model:.*}", handler)
    async with TestServer(app) as server:
        async with aiohttp.ClientSession() as session:
            cf = tc.CloudflareTTS(account_id="ACCT", token="cf-test-not-real", base_url=f"http://127.0.0.1:{server.port}",
                                  http_session=session, speaker="thalia")
            await speak(tc.TtsChain([tc.Provider("cloudflare", cf, hold=True)], CharBudget("t", path=tmp_path / "u.json")))
    assert seen["body"]["speaker"] == "thalia" and seen["body"]["encoding"] == "linear16"


# ---- the first chunk is short --------------------------------------------------------------------------------------------------------
GREETING = ["Thank you for calling Sunrise Family Clinic.", "This call may be recorded and transcribed for quality purposes.",
            "I'm the clinic's voice receptionist.", "How can I help you today?"]


def test_the_first_chunk_of_a_reply_stays_short_and_the_rest_is_still_grouped():
    chunks = group_sentences(GREETING, 190, first_max=90)
    assert chunks[0] == GREETING[0] and len(chunks) == 2 and all(len(c) <= 190 for c in chunks)
    assert chunks[1] == " ".join(GREETING[1:])
    assert group_sentences(GREETING, 190) == [" ".join(GREETING)]            # without the cap: as before


def test_the_tokenizer_applies_the_short_first_chunk():
    chunks = GroupedSentenceTokenizer().tokenize(" ".join(GREETING))
    assert chunks[0].startswith("Thank you for calling") and len(chunks[0]) <= 90 and len(chunks) == 2


def test_two_short_sentences_still_share_one_request():
    assert group_sentences(["Thank you.", "May I have your date of birth?"], 190, first_max=90) == [
        "Thank you. May I have your date of birth?"]


# ---- the diagnostics -----------------------------------------------------------------------------------------------------------------
def trace_with(arrivals, held=False, lag=0.0, idle=None):
    t = voice_diag.ChunkTrace("deepgram", 60, idle)
    t.arrivals, t.audio = arrivals, arrivals[-1][1]
    return t.result(lag=lag, held=held)


def test_a_gap_after_the_audio_ran_out_is_counted_as_a_break():
    r = trace_with([(1.0, 1.0), (1.2, 2.0), (4.2, 3.0)])                   # 2 s received by t=1.2 run out at t=3.0; the next piece comes at 4.2
    assert r["max_gap"] == pytest.approx(3.0) and r["gap_at"] == pytest.approx(2.0) and r["underrun"] == pytest.approx(1.2)


def test_audio_that_stays_ahead_of_playback_has_no_break():
    r = trace_with([(0.3, 1.0), (0.6, 2.0), (0.9, 3.0)])
    assert r["underrun"] == 0.0 and r["first_audio"] == pytest.approx(0.3)


def test_a_held_clip_reports_the_gap_as_hidden_not_heard():
    r = trace_with([(1.0, 1.0), (1.2, 2.0), (4.2, 3.0)], held=True)
    assert r["underrun"] == 0.0 and r["hidden"] == pytest.approx(1.2)


def test_the_verdict_separates_this_program_from_the_provider():
    assert "event loop was blocked at the same time" in voice_diag.verdict({"lag": 1.5, "max_gap": 2.0})
    assert "provider or the network" in voice_diag.verdict({"lag": 0.01, "max_gap": 20.0})


def test_the_summary_line_counts_breaks_and_has_no_text():
    chunks = [trace_with([(0.5, 1.0), (0.9, 2.0)]), trace_with([(1.0, 1.0), (5.0, 2.0)], idle=30.0)]
    line = voice_diag.summary(chunks)
    assert "chunks=2" in line and "breaks=1" in line and "breaks_after_10s_idle=1" in line and "by_provider={'deepgram': 2}" in line
    assert voice_diag.summary([]) is None


@pytest.mark.asyncio
async def test_the_loop_monitor_sees_a_blocked_event_loop(caplog):
    caplog.set_level(logging.INFO)
    m = voice_diag.LoopMonitor()
    m.start()
    await asyncio.sleep(0.12)
    t0 = time.perf_counter()
    time.sleep(0.4)                                                         # blocks the loop on purpose
    await asyncio.sleep(0.15)
    m.stop()
    assert m.worst_since(t0) >= 0.3 and "VOICE loop lag" in caplog.text


@pytest.mark.asyncio
async def test_the_chunk_log_and_the_call_summary_contain_no_spoken_text(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    chain = make_chain_for(tmp_path)
    await speak(chain, "Sarah Johnson, your balance is $45.00")
    chain.log_summary()
    assert "VOICE SUMMARY" in caplog.text and "TTS chunk" in caplog.text
    assert "Sarah" not in caplog.text and "$45" not in caplog.text


def make_chain_for(tmp_path):
    return chain_of(tmp_path, tc.Provider("deepgram", FakeTTS("deepgram", ["ok"]), hold=True))


# ---- refill: filler is not a failed attempt ------------------------------------------------------------------------------------------
def test_thank_you_yeah_and_the_request_said_again_do_not_use_up_the_refill_tries(state):
    c = Call(state)
    verify_as(c)
    refill_flow.begin(state)
    for said in ("Thank you.", "Yeah.", "Well, I would like to get a refill.", "yeah I would like to get a prescription rifle as well",
                 "Okay"):
        d = c.hear(said)
        assert d.say == refill_flow.Q_MED and state.refill_tries == 0 and refill_flow.active(state), said
    d = c.hear("M-E-T-F-O-R-M-I-N")
    assert "metformin" in d.say.lower() and state.pending is not None and state.pending.kind == "refill"


def test_a_real_miss_still_counts():
    assert not refill_flow.says_no_medication("I'd like to leave food")
    assert not refill_flow.says_no_medication("Lisey Nuperell")
    assert not refill_flow.says_no_medication("yeah my medicine is metaphor")
    assert refill_flow.says_no_medication("Thank you.") and not refill_flow.says_no_medication("")
