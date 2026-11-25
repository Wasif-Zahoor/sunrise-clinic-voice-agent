"""TTS provider chain, fallback behaviour and the character counter. Offline: fake providers and a fake Cloudflare server,
so no quota or credit is spent. (Real providers are exercised by scripts/tts_samples.py, not by the test suite.)"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from livekit.agents import APIStatusError, tts

from clinic import config
from clinic import tts_chain as tc
from clinic.tts_budget import CharBudget

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = config.Settings("whisper-large-v3-turbo", "openai/gpt-oss-120b", "canopylabs/orpheus-v1-english", "hannah", 9)


# ---- fakes -------------------------------------------------------------------------------------------------------------------
class FakeTTS(tts.TTS):
    """A scripted provider. script items: ok | 402 | 429 | 500 | timeout | empty | partial (then repeats the last item)."""

    def __init__(self, name: str, script: list[str], sample_rate: int = 24000) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self.name, self.script, self.calls = name, list(script), 0
        self.last_text = None

    @property
    def model(self) -> str:
        return f"fake-{self.name}"

    def synthesize(self, text, *, conn_options=None):
        return _FakeStream(tts=self, input_text=text, conn_options=conn_options or tc.DEFAULT_API_CONNECT_OPTIONS)


class _FakeStream(tts.ChunkedStream):
    async def _run(self, out):
        owner = self._tts
        owner.calls += 1
        owner.last_text = self._input_text
        step = owner.script.pop(0) if len(owner.script) > 1 else owner.script[0]
        if step.isdigit():
            raise APIStatusError(message=f"HTTP {step}", status_code=int(step), request_id=None, body=None)
        if step == "timeout":
            await asyncio.sleep(30)
        out.initialize(request_id="r", sample_rate=owner.sample_rate, num_channels=1, mime_type="audio/pcm")
        if step == "empty":
            out.flush()
            return
        out.push(b"\x10\x00" * (owner.sample_rate // 5))                 # 0.2 s of non-silent audio
        if step == "partial":
            out.flush()
            raise APIStatusError(message="HTTP 500", status_code=500, request_id=None, body=None)
        out.flush()


def make_chain(tmp_path, *providers: FakeTTS, **kw) -> tc.TtsChain:
    budget = CharBudget("test", path=tmp_path / "usage.json", warn_total=kw.pop("warn_total", 100000))
    return tc.TtsChain([tc.Provider(p.name, p) for p in providers], budget, **kw)


async def speak(chain: tc.TtsChain, text: str = "Thank you for calling.") -> list:
    frames = []
    async with chain.synthesize(text) as stream:
        async for ev in stream:
            frames.append(ev.frame)
    return frames


def audible(frames) -> bool:
    return any(any(bytes(f.data)) for f in frames)


# ---- provider order from the environment -----------------------------------------------------------------------------------
@pytest.fixture
def piper_files(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not a real model: only the json is read at construction")
    Path(str(model) + ".json").write_text(json.dumps({"audio": {"sample_rate": 22050}}), encoding="utf-8")
    return model


def names(env, settings=SETTINGS):
    return [p.name for p in tc.build_providers(settings, env)]


def test_default_order_is_deepgram_cloudflare_then_local(piper_files):
    env = {"DEEPGRAM_API_KEY": "dg-test-not-real", "PIPER_MODEL": str(piper_files)}
    assert tc.provider_order({}) == ["deepgram", "cloudflare", "local"]
    assert names(env) == ["deepgram", "local"]                 # cloudflare has no keys here, so it is skipped


def test_order_comes_from_the_environment(piper_files):
    env = {"TTS_PROVIDER_ORDER": " Local , deepgram ", "DEEPGRAM_API_KEY": "dg-test-not-real", "PIPER_MODEL": str(piper_files)}
    assert names(env) == ["local", "deepgram"]


def test_a_provider_without_its_key_or_file_is_skipped_and_the_log_names_only_variables(piper_files, caplog):
    caplog.set_level(logging.INFO)
    env = {"TTS_PROVIDER_ORDER": "deepgram,cloudflare,groq,local,nonsense", "PIPER_MODEL": str(piper_files.with_name("missing.onnx"))}
    assert names(env) == []
    assert "DEEPGRAM_API_KEY is not set" in caplog.text and "CLOUDFLARE_ACCOUNT_ID" in caplog.text
    assert "GROQ_API_KEY is not set" in caplog.text and "voice file not found" in caplog.text and "unknown provider" in caplog.text


def test_cloudflare_needs_both_token_and_account(piper_files):
    only_key = {"TTS_PROVIDER_ORDER": "cloudflare", "CLOUDFLARE_API_KEY": "cf-test-not-real"}
    assert names(only_key) == []
    assert names({**only_key, "CLOUDFLARE_ACCOUNT_ID": "acct-not-real"}) == ["cloudflare"]
    assert names({"TTS_PROVIDER_ORDER": "cloudflare", "cloudflare_api_token": "t", "cloudflare_account_id": "a"}) == ["cloudflare"]


def test_deepgram_opts_out_of_model_improvement_and_uses_the_configured_model():
    p = tc.build_providers(SETTINGS, {"TTS_PROVIDER_ORDER": "deepgram", "DEEPGRAM_API_KEY": "dg-test-not-real",
                                      "DEEPGRAM_TTS_MODEL": "aura-2-andromeda-en"})[0].tts
    assert p.model == "aura-2-andromeda-en" and p._opts.mip_opt_out is True and p.sample_rate == 24000


def test_default_deepgram_model_is_a_valid_aura_2_voice():
    p = tc.build_providers(SETTINGS, {"TTS_PROVIDER_ORDER": "deepgram", "DEEPGRAM_API_KEY": "dg-test-not-real"})[0].tts
    assert p.model == tc.DEFAULT_DEEPGRAM_MODEL and p.model.startswith("aura-2-") and p.model.endswith("-en")


def test_session_tts_chain_reports_the_first_provider_model():
    chain = tc.build_chain(SETTINGS, env={"TTS_PROVIDER_ORDER": "deepgram", "DEEPGRAM_API_KEY": "dg-test-not-real"})
    assert chain.model == "aura-2-thalia-en"
    assert tc.build_chain(SETTINGS, env={"TTS_PROVIDER_ORDER": "deepgram"}).model == "none"      # no key: no provider, no crash


# ---- fallback behaviour ----------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_primary_serves_when_it_works_and_the_fallback_is_untouched(tmp_path):
    dg, local = FakeTTS("deepgram", ["ok"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    assert audible(await speak(chain)) and (dg.calls, local.calls) == (1, 0) and chain.active == "deepgram"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["402", "429", "401", "403"])
async def test_a_permanent_failure_switches_to_local_for_the_rest_of_the_call(tmp_path, failure):
    dg, local = FakeTTS("deepgram", [failure]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    assert audible(await speak(chain, "first")) and chain.active == "local"
    assert audible(await speak(chain, "second")) and audible(await speak(chain, "third"))
    assert dg.calls == 1, "a failed provider must not be asked again during the call (no wasted credit)"
    assert local.calls == 3 and chain.stats == {"degraded": 0, "fallbacks": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["500", "503", "empty", "timeout"])
async def test_a_transient_failure_costs_one_chunk_and_the_primary_is_tried_again(tmp_path, failure):
    """One network hiccup must not change the caller's voice for the rest of the call."""
    dg, local = FakeTTS("deepgram", [failure, "ok"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local, first_audio_timeout=0.2)
    assert audible(await speak(chain, "first")) and chain.active == "local" and "deepgram" not in chain.dead
    assert chain.strikes["deepgram"] == 1
    assert audible(await speak(chain, "second")) and chain.active == "deepgram"            # the primary is back
    assert chain.strikes["deepgram"] == 0 and (dg.calls, local.calls) == (2, 1) and chain.stats["degraded"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["500", "timeout", "empty"])
async def test_two_transient_failures_in_a_row_make_it_permanent(tmp_path, failure):
    dg, local = FakeTTS("deepgram", [failure]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local, first_audio_timeout=0.2)
    for text in ("first", "second", "third", "fourth"):
        assert audible(await speak(chain, text))
    assert "deepgram" in chain.dead and dg.calls == 2 and local.calls == 4                 # not asked again after the second strike


@pytest.mark.asyncio
async def test_a_success_between_failures_resets_the_strikes(tmp_path):
    dg, local = FakeTTS("deepgram", ["500", "ok", "500", "ok"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    for text in ("a", "b", "c", "d"):
        await speak(chain, text)
    assert "deepgram" not in chain.dead and dg.calls == 4 and local.calls == 2


@pytest.mark.asyncio
async def test_a_provider_that_breaks_after_starting_still_gives_the_caller_audio(tmp_path):
    """The framework delivers no audio when the stream errors, so nothing was heard and the fallback speaks the chunk."""
    dg, local = FakeTTS("deepgram", ["partial"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    assert audible(await speak(chain)) and chain.strikes["deepgram"] == 1 and chain.active == "local"
    await speak(chain)
    assert dg.calls == 2 and local.calls == 2 and "deepgram" in chain.dead


@pytest.mark.asyncio
async def test_three_providers_fall_through_in_order(tmp_path):
    a, b, c = FakeTTS("deepgram", ["402"]), FakeTTS("cloudflare", ["429"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, a, b, c)
    assert audible(await speak(chain)) and chain.active == "local" and chain.stats["fallbacks"] == 2


@pytest.mark.asyncio
async def test_when_every_provider_fails_the_call_survives_with_silence(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    chain = make_chain(tmp_path, FakeTTS("deepgram", ["402"]), FakeTTS("local", ["500"]))
    frames = await speak(chain, "Sarah Johnson, your balance is forty five dollars.")      # must not raise
    assert frames and not audible(frames) and chain.stats["degraded"] == 1
    assert "Sarah" not in caplog.text and "balance" not in caplog.text


@pytest.mark.asyncio
async def test_a_provider_that_recovers_is_retried_only_when_all_have_failed(tmp_path):
    dg, local = FakeTTS("deepgram", ["402", "ok"]), FakeTTS("local", ["402", "ok"])
    chain = make_chain(tmp_path, dg, local)
    assert not audible(await speak(chain))                     # both failed: silence
    assert audible(await speak(chain)) and chain.active == "deepgram"            # all were dead, so they are tried again


@pytest.mark.asyncio
async def test_a_provider_at_another_sample_rate_is_resampled(tmp_path):
    local = FakeTTS("local", ["ok"], sample_rate=22050)
    frames = await speak(make_chain(tmp_path, local))
    assert frames and {f.sample_rate for f in frames} == {24000}
    seconds = sum(f.samples_per_channel for f in frames) / 24000
    assert 0.15 < seconds < 0.25                                # 0.2 s in, 0.2 s out


@pytest.mark.asyncio
async def test_no_providers_at_all_is_silent_not_a_crash(tmp_path):
    chain = tc.TtsChain([], CharBudget("t", path=tmp_path / "u.json"))
    assert not audible(await speak(chain)) and chain.stats["degraded"] == 1


# ---- a long call ------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_24_turn_call_with_deepgram_failing_on_turn_10_never_skips_a_turn(tmp_path):
    dg, local = FakeTTS("deepgram", ["ok"] * 9 + ["402"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    served = []
    for turn in range(1, 25):
        frames = await speak(chain, f"Agent reply number {turn}.")
        assert audible(frames), f"turn {turn} was silent"
        served.append(chain.active)
    assert served[:9] == ["deepgram"] * 9 and served[9:] == ["local"] * 15
    assert chain.stats == {"degraded": 0, "fallbacks": 1} and dg.calls == 10 and local.calls == 15
    assert local.last_text == "Agent reply number 24."          # the failed turn itself (10) was re-spoken by local, not dropped


@pytest.mark.asyncio
async def test_a_24_turn_call_through_the_real_session_adapter_with_grouping(tmp_path):
    """Same, but through StreamAdapter + GroupedSentenceTokenizer as agent.build_session wires it."""
    from clinic.tts_safety import GroupedSentenceTokenizer
    dg, local = FakeTTS("deepgram", ["ok"] * 9 + ["402"]), FakeTTS("local", ["ok"])
    adapter = tts.StreamAdapter(tts=make_chain(tmp_path, dg, local), sentence_tokenizer=GroupedSentenceTokenizer())
    for turn in range(1, 25):
        frames = []
        async with adapter.synthesize(f"Reply {turn}. Is there anything else I can help you with today?") as stream:
            async for ev in stream:
                frames.append(ev.frame)
        assert audible(frames), f"turn {turn} was silent"
    assert dg.calls == 10 and local.calls == 15                   # one grouped request per turn, none skipped


# ---- character counter -------------------------------------------------------------------------------------------------------
def test_counts_per_call_per_day_and_total_and_persists(tmp_path):
    day = {"d": date(2026, 11, 25)}
    b = CharBudget("c1", path=tmp_path / "u.json", warn_total=10**9, today=lambda: day["d"])
    assert b.note("deepgram", 100) == {"call": 100, "today": 100, "total": 100}
    assert b.note("deepgram", 50) == {"call": 150, "today": 150, "total": 150}
    day["d"] = date(2026, 11, 26)
    assert b.note("deepgram", 10) == {"call": 160, "today": 10, "total": 160}
    again = CharBudget("c2", path=tmp_path / "u.json", warn_total=10**9, today=lambda: day["d"])      # a new call / process
    assert again.note("deepgram", 5) == {"call": 5, "today": 15, "total": 165}
    assert again.summary()["deepgram"] == {"total": 165, "today": 15}


def test_warns_once_when_the_threshold_is_passed_and_never_for_the_free_local_voice(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    b = CharBudget("c", path=tmp_path / "u.json", warn_total=250)
    b.note("deepgram", 200)
    assert "warning threshold" not in caplog.text
    b.note("deepgram", 100)
    b.note("deepgram", 100)
    assert caplog.text.count("warning threshold 250") == 1 and "TTS_CHAR_WARN" in caplog.text
    caplog.clear()
    b.note("local", 10**7)
    assert "warning threshold" not in caplog.text


def test_threshold_is_configurable_from_the_environment(monkeypatch):
    from clinic.tts_budget import warn_threshold
    monkeypatch.delenv("TTS_CHAR_WARN", raising=False)
    assert warn_threshold() == 100000
    monkeypatch.setenv("TTS_CHAR_WARN", "5000")
    assert warn_threshold() == 5000
    monkeypatch.setenv("TTS_CHAR_WARN", "lots")
    assert warn_threshold() == 100000


def test_usage_log_has_counts_only_and_the_file_has_no_text(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    b = CharBudget("c", path=tmp_path / "u.json")
    b.note("deepgram", len("Sarah Johnson, born March 14 1985, owes $45.00"))
    assert "Sarah" not in caplog.text and "1985" not in caplog.text
    assert "Sarah" not in (tmp_path / "u.json").read_text(encoding="utf-8")


def test_an_unwritable_usage_file_does_not_break_the_call(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    b = CharBudget("c", path=blocker / "u.json")
    assert b.note("deepgram", 10)["call"] == 10


@pytest.mark.asyncio
async def test_the_chain_counts_only_what_succeeded(tmp_path):
    dg, local = FakeTTS("deepgram", ["402"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    await speak(chain, "x" * 40)
    await speak(chain, "y" * 10)
    assert "deepgram" not in chain.budget.call_chars and chain.budget.call_chars["local"] == 50


# ---- real provider classes against local fakes --------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cloudflare_request_shape_and_pcm_response(tmp_path):
    seen = {}

    async def handler(request: web.Request) -> web.Response:
        seen["path"], seen["auth"], seen["body"] = request.path, request.headers.get("Authorization"), await request.json()
        return web.Response(body=b"\x20\x00" * 4800, content_type="audio/mpeg")          # Cloudflare labels raw PCM like this

    app = web.Application()
    app.router.add_post("/accounts/ACCT/ai/run/{model:.*}", handler)
    async with TestServer(app) as server:
        async with aiohttp.ClientSession() as session:
            cf = tc.CloudflareTTS(account_id="ACCT", token="cf-test-not-real", base_url=f"http://127.0.0.1:{server.port}",
                                  http_session=session)
            chain = tc.TtsChain([tc.Provider("cloudflare", cf)], CharBudget("t", path=tmp_path / "u.json"))
            frames = await speak(chain, "Hello there.")
    assert seen["path"].endswith("/ai/run/@cf/deepgram/aura-2-en") and seen["auth"] == "Bearer cf-test-not-real"
    assert seen["body"] == {"text": "Hello there.", "encoding": "linear16", "container": "none", "sample_rate": 24000}
    assert audible(frames) and abs(sum(f.samples_per_channel for f in frames) / 24000 - 0.2) < 0.02


@pytest.mark.asyncio
async def test_cloudflare_error_status_falls_back(tmp_path):
    async def handler(request: web.Request) -> web.Response:
        return web.json_response({"errors": [{"message": "no credit"}]}, status=402)

    app = web.Application()
    app.router.add_post("/accounts/ACCT/ai/run/{model:.*}", handler)
    async with TestServer(app) as server:
        async with aiohttp.ClientSession() as session:
            cf = tc.CloudflareTTS(account_id="ACCT", token="t", base_url=f"http://127.0.0.1:{server.port}", http_session=session)
            local = FakeTTS("local", ["ok"])
            chain = tc.TtsChain([tc.Provider("cloudflare", cf), tc.Provider("local", local)],
                                CharBudget("t", path=tmp_path / "u.json"))
            assert audible(await speak(chain)) and chain.active == "local" and "cloudflare" in chain.dead


@pytest.mark.asyncio
async def test_piper_wrapper_runs_off_the_event_loop_and_returns_audio(piper_files, tmp_path, monkeypatch):
    loop_thread = {}

    def fake_sentences(self, text):
        import threading
        loop_thread["worker"] = threading.current_thread().name
        yield b"\x30\x00" * 1100                                    # two sentences, about 0.05 s each at 22050 Hz
        yield b"\x30\x00" * 1105

    monkeypatch.setattr(tc.PiperTTS, "sentences", fake_sentences)
    piper = tc.PiperTTS(piper_files)
    assert piper.sample_rate == 22050 and piper.model == "voice"
    chain = tc.TtsChain([tc.Provider("local", piper)], CharBudget("t", path=tmp_path / "u.json"))
    frames = await speak(chain)
    assert audible(frames) and {f.sample_rate for f in frames} == {24000}
    assert loop_thread["worker"] != "MainThread"


@pytest.mark.asyncio
async def test_piper_that_produces_nothing_counts_as_a_failure(piper_files, tmp_path, monkeypatch):
    monkeypatch.setattr(tc.PiperTTS, "sentences", lambda self, text: iter(()))
    chain = tc.TtsChain([tc.Provider("local", tc.PiperTTS(piper_files))], CharBudget("t", path=tmp_path / "u.json"))
    assert not audible(await speak(chain)) and chain.stats["degraded"] == 1


# ---- secrets and wiring ------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_key_value_or_spoken_text_in_logs_when_providers_fail(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    secret = "dg-SECRETVALUE-123456"
    env = {"TTS_PROVIDER_ORDER": "deepgram", "DEEPGRAM_API_KEY": secret}
    tc.build_chain(SETTINGS, env=env)
    chain = make_chain(tmp_path, FakeTTS("deepgram", ["402"]), FakeTTS("local", ["ok"]))
    await speak(chain, "Michael Brown, date of birth November 8 1972")
    assert "SECRETVALUE" not in caplog.text and "Michael" not in caplog.text and "1972" not in caplog.text
    assert "deepgram failed (APIStatusError HTTP 402)" in caplog.text


@pytest.mark.asyncio
async def test_build_session_wires_the_chain_with_grouping(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "placeholder-not-real")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-not-real")
    monkeypatch.setenv("TTS_PROVIDER_ORDER", "deepgram,local")
    import agent
    from clinic.tts_safety import GroupedSentenceTokenizer
    session = agent.build_session()
    assert isinstance(session.tts._wrapped_tts, tc.TtsChain) and session.tts_chain is session.tts._wrapped_tts
    assert isinstance(session.tts._sentence_tokenizer, GroupedSentenceTokenizer)
    assert session.tts_chain.providers[0].name == "deepgram" and session.tts.model == "aura-2-thalia-en"


def test_models_and_usage_file_are_gitignored_and_documented():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf8").splitlines()
    assert "models/" in ignore and "data/tts_usage.json" in ignore
    example = (ROOT / ".env.example").read_text(encoding="utf8")
    for name in ("TTS_PROVIDER_ORDER", "DEEPGRAM_API_KEY", "DEEPGRAM_TTS_MODEL", "PIPER_MODEL", "CLOUDFLARE_ACCOUNT_ID",
                 "CLOUDFLARE_TTS_MODEL", "TTS_CHAR_WARN"):
        assert name in example
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf8")
    assert "livekit-plugins-deepgram==1.8.13" in requirements and "piper-tts" in requirements


def test_deepgram_plugin_loads_when_the_agent_is_imported_and_works_in_a_worker_thread():
    """Regression: plugins can only be registered on the main thread, and on Windows each call runs in a worker thread, so a lazy
    import inside build_providers failed at call time (the chain silently became local-only). Needs a fresh interpreter."""
    import subprocess
    import sys
    code = (
        "import sys, threading; sys.path.insert(0, r'%s')\n"
        "import agent\n"
        "from clinic import config, tts_chain as tc\n"
        "out = {}\n"
        "def work():\n"
        "    s = config.Settings('w', 'l', 't', 'v', 9)\n"
        "    out['n'] = [p.name for p in tc.build_providers(s, {'TTS_PROVIDER_ORDER': 'deepgram', 'DEEPGRAM_API_KEY': 'dg-test-not-real'})]\n"
        "t = threading.Thread(target=work); t.start(); t.join()\n"
        "print(out)\n" % ROOT)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert "{'n': ['deepgram']}" in result.stdout, result.stdout + result.stderr[-500:]


def test_the_error_description_names_the_cause_type_only():
    from livekit.agents import APIConnectionError
    try:
        try:
            raise ConnectionResetError("secret caller words in a message")
        except ConnectionResetError as inner:
            raise APIConnectionError() from inner
    except APIConnectionError as e:
        text = tc._describe(e)
    assert text == "APIConnectionError caused by ConnectionResetError" and "secret" not in text
    assert tc._describe(APIStatusError(message="m", status_code=402, request_id=None, body=None)) == "APIStatusError HTTP 402"


@pytest.mark.asyncio
async def test_the_log_says_when_the_primary_will_be_retried(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    chain = make_chain(tmp_path, FakeTTS("deepgram", ["500", "ok"]), FakeTTS("local", ["ok"]))
    await speak(chain)
    assert "will be tried again (1 of 2)" in caplog.text
    caplog.clear()
    chain2 = make_chain(tmp_path, FakeTTS("deepgram", ["402"]), FakeTTS("local", ["ok"]))
    await speak(chain2)
    assert "deepgram is off for the rest of this call; speaking this chunk with local" in caplog.text


def test_piper_applies_the_pace_and_volume_that_match_deepgram(piper_files):
    seen = {}

    class FakeVoice:
        def synthesize(self, text, config=None):
            seen["config"] = config
            return [type("C", (), {"audio_int16_bytes": b"\x01\x00" * 100})()]

    piper = tc.PiperTTS(piper_files)
    piper._voice = FakeVoice()
    assert piper.render("Hello.") == b"\x01\x00" * 100
    assert seen["config"].length_scale == tc.DEFAULT_PIPER_LENGTH_SCALE == 1.04
    assert seen["config"].volume == tc.DEFAULT_PIPER_VOLUME == 0.4


def test_piper_pace_and_volume_come_from_the_environment(piper_files):
    env = {"TTS_PROVIDER_ORDER": "local", "PIPER_MODEL": str(piper_files), "PIPER_LENGTH_SCALE": "1.2", "PIPER_VOLUME": "0.6"}
    piper = tc.build_providers(SETTINGS, env)[0].tts
    assert (piper._length_scale, piper._volume) == (1.2, 0.6)


def test_the_default_local_voice_is_the_female_one_measured_closest_to_thalia():
    assert tc.DEFAULT_PIPER_MODEL.endswith("en_US-hfc_female-medium.onnx")
