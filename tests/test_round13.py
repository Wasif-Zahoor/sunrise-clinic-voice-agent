"""Round 13 (a recorded call, 2026-11-25): the Cloudflare token started to be refused from a new network address (HTTP 401) and its failure
would have been remembered for an hour even after the token was replaced; the first call waited 5.7 s for a worker, 7.6 s of one-time
set-up landed on the first greeting, and the greeting could fall through to a cold local voice. Offline."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from clinic import tts_chain as tc
from clinic.tts_budget import CharBudget
from clinic.tts_health import ProviderHealth
from test_round12 import Clock, chain_with
from test_tts_chain import SETTINGS, FakeTTS, audible, piper_files, speak  # noqa: F401  (piper_files is a fixture)


# ---- a replaced key is not held back by the old key's failure ---------------------------------------------------------------------------
@pytest.fixture
def health(tmp_path):
    clock = Clock()
    h = ProviderHealth(tmp_path / "health.json", now=clock)
    h.clock = clock
    return h


def test_a_failure_recorded_for_one_key_is_ignored_for_another(health):
    health.mark_down("cloudflare", 3600, "APIStatusError HTTP 401", "aaaa")
    assert "cloudflare" in health.cooling({"cloudflare": "aaaa"})
    assert health.cooling({"cloudflare": "bbbb"}) == {}             # the token was replaced
    assert "cloudflare" in health.cooling()                         # no fingerprints given: the record is still there


def test_an_old_record_without_a_fingerprint_does_not_block_a_provider_that_has_one(health):
    health.mark_down("cloudflare", 3600, "x")                       # what the previous version wrote
    assert health.cooling({"cloudflare": "aaaa"}) == {}


def test_other_providers_are_not_affected_by_one_providers_fingerprint(health):
    health.mark_down("deepgram", 600, "x", "d1")
    assert "deepgram" in health.cooling({"cloudflare": "c1"})       # no fingerprint known for deepgram here: unchanged behaviour


@pytest.mark.asyncio
async def test_after_a_failure_the_replaced_key_is_used_by_the_next_call(tmp_path, health, caplog):
    caplog.set_level(logging.INFO)
    old = tc.Provider("cloudflare", FakeTTS("cloudflare", ["500"]), fingerprint="old-key")      # two slow failures in a row
    local = tc.Provider("local", FakeTTS("local", ["ok"]))
    first = tc.TtsChain([old, local], CharBudget("t", path=tmp_path / "u.json"), health=health)
    await speak(first, "one")
    assert audible(await speak(first, "two")) and "cloudflare" in first.dead
    assert health.cooling({"cloudflare": "old-key"})["cloudflare"]["why"].startswith("APIStatusError HTTP 500")

    same = tc.TtsChain([tc.Provider("cloudflare", FakeTTS("cloudflare", ["ok"]), fingerprint="old-key"),
                        tc.Provider("local", FakeTTS("local", ["ok"]))], CharBudget("t", path=tmp_path / "u.json"), health=health)
    assert "cloudflare" in same.dead                                 # same key: still skipped

    fixed_cf = FakeTTS("cloudflare", ["ok"])
    fixed = tc.TtsChain([tc.Provider("cloudflare", fixed_cf, fingerprint="new-key"), tc.Provider("local", FakeTTS("local", ["ok"]))],
                        CharBudget("t", path=tmp_path / "u.json"), health=health)
    assert fixed.dead == set()
    assert audible(await speak(fixed)) and fixed.active == "cloudflare" and fixed_cf.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["401", "402", "403", "429"])
async def test_a_refused_key_or_used_up_limit_is_not_remembered_for_later_calls(tmp_path, health, status):
    """It fails at once, so remembering it saves nothing, and would block a token that was fixed without changing its value (for example
    after removing an IP-address restriction in the Cloudflare dashboard)."""
    cf = FakeTTS("cloudflare", [status])
    first = chain_with(health, cf, FakeTTS("local", ["ok"]), tmp_path=tmp_path)
    assert audible(await speak(first)) and "cloudflare" in first.dead
    assert health.cooling() == {}
    cf2 = FakeTTS("cloudflare", ["ok"])
    second = chain_with(health, cf2, FakeTTS("local", ["ok"]), tmp_path=tmp_path)
    assert second.dead == set() and audible(await speak(second)) and second.active == "cloudflare"


def test_the_fingerprint_is_a_short_one_way_hash_that_does_not_contain_the_key():
    key = "cfat_" + "A1b2C3d4" * 6
    fp = tc._fingerprint(key, "account", "model")
    assert len(fp) == 12 and key not in fp and "account" not in fp
    assert fp == tc._fingerprint(key, "account", "model") and fp != tc._fingerprint(key + "x", "account", "model")
    assert fp != tc._fingerprint(key, "other-account", "model")


def test_providers_get_their_fingerprints_from_the_keys_in_use(piper_files):
    env = {"TTS_PROVIDER_ORDER": "deepgram,cloudflare,local", "DEEPGRAM_API_KEY": "dg-one", "CLOUDFLARE_API_TOKEN": "cf-one",
           "CLOUDFLARE_ACCOUNT_ID": "acct", "PIPER_MODEL": str(piper_files)}
    a = {p.name: p.fingerprint for p in tc.build_providers(SETTINGS, env)}
    b = {p.name: p.fingerprint for p in tc.build_providers(SETTINGS, {**env, "CLOUDFLARE_API_TOKEN": "cf-two"})}
    c = {p.name: p.fingerprint for p in tc.build_providers(SETTINGS, {**env, "CLOUDFLARE_ACCOUNT_ID": "other"})}
    assert a["deepgram"] == b["deepgram"] and a["cloudflare"] != b["cloudflare"] and a["cloudflare"] != c["cloudflare"]
    assert a["local"] == ""
    assert "cf-one" not in a["cloudflare"] and "dg-one" not in a["deepgram"]


# ---- the local voice is loaded once and shared -----------------------------------------------------------------------------------------------
class FakePiper:
    loads = 0

    class PiperVoice:
        @staticmethod
        def load(path):
            FakePiper.loads += 1
            return FakePiper.Voice()

    class Voice:
        def synthesize(self, text, config):
            yield types.SimpleNamespace(audio_int16_bytes=b"\x10\x00" * 2400)

    class SynthesisConfig:
        def __init__(self, **kw):
            self.kw = kw


@pytest.fixture
def fake_piper(monkeypatch):
    FakePiper.loads = 0
    tc._VOICES.clear()
    module = types.ModuleType("piper")
    module.PiperVoice, module.SynthesisConfig = FakePiper.PiperVoice, FakePiper.SynthesisConfig
    monkeypatch.setitem(sys.modules, "piper", module)
    yield
    tc._VOICES.clear()


def test_two_calls_share_one_loaded_voice(piper_files, fake_piper):
    a, b = tc.PiperTTS(piper_files), tc.PiperTTS(piper_files)
    assert a._load() is b._load() and FakePiper.loads == 1


def test_warming_the_voice_at_start_means_a_later_call_loads_nothing(piper_files, fake_piper):
    assert tc.warm_local_voice({"PIPER_MODEL": str(piper_files)}) is True and FakePiper.loads == 1
    later = tc.PiperTTS(piper_files)
    assert later.render("Thank you for calling.") and FakePiper.loads == 1


def test_warming_without_a_voice_file_does_nothing(tmp_path):
    assert tc.warm_local_voice({"PIPER_MODEL": str(tmp_path / "missing.onnx")}) is False


def test_a_broken_voice_file_does_not_break_the_warm_up(piper_files, monkeypatch):
    broken = types.ModuleType("piper")
    broken.PiperVoice = types.SimpleNamespace(load=lambda p: (_ for _ in ()).throw(RuntimeError("bad model")))
    broken.SynthesisConfig = FakePiper.SynthesisConfig
    monkeypatch.setitem(sys.modules, "piper", broken)
    tc._VOICES.clear()
    assert tc.warm_local_voice({"PIPER_MODEL": str(piper_files)}) is False


# ---- the worker warms up before the first call --------------------------------------------------------------------------------------------
def test_the_worker_keeps_a_spare_process_and_warms_it_at_start():
    import agent
    assert agent.server._num_idle_processes == 1 and agent.server.setup_fnc is agent.prewarm


def test_the_warm_up_runs_without_a_call_and_never_raises(monkeypatch):
    import agent
    called = []
    monkeypatch.setattr(tc, "warm_local_voice", lambda env=None: called.append(True) or True)
    agent.prewarm(None)
    assert called == [True]
    monkeypatch.setattr(tc, "warm_local_voice", lambda env=None: (_ for _ in ()).throw(RuntimeError("boom")))
    agent.prewarm(None)                                             # a failing warm-up must never stop the worker
