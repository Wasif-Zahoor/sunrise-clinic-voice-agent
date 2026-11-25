"""Round 12 (fifth live call, an after-hours call, 2026-11-25): "what is my last status" (speech-to-text for "lab status") after the
message was saved got a vague "I can't access that information" and a second offer of a callback; and the greeting had 5 s and then 9 s of
silence because Deepgram timed out and Cloudflare's daily limit was used up (HTTP 429), and every new call found that out again.
Offline."""

from __future__ import annotations

import logging

import pytest

from clinic import callback_flow, gate, intent, tts_chain as tc
from clinic.clock import HOURS_TEXT
from clinic.tts_budget import CharBudget
from clinic.tts_health import COOLDOWN_TRANSIENT, ProviderHealth
from clinic.turn import assess_turn
from conftest import CLOSED_EVENING, CLOSED_SATURDAY
from test_call_flow_fixes import Call
from test_tts_chain import SETTINGS, FakeTTS, audible, speak


# ---- after hours: one fixed answer, in code ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("said", [
    "what is my lab status", "Can you tell me my last status?", "can you tell me my lap results", "what's my balance", "how much do I owe",
    "I need a prescription refill", "I'd like to schedule an appointment", "when is my next appointment"])
def test_a_patient_request_after_hours_gets_the_closed_answer_and_an_offer(make_state, said):
    s = make_state(CLOSED_EVENING)
    d = assess_turn(s, said)
    assert d.say.startswith("The clinic is closed, so I can't share or change patient information right now.")
    assert HOURS_TEXT in d.say and d.say.endswith("I can take a message for our team. Would you like that?")
    assert s.choice == "callback_yes" and not s.verified and s.vstep == "idle" and s.bstep == "idle" and not s.awaiting_ssn


def test_the_weekend_is_the_same(make_state):
    s = make_state(CLOSED_SATURDAY)
    assert "The clinic is closed" in assess_turn(s, "what is my lab status").say


def test_yes_starts_the_message_and_no_leaves_it(make_state):
    s = make_state(CLOSED_EVENING)
    assess_turn(s, "what is my balance")
    assert assess_turn(s, "yes please").say == callback_flow.QUESTIONS["name"] and s.cstep == "name"
    s2 = make_state(CLOSED_EVENING)
    assess_turn(s2, "what is my balance")
    assert assess_turn(s2, "no thanks").say == "No problem. Is there anything else I can help you with?" and s2.choice is None


def test_the_replay_of_the_real_call_a_saved_message_is_not_offered_again(make_state, conn):
    s = make_state(CLOSED_EVENING)
    c = Call(s)
    callback_flow.start(s)
    for said in ("Sarah", "555 111 2222", "an appointment question"):
        c.hear(said)
    assert c.hear("yes").say.endswith("Have a good night.")
    d = c.hear("Can you tell me my last status?")
    assert d.say == (f"The clinic is closed, so I can't share or change patient information right now. Our hours are {HOURS_TEXT}. "
                     "Your message is recorded, and our team will follow up on the next business day. "
                     "Is there anything else I can help you with?")
    assert s.choice is None
    assert conn.execute("SELECT COUNT(*) FROM requests WHERE request_type='callback'").fetchone()[0] == 1       # nothing saved twice


def test_nothing_is_looked_up_after_hours_even_for_a_verified_caller(make_state, conn):
    s = make_state(CLOSED_EVENING)
    s.verified_patient_id = "P001"
    d = assess_turn(s, "what is my lab status")
    assert "closed" in d.say and "ready" not in d.say.lower() and "$" not in d.say
    assert conn.execute("SELECT COUNT(*) FROM audit_logs WHERE action IN ('get_lab_status','get_balance')").fetchone()[0] == 0


def test_an_emergency_after_hours_still_comes_first(make_state):
    s = make_state(CLOSED_EVENING)
    d = assess_turn(s, "I'm having chest pain")
    assert "911" in d.say and s.emergency and s.choice is None


def test_a_question_about_someone_else_after_hours_is_left_to_the_llm_with_its_note(make_state):
    s = make_state(CLOSED_EVENING)
    d = assess_turn(s, "what does my husband owe")
    assert d.say is None and d.note and "someone else" in d.note


def test_an_ordinary_question_after_hours_is_not_taken_over(make_state):
    assert intent.route(make_state(CLOSED_EVENING), "what are your office hours") is None


@pytest.mark.parametrize("text", ["what is my last status", "my lap results please", "what is my lab status"])
def test_the_lab_wording_speech_to_text_produces(text):
    assert intent.detect(text) == "lab"


def test_last_status_is_not_taken_for_something_else():
    assert intent.detect("what was the last thing you said") is None


# ---- the provider health memory ----------------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def health(tmp_path):
    clock = Clock()
    h = ProviderHealth(tmp_path / "health.json", now=clock)
    h.clock = clock
    return h


def chain_with(health, *providers, tmp_path):
    return tc.TtsChain([tc.Provider(p.name, p) for p in providers], CharBudget("t", path=tmp_path / "u.json"), health=health)


def test_a_failure_is_remembered_until_its_cooldown_ends(health):
    health.mark_down("deepgram", 600, "APITimeoutError")
    assert health.cooling()["deepgram"]["why"] == "APITimeoutError" and health.cooling()["deepgram"]["minutes_left"] == 10
    health.clock.t += 599
    assert "deepgram" in health.cooling()
    health.clock.t += 2
    assert health.cooling() == {}


def test_a_damaged_or_unwritable_file_is_ignored(tmp_path):
    p = tmp_path / "health.json"
    p.write_text("{not json", encoding="utf-8")
    h = ProviderHealth(p, now=Clock())
    assert h.cooling() == {}
    h.mark_down("x", 60, "why")                                   # overwrites the damaged file
    assert "x" in h.cooling()
    blocked = ProviderHealth(tmp_path / "health.json" / "nested.json", now=Clock())     # a path that cannot be created
    blocked.mark_down("x", 60, "why")
    assert blocked.cooling() == {}


def test_mark_ok_forgets_a_provider(health):
    health.mark_down("deepgram", 600, "x")
    health.mark_ok("deepgram")
    health.mark_ok("never-recorded")
    assert health.cooling() == {}


@pytest.mark.asyncio
async def test_the_next_call_does_not_wait_for_a_provider_the_last_one_gave_up_on(tmp_path, health, caplog):
    caplog.set_level(logging.INFO)
    dg1, local1 = FakeTTS("deepgram", ["500"]), FakeTTS("local", ["ok"])           # slow failures: a timeout or error twice in a row
    first = chain_with(health, dg1, local1, tmp_path=tmp_path)
    await speak(first, "one")
    assert audible(await speak(first, "two")) and "deepgram" in first.dead
    assert health.cooling()["deepgram"]["minutes_left"] == COOLDOWN_TRANSIENT // 60

    caplog.clear()
    dg2, local2 = FakeTTS("deepgram", ["ok"]), FakeTTS("local", ["ok"])
    second = chain_with(health, dg2, local2, tmp_path=tmp_path)
    assert "deepgram" in second.dead and "skipped for this call" in caplog.text
    assert audible(await speak(second)) and (dg2.calls, local2.calls) == (0, 1) and second.active == "local"


@pytest.mark.asyncio
async def test_two_transient_failures_are_remembered_for_the_shorter_time(tmp_path, health):
    dg, local = FakeTTS("deepgram", ["500"]), FakeTTS("local", ["ok"])
    chain = chain_with(health, dg, local, tmp_path=tmp_path)
    await speak(chain, "one")
    assert health.cooling() == {}                                  # one failure is only a strike
    await speak(chain, "two")
    assert health.cooling()["deepgram"]["minutes_left"] == COOLDOWN_TRANSIENT // 60


@pytest.mark.asyncio
async def test_after_the_cooldown_the_provider_is_tried_again(tmp_path, health):
    health.mark_down("deepgram", 600, "x")
    health.clock.t += 601
    dg, local = FakeTTS("deepgram", ["ok"]), FakeTTS("local", ["ok"])
    chain = chain_with(health, dg, local, tmp_path=tmp_path)
    assert chain.dead == set()
    await speak(chain)
    assert chain.active == "deepgram"


def test_the_chain_is_never_left_without_a_provider(tmp_path, health):
    health.mark_down("deepgram", 600, "x")
    health.mark_down("local", 600, "x")
    chain = chain_with(health, FakeTTS("deepgram", ["ok"]), FakeTTS("local", ["ok"]), tmp_path=tmp_path)
    assert chain.dead == set()


def test_a_provider_that_works_again_is_forgotten(tmp_path, health):
    chain = chain_with(health, FakeTTS("deepgram", ["ok"]), tmp_path=tmp_path)
    chain._remember_down("deepgram", "x")
    assert "deepgram" in health.cooling()
    chain._remember_ok("deepgram")
    assert health.cooling() == {}


def test_the_memory_is_off_for_tests_and_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-test-not-real")
    monkeypatch.setenv("TTS_HEALTH_FILE", str(tmp_path / "h.json"))
    monkeypatch.delenv("TTS_HEALTH", raising=False)
    assert tc.build_chain(SETTINGS, "t", {"DEEPGRAM_API_KEY": "dg-test-not-real"}, CharBudget("t", path=tmp_path / "u.json")).health is None
    on = tc.build_chain(SETTINGS, "t", None, CharBudget("t", path=tmp_path / "u.json"))
    assert on.health is not None and on.health.path == tmp_path / "h.json"
    monkeypatch.setenv("TTS_HEALTH", "off")
    assert tc.build_chain(SETTINGS, "t", None, CharBudget("t", path=tmp_path / "u.json")).health is None


# ---- the log says what actually happens ---------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_log_names_who_speaks_this_chunk_and_who_starts_the_next_ones(tmp_path, caplog):
    """The real call logged 'using deepgram for the rest of this call' although the chunk went to the local voice."""
    caplog.set_level(logging.INFO)
    dg, cf, local = FakeTTS("deepgram", ["500", "ok"]), FakeTTS("cloudflare", ["429"]), FakeTTS("local", ["ok"])
    chain = chain_with(None, dg, cf, local, tmp_path=tmp_path)
    await speak(chain)
    assert "cloudflare is off for the rest of this call; speaking this chunk with local, later chunks start with deepgram" in caplog.text
    assert chain.active == "local" and "cloudflare" in chain.dead and "deepgram" not in chain.dead
