"""Silence, "no appointment" choice, interruption/unclear answers, greeting delivery, connection options. Offline."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from clinic import booking_flow, callback_flow, gate, silence, tools, verify_flow
from clinic.turn import assess_turn
from conftest import CLOSED_EVENING, MICHAEL, SARAH


def verified(state, who=SARAH):
    verify_flow.start(state)
    for t in (who[0], "yes", who[1], " ".join("0123456789"[int(c)] for c in who[2])):
        pass
    name, dob, last4 = who
    for text in (name, "yes", {"1985-03-14": "March 14 1985", "1972-11-08": "November 8 1972"}[dob],
                 {"3344": "three three four four", "1020": "one zero two zero"}[last4]):
        assess_turn(state, gate.scrub_transcript(state, text))
    assert state.verified


# ---- silence ----------------------------------------------------------------------------------------------------------------
def test_silence_repeats_the_open_verification_question(state):
    verify_flow.start(state)
    first = silence.reprompt(state)
    assert first.startswith("Are you still there?") and "full name" in first


@pytest.mark.parametrize("setup,expect", [
    (lambda s: verify_flow.start(s), "full name"),
    (lambda s: booking_flow.start(s, "yes"), "full name"),
    (lambda s: callback_flow.start(s), "your name"),
    (lambda s: None, "How can I help"),
])
def test_silence_asks_the_right_thing_for_each_flow(state, setup, expect):
    setup(state)
    assert expect in silence.reprompt(state)


def test_silence_during_the_ssn_question_never_says_digits_and_asks_for_last_four_only(state):
    verify_flow.start(state)
    for t in ("Sarah Johnson", "yes", "March 14 1985"):
        assess_turn(state, t)
    assert state.awaiting_ssn
    say = silence.reprompt(state)
    assert "last four digits" in say and "full" not in say.lower().replace("full name", "")


def test_silence_escalates_then_goes_quiet_so_no_quota_is_wasted(state):
    verify_flow.start(state)
    first, second, third = (silence.reprompt(state) for _ in range(3))
    assert "still here" in second
    assert "staff member" in third and state.choice == "staff_or_callback" and not verify_flow.active(state)
    assert silence.reprompt(state) is None and silence.reprompt(state) is None       # no more speech


def test_the_callers_next_words_reset_the_silence_counter(state):
    silence.reprompt(state)
    assert state.silence_count == 1
    assess_turn(state, "hello")
    assert state.silence_count == 0


def test_no_reprompt_during_an_emergency(state):
    assess_turn(state, "I have chest pain")
    assert silence.reprompt(state) is None


def test_the_offer_after_silence_is_handled_in_code(state):
    callback_flow.start(state)
    for _ in range(3):
        silence.reprompt(state)
    d = assess_turn(state, "leave a message please")
    assert d.say and "name" in d.say.lower() and callback_flow.active(state)


class FakeSpeech:
    def __init__(self, on_done=None) -> None:
        self.on_done = on_done

    def __await__(self):
        yield from asyncio.sleep(0).__await__()
        if self.on_done:
            self.on_done()


class RecordingSession:
    def __init__(self, on_say=None) -> None:
        self.said, self.on_say = [], on_say

    def say(self, text, **kw):
        self.said.append(text)
        return FakeSpeech(self.on_say)


@pytest.mark.asyncio
async def test_watch_speaks_at_most_three_reprompts(state):
    session = RecordingSession()
    assert await silence.watch(session, state, wait=0) == 3 == len(session.said)
    assert await silence.watch(session, state, wait=0) == 0                          # still silent: stays quiet


@pytest.mark.asyncio
async def test_watch_stops_as_soon_as_the_caller_speaks(state):
    session = RecordingSession(on_say=lambda: assess_turn(state, "I'm here, sorry"))
    assert await silence.watch(session, state, wait=0) == 1


@pytest.mark.asyncio
async def test_watch_survives_a_failing_session(state):
    class Broken:
        def say(self, *a, **k):
            raise RuntimeError("tts down")
    assert await silence.watch(Broken(), state, wait=0) == 0


def test_agent_registers_a_silence_watch_on_away():
    import agent
    handlers = {}
    session = SimpleNamespace(on=lambda name, fn: handlers.__setitem__(name, fn))
    agent.attach_silence_watch(session, SimpleNamespace(turn=0))
    assert "user_state_changed" in handlers
    handlers["user_state_changed"](SimpleNamespace(new_state="speaking"))             # no task yet: must not fail


# ---- verified caller with no appointment: book or callback, decided in code ------------------------------------------------
def test_no_appointment_then_book_starts_the_booking_questions(state):
    verified(state, MICHAEL)
    say, _ = tools.speak_next_appointment(state)
    assert "book one" in say
    d = assess_turn(state, "yes, please book me one")
    assert booking_flow.active(state) and d.say and state.choice is None


def test_no_appointment_then_callback_starts_the_message_flow(state):
    verified(state, MICHAEL)
    tools.speak_next_appointment(state)
    d = assess_turn(state, "just leave a callback message")
    assert callback_flow.active(state) and "number" in d.say.lower() or callback_flow.active(state)


def test_no_appointment_then_no_thanks_closes_the_offer(state):
    verified(state, MICHAEL)
    tools.speak_next_appointment(state)
    d = assess_turn(state, "no thanks")
    assert state.choice is None and "anything else" in d.say and not booking_flow.active(state)


def test_no_appointment_unclear_answer_is_asked_again_once_then_dropped(state):
    verified(state, MICHAEL)
    tools.speak_next_appointment(state)
    assert "book an appointment, or leave a callback" in assess_turn(state, "purple elephants").say
    assess_turn(state, "banana")
    assert state.choice is None                                                       # no endless nagging


# ---- unclear / interrupted conversation -------------------------------------------------------------------------------------
def test_garbled_answers_never_verify_and_end_with_a_staff_offer(state):
    verify_flow.start(state)
    last = None
    for t in ("uh", "12345", "....", "98765", "...."):
        if verify_flow.active(state):
            last = assess_turn(state, t)
    assert not state.verified and "staff member" in (last.say or "") and not verify_flow.active(state)


def test_caller_changing_the_subject_mid_verification_with_an_emergency_is_handled_first(state):
    verify_flow.start(state)
    d = assess_turn(state, "wait, I'm having trouble breathing")
    assert "911" in d.say and state.emergency and not verify_flow.active(state)


def test_stop_or_wait_during_a_flow_is_not_counted_as_an_unclear_answer(state):
    verify_flow.start(state)
    assert assess_turn(state, "hold on a second").say == verify_flow.WAIT_SAY
    assert state.unclear_count == 0


# ---- greeting delivery ------------------------------------------------------------------------------------------------------
class GreetingSession:
    def __init__(self, interrupted=False, degrade=False) -> None:
        self.tts_chain = SimpleNamespace(stats={"degraded": 0})
        self.calls, self.interrupted, self.degrade = [], interrupted, degrade

    def say(self, text, **kw):
        self.calls.append((text, kw))
        if self.degrade:
            self.tts_chain.stats["degraded"] += 1
        handle = FakeSpeech()
        handle.interrupted = self.interrupted
        return handle


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupted,degrade,delivered", [(False, False, True), (True, False, False), (False, True, False)])
async def test_greeting_is_uninterruptible_and_transcripts_start_only_if_it_was_heard(state, tmp_path, interrupted, degrade, delivered):
    import agent
    from clinic.transcript import TranscriptWriter
    writer = TranscriptWriter(state, directory=tmp_path)
    writer.start()
    session = GreetingSession(interrupted, degrade)
    assert await agent.deliver_greeting(session, state, writer) is delivered
    text, kwargs = session.calls[0]
    assert kwargs["allow_interruptions"] is False and "may be recorded and transcribed for quality purposes" in text
    assert writer.caller_allowed is delivered


@pytest.mark.asyncio
async def test_greeting_with_transcripts_off_says_it_is_not_recorded(state, tmp_path):
    import agent
    from clinic.transcript import TranscriptWriter
    writer = TranscriptWriter(state, directory=tmp_path, enabled=False)
    session = GreetingSession()
    await agent.deliver_greeting(session, state, writer)
    assert "not being recorded" in session.calls[0][0] and not writer.caller_allowed


def test_closed_greeting_has_the_notice_hours_and_911(make_state):
    from clinic.prompts import build_greeting
    text = build_greeting(make_state(CLOSED_EVENING).clock)
    assert "may be recorded and transcribed" in text and "911" in text and "Monday to Friday" in text


def test_default_greeting_is_two_tts_requests_the_first_one_short():
    from clinic.prompts import GREETING
    from clinic.tts_safety import GroupedSentenceTokenizer
    chunks = GroupedSentenceTokenizer().tokenize(GREETING)
    assert len(chunks) == 2 and len(chunks[0]) <= 90 and " ".join(chunks).split() == GREETING.split()


# ---- connection options: retries must not burn the TTS quota -------------------------------------------------------------------
def test_session_uses_one_tts_retry_and_tolerates_brief_stt_llm_outages():
    import agent
    opts = agent.CONN_OPTIONS
    assert opts.tts_conn_options.max_retry == 1 and opts.max_unrecoverable_errors >= 5


@pytest.mark.asyncio
async def test_built_session_carries_the_connection_options_and_the_tts_handle(monkeypatch):
    import agent
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    session = agent.build_session()
    assert session.conn_options is agent.CONN_OPTIONS and hasattr(session.tts_chain, "stats")


@pytest.mark.asyncio
async def test_stt_and_llm_clients_are_built_off_the_event_loop(monkeypatch):
    import threading
    import agent
    monkeypatch.setenv("GROQ_API_KEY", "placeholder-not-real")
    seen = {}
    real = agent.build_clients

    def spy(s):
        seen["thread"] = threading.current_thread().name
        return real(s)

    monkeypatch.setattr(agent, "build_clients", spy)
    session = await agent.build_session_async()
    assert seen["thread"] != threading.main_thread().name and session.stt is not None and session.llm is not None
    assert hasattr(session, "tts_chain")
