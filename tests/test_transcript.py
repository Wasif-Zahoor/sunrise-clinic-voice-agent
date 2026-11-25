"""Per-call transcripts: generation, redaction, saving after abnormal ends, consent gating, and the agent hooks. Offline."""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from clinic import config, gate, transcript as tr
from clinic.transcript import TranscriptWriter, redact
from conftest import CLOSED_EVENING

ROOT = Path(__file__).resolve().parent.parent


class Ticker:
    """A deterministic clock for the writer."""
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 24, 9, 0, 0)

    def __call__(self) -> datetime:
        self.t = self.t.replace(second=(self.t.second + 1) % 60)
        return self.t


def make_writer(state, tmp_path, notice=True) -> TranscriptWriter:
    w = TranscriptWriter(state, directory=tmp_path / "transcripts", now=Ticker())
    w.start()
    if notice:
        w.notice_given(True)
    return w


class FakeSession:
    """Stands in for AgentSession.say: speaks into the transcript like conversation_item_added would."""
    def __init__(self, writer) -> None:
        self.writer, self.said = writer, []

    def say(self, text, **kwargs):
        self.said.append(text)
        self.writer.agent(text)
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(None)
        return fut


async def hear(agent, session, text):
    """Drive the REAL Receptionist.on_user_turn_completed hook (caller logging + SSN gate + turn logic)."""
    from livekit.agents import ChatContext, ChatMessage, StopResponse
    from agent import Receptionist
    scrubbed = gate.scrub_transcript(agent._state, text)                 # what stt_node does before the hook
    ctx = ChatContext()
    msg = ChatMessage(role="user", content=[scrubbed])
    try:
        await Receptionist.on_user_turn_completed(agent, ctx, msg)
    except StopResponse:
        pass


@pytest.fixture
def agent_and_session(state, tmp_path, monkeypatch):
    import agent as agent_module
    writer = make_writer(state, tmp_path)
    session = FakeSession(writer)
    a = agent_module.Receptionist(state, writer)
    monkeypatch.setattr(agent_module.Receptionist, "session", property(lambda self: session))
    return a, session, writer


def text_of(writer) -> str:
    return writer.path.read_text(encoding="utf-8")


# ---- generation -----------------------------------------------------------------------------------------------------------
def test_transcript_file_has_ids_times_speakers_and_order(state, tmp_path):
    w = make_writer(state, tmp_path)
    w.agent("Thank you for calling Sunrise Family Clinic.")
    w.caller("I need to check my appointment")
    w.agent("Sure. May I have your full name, please?")
    w.close("test over")
    body = text_of(w)
    assert w.path.parent == tmp_path / "transcripts" and w.path.suffix == ".txt" and state.call_id in w.path.name
    assert f"Call ID:      {state.call_id}" in body and "Started:" in body and "Call ended:" in body
    lines = [ln for ln in body.splitlines() if re.match(r"\[\d\d:\d\d:\d\d\] (CALLER|AGENT)", ln)]
    assert [ln.split("] ")[1].split(":")[0] for ln in lines] == ["AGENT", "CALLER", "AGENT"]
    assert "reason: test over" in body and "caller turns 1, agent turns 2" in body


def test_each_call_gets_its_own_file(make_state, tmp_path):
    a, b = make_writer(make_state(), tmp_path), make_writer(make_state(), tmp_path)
    a.close(), b.close()
    assert a.path != b.path and len(list((tmp_path / "transcripts").glob("*.txt"))) == 2


# ---- saving when the call ends abnormally ---------------------------------------------------------------------------------
def test_lines_are_on_disk_even_if_the_call_never_closes_cleanly(state, tmp_path):
    w = make_writer(state, tmp_path)
    w.caller("hello there")
    w.agent("Hi, how can I help?")
    # no close(): the process is killed. A fresh read of the file must already show both lines.
    body = w.path.read_text(encoding="utf-8")
    assert "CALLER: hello there" in body and "AGENT: Hi, how can I help?" in body and "Call ended:" not in body


def test_close_is_idempotent_and_late_writes_are_ignored(state, tmp_path):
    w = make_writer(state, tmp_path)
    w.agent("one")
    w.close("session closed (error)")
    w.close("session shutdown")                       # the shutdown callback runs after the close event
    w.agent("two")                                    # nothing may be appended after the end marker
    body = text_of(w)
    assert body.count("Call ended:") == 1 and "two" not in body and "session closed (error)" in body


def test_session_events_write_agent_lines_errors_and_the_end_marker(state, tmp_path):
    w = make_writer(state, tmp_path)
    handlers = {}
    session = SimpleNamespace(on=lambda name, fn: handlers.__setitem__(name, fn))
    tr.attach(session, w)
    assert set(handlers) == {"conversation_item_added", "error", "close"}
    item = lambda role, text, interrupted=False, type_="message": SimpleNamespace(
        item=SimpleNamespace(type=type_, role=role, text_content=text, interrupted=interrupted))
    handlers["conversation_item_added"](item("assistant", "Welcome."))
    handlers["conversation_item_added"](item("user", "caller words must come from the hook, not here"))
    handlers["conversation_item_added"](item("assistant", "Your next appointment is", interrupted=True))
    handlers["conversation_item_added"](item("assistant", "", type_="function_call"))
    handlers["error"](SimpleNamespace(source=SimpleNamespace(), error=TimeoutError("secret caller words")))
    handlers["close"](SimpleNamespace(reason=SimpleNamespace(value="error")))
    body = text_of(w)
    assert "AGENT: Welcome." in body and "AGENT (interrupted by caller): Your next appointment is" in body
    assert "must come from the hook" not in body and "secret caller words" not in body
    assert "EVENT: SimpleNamespace error: TimeoutError" in body
    assert "reason: session closed (error)" in body


def test_handler_exceptions_never_escape(state, tmp_path):
    w = make_writer(state, tmp_path)
    handlers = {}
    tr.attach(SimpleNamespace(on=lambda name, fn: handlers.__setitem__(name, fn)), w)
    for name in handlers:
        handlers[name](object())                      # malformed events
        handlers[name](None)


# ---- the writer must never break a call -----------------------------------------------------------------------------------
def test_unwritable_location_does_not_raise(state, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    w = TranscriptWriter(state, directory=blocker / "sub")      # mkdir fails
    w.start(), w.notice_given(True), w.caller("x"), w.agent("y"), w.event("z"), w.close("done")
    assert w.path is None


def test_write_failure_midway_is_swallowed(state, tmp_path, caplog):
    w = make_writer(state, tmp_path)
    w._fh.close()                                     # simulates a disk error / closed handle
    w.agent("this write fails"), w.caller("so does this"), w.close("end")
    assert caplog.text.count("transcript writing failed") == 1  # one warning, no spam, no text in it
    assert "this write fails" not in caplog.text


def test_disabled_transcripts_create_nothing(state, tmp_path):
    w = TranscriptWriter(state, directory=tmp_path / "transcripts", enabled=False)
    w.start(), w.notice_given(True), w.caller("secret"), w.agent("hello"), w.close("x")
    assert not (tmp_path / "transcripts").exists() and w.path is None and not w.caller_allowed


def test_settings_from_environment(monkeypatch):
    monkeypatch.delenv("TRANSCRIPTS", raising=False)
    assert config.transcripts_enabled() is True                      # on by default
    for off in ("off", "0", "false", "NO"):
        monkeypatch.setenv("TRANSCRIPTS", off)
        assert config.transcripts_enabled() is False
    monkeypatch.setenv("TRANSCRIPT_DIR", "calls")
    assert config.transcript_dir() == ROOT / "calls"
    monkeypatch.setenv("TRANSCRIPT_DIR", str(ROOT / "elsewhere"))
    assert config.transcript_dir() == ROOT / "elsewhere"


def test_transcripts_folder_is_git_ignored_and_documented():
    ignore = (ROOT / ".gitignore").read_text(encoding="utf8").splitlines()
    assert any(line.strip().rstrip("/") == "transcripts" for line in ignore)
    example = (ROOT / ".env.example").read_text(encoding="utf8")
    assert "TRANSCRIPTS=" in example and "TRANSCRIPT_DIR=" in example


# ---- consent: nothing from the caller is stored before the notice was heard ------------------------------------------------
def test_caller_words_are_not_stored_before_the_notice_was_heard(state, tmp_path):
    w = make_writer(state, tmp_path, notice=False)
    w.caller("my birthday is whatever")
    assert "whatever" not in text_of(w) and "speech not transcribed" in text_of(w)
    w.notice_given(True)
    w.caller("how are you")
    assert "CALLER: how are you" in text_of(w)


def test_failed_notice_means_the_call_is_never_transcribed(state, tmp_path):
    w = make_writer(state, tmp_path, notice=False)
    w.notice_given(False)
    w.caller("anything at all")
    body = text_of(w)
    assert "anything at all" not in body and "NOT delivered" in body


# ---- redaction ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,must_not_contain", [
    ("my social is 123-45-6789", "6789"),
    ("last four 3344", "3344"),
    ("call me on (555) 123-4567", "4567"),
    ("call me on 555.123.4567 please", "123"),
    ("it's +1 555 123 4567", "4567"),
    ("email john.smith@example.com", "john.smith"),
    ("born 1985-03-14", "1985"),
    ("born 3/14/1985", "1985"),
    ("born March 14th, 1985", "1985"),
    ("born 14 March 1985", "1985"),
    ("you owe $45.00", "45"),
    ("member id BS12345 on file", "BS12345"),
    ("it is five five five one two three four", "five five"),
    ("double three four four", "three four"),
    ("take Metformin 500 mg daily", "500"),
    ("policy 123456789", "123456789"),
])
def test_patterns_are_removed(raw, must_not_contain):
    out = redact(raw)
    assert must_not_contain.lower() not in out.lower(), out


def test_known_names_and_emails_are_masked_including_spelled_out():
    from clinic.services import spell_email
    out = redact(f"Thanks Sarah Johnson. I have {spell_email('sjohnson@example.com')}, is that right?",
                 known=["Sarah", "Johnson"], emails=["sjohnson@example.com"])
    assert "Sarah" not in out and "Johnson" not in out and "example" not in out and "[EMAIL]" in out
    assert redact("ordinary words stay", known=["Sarah"]) == "ordinary words stay"


@pytest.mark.parametrize("raw", [
    "it is john at gmail dot com",
    "j o h n s m i t h at e x a m p l e dot c o m",
    "john at the rate example dot com",
    "j o h n at example dot co dot uk",
])
def test_addresses_said_as_words_are_masked_without_knowing_them(raw):
    out = redact(raw)
    assert "gmail" not in out and "example" not in out and "j o h n" not in out and "[EMAIL]" in out, out


def test_911_survives_redaction():
    assert "911" in redact("Please hang up and call 911 now. Your balance is 4500 dollars, call 5551234567.")
    assert "5551234567" not in redact("call 5551234567") and "4500" not in redact("4500 dollars")
    assert "1 2 3" not in redact("my number is 5 5 5 1 2 3 4") and "[digits]" in redact("my number is 5 5 5 1 2 3 4")


def test_visit_and_call_reasons_are_masked_in_agent_readbacks():
    out = redact("for annual check-up. Regarding a billing question.", reasons=["annual check-up", "a billing question"])
    assert "check-up" not in out and "billing" not in out and out.count("[REASON]") == 2


def test_ordinary_scheduling_words_survive():
    out = redact("Your next appointment is Thursday, December 3rd at 10:30 AM with Dr. Patel. Is that right?")
    assert "December 3rd at 10:30 AM with Dr. Patel" in out


# ---- end to end through the real agent hook --------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_verified_call_leaves_no_credentials_in_the_transcript(agent_and_session, state):
    agent, session, writer = agent_and_session
    from clinic import verify_flow
    session.say(verify_flow.start(state).say)
    for utterance in ("Sarah Johnson", "yes", "March 14th 1985", "three three four four"):
        await hear(agent, session, utterance)
    assert state.verified, "the scripted verification should have succeeded"
    await hear(agent, session, "what do I owe, I think about forty five dollars")
    body = text_of(writer)
    for needle in ("Sarah", "Johnson", "1985", "March 14", "3344", "three three four four", "forty five"[:0] or "3-3"):
        assert needle not in body, f"{needle!r} leaked into the transcript:\n{body}"
    assert "[name provided - not stored]" in body and "[date of birth provided - not stored]" in body
    assert "[last four digits provided - not stored]" in body
    assert body.index("name provided") < body.index("date of birth provided") < body.index("last four digits provided")


@pytest.mark.asyncio
async def test_booking_details_are_withheld_and_agent_readbacks_are_masked(make_state, tmp_path, monkeypatch):
    import agent as agent_module
    from clinic import booking_flow
    st = make_state()
    writer = make_writer(st, tmp_path)
    session = FakeSession(writer)
    a = agent_module.Receptionist(st, writer)
    monkeypatch.setattr(agent_module.Receptionist, "session", property(lambda self: session))
    session.say(booking_flow.start(st, "yes").say)
    for utterance in ("My name is John Smith", "yes", "April 9 1992", "555 123 4567", "john at example dot com", "Blue Shield",
                      "B S 1 2 3 4 5", "annual check-up", "no preference"):
        await hear(a, session, utterance)
    body = text_of(writer)
    for needle in ("John", "Smith", "1992", "April 9", "555", "4567", "example", "john@", "Blue Shield", "BS12345",
                   "annual check-up"):
        assert needle.lower() not in body.lower(), f"{needle!r} leaked:\n{body}"
    assert "provided - not stored" in body


@pytest.mark.asyncio
async def test_emergency_and_third_party_lines_are_kept_readable(agent_and_session, state):
    agent, session, writer = agent_and_session
    await hear(agent, session, "I have chest pain")
    body = text_of(writer)
    assert "CALLER: I have chest pain" in body and "911" in body


@pytest.mark.asyncio
async def test_callback_after_hours_transcript(make_state, tmp_path, monkeypatch):
    import agent as agent_module
    from clinic import callback_flow
    st = make_state(CLOSED_EVENING)
    writer = make_writer(st, tmp_path)
    session = FakeSession(writer)
    a = agent_module.Receptionist(st, writer)
    monkeypatch.setattr(agent_module.Receptionist, "session", property(lambda self: session))
    session.say(callback_flow.start(st).say)
    for utterance in ("Jane Doe", "555 111 2222", "a billing question", "yes"):
        await hear(a, session, utterance)
    body = text_of(writer)
    assert "Jane" not in body and "Doe" not in body and "111" not in body and "2222" not in body
    assert "next business day" in body and body.count("CALLER:") == 4


# ---- command line ---------------------------------------------------------------------------------------------------------
def test_cli_lists_and_shows_transcripts(state, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path / "transcripts"))
    done, unfinished = make_writer(state, tmp_path), None
    done.agent("hello")
    done.close("finished")
    assert tr.main(["list"]) == 0
    out = capsys.readouterr().out
    assert done.path.name in out and "complete" in out
    assert tr.main(["show", "latest"]) == 0 and "AGENT: hello" in capsys.readouterr().out
    assert tr.main(["show", "no-such-call"]) == 1
