"""Round 9 (second live call, 2026-11-25): the refill never reached the tool, "Thank you" became a name, booking took three LLM round
trips, and a dropped idle connection switched the caller's voice. Offline."""

from __future__ import annotations

import logging

import pytest
from livekit.agents import APIConnectionError

from clinic import booking_flow, callback_flow, gate, intent, refill_flow, speech, tools, verify_flow
from clinic import services as svc
from clinic import tts_chain as tc
from clinic.turn import assess_turn
from conftest import CLOSED_EVENING, MICHAEL, SARAH
from test_tts_chain import FakeTTS, audible, make_chain, speak


class Call:
    def __init__(self, state):
        self.state, self.says = state, []

    def hear(self, text):
        d = assess_turn(self.state, gate.scrub_transcript(self.state, text))
        if d.say:
            self.says.append(d.say)
        return d


def verify_as(call, who=SARAH):
    name, dob, last4 = who
    spoken_dob = {"1985-03-14": "March 14 1985", "1972-11-08": "November 8 1972"}[dob]
    spoken_last4 = {"3344": "three three four four", "1020": "one zero two zero"}[last4]
    verify_flow.start(call.state)
    for t in (name, "yes", spoken_dob, spoken_last4):
        call.hear(t)
    assert call.state.verified


def requests(conn):
    return [tuple(r) for r in conn.execute("SELECT request_type, status FROM requests ORDER BY id")]


# ---- the failed refill, replayed ---------------------------------------------------------------------------------------------------
def test_the_failed_call_replayed_garbled_drug_names_end_in_a_saved_refill(state, conn):
    c = Call(state)
    assert c.hear("I would like to get my prescription refill.").say == verify_flow.Q_NAME
    for t in ("Sarah Johnson", "yes", "March 14, 1985"):
        c.hear(t)
    d = c.hear("three three four four")
    assert d.say == "Thank you, you're verified. Which medication would you like to refill?" and refill_flow.active(state)
    d = c.hear("I'd like to leave food.")                                   # nothing like a medication: asked again, not given up
    assert "couldn't find that one" in d.say and refill_flow.active(state) and state.refill_tries == 1
    d = c.hear("Lisey Nuperell")                                           # what speech-to-text made of "Lisinopril"
    assert d.say == "Lisinopril 10mg at CVS Main St. Is that right?" and state.pending.kind == "refill"
    assert not refill_flow.active(state)
    d = c.hear("yes")
    assert d.say.startswith("Done. Your refill request has been submitted") and "approved" not in d.say.lower()
    assert requests(conn) == [("refill", "submitted_to_pharmacy")]


def test_the_spelled_out_name_is_matched(state, conn):
    c = Call(state)
    verify_as(c)
    c.hear("I need a refill")
    assert c.hear("The spelling is L-I-S-I-L.").say.startswith("Lisinopril 10mg")
    c.hear("no")
    state.pending = None
    refill_flow.begin(state)
    assert c.hear("N-O-P-R-I-L").say.startswith("Lisinopril 10mg")


def test_a_refill_with_no_refills_left_goes_to_the_doctor_and_is_never_called_approved(state, conn):
    c = Call(state)
    verify_as(c)
    d = c.hear("I need to refill my metformin")                             # named in the opening sentence: no extra question
    assert d.say.startswith("Metformin 500mg at CVS Main St.") and "doctor for approval" in d.say
    done = c.hear("yes").say
    assert "sent the request to your doctor for approval" in done and "approved." not in done.replace("for approval", "")
    assert requests(conn) == [("refill", "pending_doctor_approval")]


def test_a_confirmed_refill_is_not_saved_twice(state, conn):
    c = Call(state)
    verify_as(c)
    c.hear("refill my lisinopril")
    c.hear("yes")
    c.hear("yes")
    assert len(requests(conn)) == 1


def test_three_misses_offer_staff_or_a_callback(state):
    c = Call(state)
    verify_as(c)
    c.hear("I need a refill")
    for t in ("the blue one", "no idea what it is called"):
        assert "couldn't find that one" in c.hear(t).say
    d = c.hear("the little round one")
    assert d.say == verify_flow.STAFF_OFFER_SAY and state.choice == "staff_or_callback" and not refill_flow.active(state)
    assert "name" in c.hear("leave a callback message").say.lower() or callback_flow.active(state)


# ---- fuzzy matching is limited to the caller's own medications --------------------------------------------------------------------------
def test_another_patients_medicine_is_not_found_and_nothing_is_disclosed(state, conn):
    c = Call(state)
    verify_as(c, MICHAEL)                                                   # Michael takes Atorvastatin; Sarah's drugs are not his
    c.hear("I need a refill")
    d = c.hear("lisinopril")
    assert "couldn't find that one" in d.say and "Lisinopril" not in d.say and "Metformin" not in d.say
    assert c.hear("atorvastatin").say.startswith("Atorvastatin")


@pytest.mark.parametrize("sentence", ["I would like to get my prescription refill", "what are your office hours", "my balance please",
                                      "hello there", "can you schedule an appointment for me", "thank you so much", "March fourteenth"])
def test_ordinary_sentences_never_match_a_medication(state, sentence):
    c = Call(state)
    verify_as(c)
    assert svc.has_medication(state, sentence) is False


@pytest.mark.parametrize("heard", ["Lisey Nuperell", "L-I-S-I-L N-O-P-R-I-L", "lisinopril", "lisinipril", "Lisa nopril"])
def test_garbled_versions_of_a_medication_match_it(state, heard):
    c = Call(state)
    verify_as(c)
    assert svc.has_medication(state, heard) is True


def test_ambiguity_is_reported_not_guessed():
    rows = [{"name": "Atenolol 25mg"}, {"name": "Atenolol 50mg"}]
    assert len(svc._select_medications("atenolol", rows)) == 2
    close = [{"name": "Metformin 500mg"}, {"name": "Metoprolol 25mg"}]
    assert len(svc._select_medications("metfo", close)) == 1


def test_the_medication_check_requires_verification(state):
    with pytest.raises(svc.ServiceError):
        svc.has_medication(state, "lisinopril")


# ---- the refill conversation can be left ----------------------------------------------------------------------------------------------------
def test_wait_cancel_staff_and_emergency_leave_the_refill_flow_cleanly(state):
    c = Call(state)
    verify_as(c)
    c.hear("I need a refill")
    assert c.hear("hold on a second").say == verify_flow.WAIT_SAY and state.refill_tries == 0
    assert c.hear("never mind").note and not refill_flow.active(state)
    c.hear("I need a refill")
    assert "person" in c.hear("can I speak to a real person").note and not refill_flow.active(state)
    c.hear("I need a refill")
    d = c.hear("I'm having chest pain")
    assert "911" in d.say and not refill_flow.active(state) and state.emergency


def test_an_unverified_refill_request_through_the_llm_tool_is_still_refused(state):
    assert tools.prepare_refill(state, "lisinopril").startswith("NOT ALLOWED (not_verified)")


def test_a_refill_for_someone_else_is_left_to_the_llm_note(state):
    d = assess_turn(state, "I need to refill my husband's prescription")
    assert d.say is None and d.note and "someone else" in d.note and state.after_verify is None


# ---- booking and lookups are routed in code ------------------------------------------------------------------------------------------------
def test_scheduling_starts_the_booking_questions_without_the_llm(state):
    d = assess_turn(state, "Okay, can you schedule an appointment for me?")
    assert d.say == booking_flow.Q_FIRST_VISIT and booking_flow.active(state)
    d = assess_turn(state, "No")
    assert d.say.startswith("Let me verify your identity first") and verify_flow.active(state)


def test_a_verified_caller_who_wants_to_book_skips_what_is_already_on_file(state):
    c = Call(state)
    verify_as(c)
    d = c.hear("I'd like to book an appointment")
    assert booking_flow.active(state) and d.say and "date of birth" not in d.say.lower()


def test_verified_lookups_are_answered_immediately(state):
    c = Call(state)
    verify_as(c)
    assert c.hear("how much do I owe").say.startswith("Your balance due is $45.00")
    assert "lab results" in c.hear("what's the status of my lab results").say.lower()
    assert "Thursday, December 3rd" in c.hear("when is my next appointment").say


def test_scheduling_is_not_routed_while_closed_or_mid_conversation(make_state):
    closed = make_state(CLOSED_EVENING)
    say = intent.route(closed, "schedule an appointment")            # after hours: the fixed closed answer, nothing is started
    assert say == intent.closed_say(make_state(CLOSED_EVENING)) and not booking_flow.active(closed) and not verify_flow.active(closed)
    s = make_state()
    verify_flow.start(s)
    assert intent.route(s, "schedule an appointment") is None


# ---- names ---------------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", ["Thank you.", "Thanks", "thank you very much", "Hello", "Okay", "Please", "goodbye", "Nothing", "you're welcome"])
def test_an_acknowledgement_is_not_a_name(text):
    assert speech.extract_name(text)[0] == "unusable"


@pytest.mark.parametrize("text,expected", [("Sarah Johnson", "Sarah Johnson"), ("my name is Hope Rose", "Hope Rose"),
                                           ("it's Joy Hill, thank you", "Joy Hill"), ("Maria Garcia please", "Maria Garcia")])
def test_real_names_still_work(text, expected):
    assert speech.extract_name(text) == ("ok", expected)


def test_thank_you_at_the_name_question_is_asked_again_not_read_back(state):
    c = Call(state)
    c.hear("I would like to get my prescription refill")
    d = c.hear("Thank you.")
    assert "didn't catch your name" in d.say and state.vstep == "name" and state.pending_name is None


# ---- TTS: a dropped idle connection must not change the voice ------------------------------------------------------------------------------------
class ConnFakeTTS(FakeTTS):
    """script: conn (APIConnectionError caused by a dropped connection) | ok | 500"""


def _install_conn_step():
    import clinic  # noqa: F401
    from test_tts_chain import _FakeStream
    original = _FakeStream._run

    async def run(self, out):
        owner = self._tts
        if owner.script and owner.script[0] == "conn" or (len(owner.script) > 1 and owner.script[0] == "conn"):
            owner.calls += 1
            owner.script.pop(0) if len(owner.script) > 1 else None
            try:
                raise ConnectionResetError("server closed the idle connection")
            except ConnectionResetError as inner:
                raise APIConnectionError() from inner
        return await original(self, out)
    return _FakeStream, original, run


@pytest.fixture
def conn_steps(monkeypatch):
    stream, original, run = _install_conn_step()
    monkeypatch.setattr(stream, "_run", run)


@pytest.mark.asyncio
async def test_a_dropped_connection_is_retried_on_the_same_provider_before_any_fallback(tmp_path, conn_steps, caplog):
    caplog.set_level(logging.INFO)
    dg, local = FakeTTS("deepgram", ["conn", "ok"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    assert audible(await speak(chain)) and chain.active == "deepgram"
    assert (dg.calls, local.calls) == (2, 0) and chain.stats == {"degraded": 0, "fallbacks": 0} and chain.strikes["deepgram"] == 0
    assert "retrying once" in caplog.text and "caused by ConnectionResetError" in caplog.text


@pytest.mark.asyncio
async def test_a_connection_that_stays_down_falls_back_after_the_single_retry(tmp_path, conn_steps):
    dg, local = FakeTTS("deepgram", ["conn"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    assert audible(await speak(chain, "first")) and chain.active == "local" and dg.calls == 2 and chain.strikes["deepgram"] == 1
    assert audible(await speak(chain, "second")) and "deepgram" in chain.dead and dg.calls == 4       # two strikes: given up


@pytest.mark.asyncio
async def test_an_error_status_is_not_retried(tmp_path):
    dg, local = FakeTTS("deepgram", ["500"]), FakeTTS("local", ["ok"])
    chain = make_chain(tmp_path, dg, local)
    await speak(chain)
    assert dg.calls == 1 and local.calls == 1


@pytest.mark.asyncio
async def test_every_chunk_logs_its_timing_without_any_text(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    chain = make_chain(tmp_path, FakeTTS("deepgram", ["ok"]))
    await speak(chain, "Sarah Johnson, your balance is $45.00")
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("TTS chunk provider=deepgram"))
    assert "chars=" in line and "first_audio=" in line and "synthesis=" in line and "audio=" in line
    assert "Sarah" not in caplog.text and "$45" not in caplog.text


def test_the_local_voice_is_warmed_after_the_greeting_not_during_it():
    import inspect
    import agent
    source = inspect.getsource(agent.entrypoint)
    assert source.index("await deliver_greeting") < source.index(".prewarm()")
