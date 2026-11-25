"""The verification conversation: strict one-question-at-a-time flow, natural dates, waits and
interruptions, attempt limits, no duplicate prompts or calls. Synthetic patients only; no network."""

from __future__ import annotations

import re
from datetime import date

import pytest

from clinic import dates, gate, tools, verify_flow
from clinic.turn import assess_turn
from clinic.verify_flow import (CLOSED_SAY, LOCKED_SAY, Q_DOB, Q_NAME, Q_NAME_SPELL, Q_SSN, RETRY_SAY,
                                STAFF_OFFER_SAY, WAIT_SAY, q_name_confirm)
from conftest import CLOSED_EVENING, SARAH

TODAY = date(2026, 11, 24)


class Call:
    """Simulates the caller side exactly as the agent does: STT scrub, then the turn hook."""

    def __init__(self, state):
        self.state = state
        self.says: list[str] = []
        self.notes: list[str] = []

    def hear(self, text: str):
        d = assess_turn(self.state, gate.scrub_transcript(self.state, text))
        if d.say:
            self.says.append(d.say)
        if d.note:
            self.notes.append(d.note)
        return d


def to_dob(call, name="Sarah Johnson"):
    """Give the name and confirm the read-back, so the date of birth is asked next."""
    call.hear(name)
    call.hear("Yes, that's right")


def to_ssn(call):
    to_dob(call)
    call.hear("March 14 1985")


@pytest.fixture
def call(state):
    c = Call(state)
    r = verify_flow.start(state)
    c.says.append(r.say)
    return c


# ===== natural date handling ==========================================================================
@pytest.mark.parametrize("spoken", [
    "February 17, 1973", "17 February 1973", "02/17/1973", "2/17/1973", "1973-02-17", "Feb 17th, 1973",
    "the seventeenth of February nineteen seventy-three", "February seventeenth, nineteen seventy three",
    "it's February 17th 1973 thanks", "my birthday is 17th February, 1973",
])
def test_natural_spoken_dates_normalize_to_iso(spoken):
    r = dates.parse_spoken_date(spoken, TODAY)
    assert (r.status, r.iso) == ("ok", "1973-02-17"), spoken


@pytest.mark.parametrize("spoken,iso", [
    ("twenty second of March two thousand five", "2005-03-22"), ("January first nineteen ninety", "1990-01-01"),
    ("July 4 1976", "1976-07-04"), ("December thirty first nineteen oh five", "1905-12-31"),
    ("13/02/1990", "1990-02-13"),     # 13 can only be a day, so this is unambiguous
])
def test_more_date_forms(spoken, iso):
    assert dates.parse_spoken_date(spoken, TODAY).iso == iso


def test_ambiguous_numeric_date_is_flagged_not_guessed():
    r = dates.parse_spoken_date("03/04/1985", TODAY)
    assert r.status == "ambiguous" and r.iso is None
    assert r.candidates == ["1985-03-04", "1985-04-03"]
    assert dates.parse_spoken_date("05/05/1985", TODAY).iso == "1985-05-05"   # same either way: not ambiguous
    assert dates.resolve_ambiguity("April third", r.candidates) == "1985-04-03"
    assert dates.resolve_ambiguity("the first one", r.candidates) == "1985-03-04"
    assert dates.resolve_ambiguity("hmm", r.candidates) is None


@pytest.mark.parametrize("spoken", ["February 30, 1973", "February 29, 1975", "13/13/1985", "31 April 1980",
                                    "February 17, 2090", "February 17, 1850", "99/99/9999"])
def test_invalid_dates(spoken):
    assert dates.parse_spoken_date(spoken, TODAY).status == "invalid", spoken


@pytest.mark.parametrize("spoken,missing", [("February 17", "year"), ("February 1973", "day"),
                                            ("17 1973", "month"), ("nineteen seventy three", "month"),
                                            ("February 17 73", "year")])
def test_incomplete_dates(spoken, missing):
    r = dates.parse_spoken_date(spoken, TODAY)
    assert r.status == "incomplete" and missing in r.missing, spoken


def test_no_date_at_all():
    assert dates.parse_spoken_date("hello there", TODAY).status == "none"
    assert dates.parse_spoken_date("", TODAY).status == "none"


def test_leap_day_is_valid():
    assert dates.parse_spoken_date("February 29, 1976", TODAY).iso == "1976-02-29"


# ===== strict sequential conversation ==========================================================================
def test_questions_come_one_at_a_time_in_order(call, state):
    assert call.says == [Q_NAME] and state.vstep == "name"
    d = call.hear("My name is Sarah Johnson.")
    assert d.say == q_name_confirm("Sarah Johnson") and state.vstep == "name_confirm"   # the name is read back
    assert not state.awaiting_ssn
    d = call.hear("Yes, that's right")
    assert d.say == Q_DOB and state.vstep == "dob" and not state.awaiting_ssn      # DOB asked, SSN NOT yet
    d = call.hear("March fourteenth, nineteen eighty-five")
    assert d.say == Q_SSN and state.awaiting_ssn and state.pending_dob == "1985-03-14"
    d = call.hear("three three four four")
    assert d.say is None and "VERIFIED" in d.note.upper() and state.verified_patient_id == "P001"
    for q in (Q_NAME, Q_DOB, Q_SSN):
        assert call.says.count(q) == 1, q                       # each question asked exactly once
    for s in call.says:
        assert s.count("?") == 1                                # never two questions in one prompt
        assert "YYYY" not in s and "format" not in s.lower() and not re.search(r"\d{4}-\d{2}-\d{2}", s)
        assert "callback" not in s.lower()                      # no premature callback


def test_asks_for_date_of_birth_naturally(call):
    to_dob(call)
    assert call.says[-1] == "Thank you. May I have your date of birth, please?"


def test_name_hint_from_the_llm_is_read_back_before_anything_else(state):
    r = verify_flow.start(state, "Sarah Johnson")
    assert r.say == q_name_confirm("Sarah Johnson") and state.pending_name == "Sarah Johnson"
    assert state.vstep == "name_confirm"


def test_name_and_date_in_one_answer_are_not_asked_again(call, state):
    d = call.hear("Sarah Johnson, born March 14th 1985")
    assert d.say == q_name_confirm("Sarah Johnson")              # the name is still read back first
    d = call.hear("yes")
    assert d.say == Q_SSN and state.awaiting_ssn                 # then straight to the digits
    assert Q_DOB not in call.says


def test_first_name_only_asks_for_the_last_name_then_reads_back(call, state):
    assert call.hear("Sarah").say == "Thanks, Sarah. And your last name?"
    assert state.vstep == "name" and state.unclear_count == 0
    assert call.hear("Johnson").say == q_name_confirm("Sarah Johnson") and state.pending_name == "Sarah Johnson"


@pytest.mark.parametrize("answer", ["Sarah Johnson.", "this is Sarah Johnson", "It's Sarah Johnson, please",
                                    "um, yeah, my name is sarah johnson", "My full name is Sarah Johnson.",
                                    "Yeah, my full name is Sarah Johnson"])
def test_name_answers_in_different_wording(call, state, answer):
    assert call.hear(answer).say == q_name_confirm("Sarah Johnson") and state.pending_name == "Sarah Johnson"


def test_spelled_out_names_are_understood(call, state):
    assert call.hear("S-A-R-A-H").say == "Thanks, Sarah. And your last name?"
    assert call.hear("J-O-H-N-S-O-N").say == q_name_confirm("Sarah Johnson")


def test_wrong_name_read_back_leads_to_spelling_not_to_a_burned_attempt(call, state):
    call.hear("Sara Jonson")
    d = call.hear("No, that's wrong")
    assert d.say == Q_NAME_SPELL and state.vstep == "name_spell" and state.failed_attempts == 0
    assert call.hear("S-A-R-A-H").say == "Thanks, Sarah. And your last name?"
    assert call.hear("J-O-H-N-S-O-N").say == q_name_confirm("Sarah Johnson")
    assert call.hear("yes").say == Q_DOB


def test_name_correction_in_the_same_breath_is_taken(call, state):
    call.hear("Sara Jonson")
    assert call.hear("No, it's Sarah Johnson").say == q_name_confirm("Sarah Johnson")


def test_name_confirmation_needs_a_clear_answer(call, state):
    call.hear("Sarah Johnson")
    d = call.hear("maybe")
    assert "yes or no" in d.say and state.vstep == "name_confirm" and state.pending_name == "Sarah Johnson"


def test_i_already_told_you_gets_an_apology_and_the_same_question_not_an_argument(call, state):
    call.hear("Sarah Johnson")
    d = call.hear("I already gave you that")
    assert d.say.startswith("I'm sorry about that.") and "Is that right?" in d.say and state.vstep == "name_confirm"


def test_dob_step_rejects_without_advancing_and_never_exposes_formats(call, state):
    to_dob(call)
    d = call.hear("February 30th 1985")
    assert "month, day and year" in d.say and state.vstep == "dob" and not state.awaiting_ssn
    assert "YYYY" not in d.say and not re.search(r"\d{4}-\d{2}-\d{2}", d.say)
    assert call.hear("March 14").silent                              # half a date: held, the caller may be about to say the year
    d = call.hear("banana")                                          # nothing useful follows: ask once
    assert "month, day and year" in d.say and state.vstep == "dob" and not state.awaiting_ssn
    assert state.failed_attempts == 0 and state.unclear_count == 2   # unclear answers are not failed attempts
    assert call.hear("banana").say == STAFF_OFFER_SAY                # the third in a row offers staff, not earlier


def test_ambiguous_date_gets_one_clarifying_question(call, state):
    to_dob(call)
    d = call.hear("03/04/1985")
    assert state.vstep == "dob_clarify" and "March 4th, 1985" in d.say and "April 3rd, 1985" in d.say
    d = call.hear("April third")
    assert d.say == Q_SSN and state.pending_dob == "1985-04-03"


def test_unclear_clarification_is_asked_again_briefly(call, state):
    to_dob(call)
    call.hear("03/04/1985")
    d = call.hear("what?")
    assert "which was it" in d.say.lower() and state.vstep == "dob_clarify"


# ===== waits, fillers, interruptions ==========================================================================
@pytest.mark.parametrize("step_setup,filler", [
    (lambda c: None, "Sorry"), (lambda c: None, "Wait"), (lambda c: None, "one moment please"),
    (lambda c: None, "hold on, let me check"), (lambda c: None, "um"),
    (lambda c: c.hear("Sarah Johnson"), "hold on, let me think"),
    (lambda c: to_dob(c), "Sorry, hold on a second"),
    (lambda c: to_ssn(c), "one moment"),
])
def test_waiting_or_fillers_get_a_brief_reply_and_nothing_advances(call, state, step_setup, filler):
    step_setup(call)
    before = (state.vstep, state.awaiting_ssn, state.pending_name, state.pending_dob, state.failed_attempts)
    d = call.hear(filler)
    assert d.say == WAIT_SAY
    assert (state.vstep, state.awaiting_ssn, state.pending_name, state.pending_dob, state.failed_attempts) == before
    assert state.unclear_count == 0                             # waiting is never penalized


def test_waiting_does_not_lose_progress(call, state):
    to_dob(call)
    call.hear("hold on")
    assert call.hear("March 14th 1985").say == Q_SSN


@pytest.mark.parametrize("utterance,fragment", [
    ("Can I speak to a real person?", "request_staff_transfer"), ("Let me talk to someone", "request_staff_transfer"),
    ("Never mind, forget it", "cancelled"),
])
def test_caller_can_leave_verification_without_a_stuck_state(call, state, utterance, fragment):
    call.hear("Sarah Johnson")
    d = call.hear(utterance)
    assert d.say is None and fragment in d.note
    assert state.vstep == "idle" and not state.awaiting_ssn and state.pending_name is None and state.ssn_buffer == ""
    assert not verify_flow.active(state)


@pytest.mark.parametrize("utterance", ["Actually I'm a new patient", "It's my first visit", "I'm sorry, I forgot. I'm a new patient."])
def test_saying_new_patient_hands_over_to_the_scripted_booking_not_the_llm(call, state, utterance):
    to_dob(call)
    d = call.hear(utterance)
    assert d.say and d.say.endswith(Q_NAME) and d.note is None          # ONE question, spoken by code
    assert not verify_flow.active(state) and state.bstep == "name" and state.intake == {"new": True}


def test_emergency_interrupts_verification(call, state):
    call.hear("Sarah Johnson")
    d = call.hear("I'm having chest pain")
    assert "911" in d.say and state.emergency and not verify_flow.active(state)


def test_rapid_turns_each_produce_exactly_one_reply_and_never_repeat_answered_questions(call, state):
    replies = [call.hear(t).say for t in ("Sarah Johnson", "yes", "March 14 1985")]
    assert all(replies) and len(replies) == 3
    assert replies == [q_name_confirm("Sarah Johnson"), Q_DOB, Q_SSN]
    assert call.says.count(Q_NAME) == 1 and call.says.count(Q_DOB) == 1


# ===== SSN step ==========================================================================================
def test_ssn_digits_are_scrubbed_everywhere_and_never_spoken_back(call, state):
    to_ssn(call)
    scrubbed = gate.scrub_transcript(state, "it's three three four four")
    assert "three" not in scrubbed and "3344" not in scrubbed and gate.WITHHELD in scrubbed
    state.ssn_buffer = ""                                       # (the real turn above consumed its own copy)
    call.hear("three three four four")
    assert state.ssn_buffer == "" and state.verified
    assert not any("3344" in s or "three three" in s for s in call.says + call.notes)


def test_short_and_long_digit_answers_ask_again_without_a_failed_attempt(call, state):
    to_ssn(call)
    assert call.hear("3 3").silent                              # a first short piece is held (more digits may follow)
    assert "only the last four" in call.hear("zero zero zero, one two, three three four four").say
    assert state.failed_attempts == 0 and state.awaiting_ssn and not state.verified
    assert call.says.count(Q_SSN) == 1                          # the SSN question itself was not repeated


def test_refusing_the_digits_offers_staff_at_that_point_only(call, state):
    to_ssn(call)
    assert not any("callback" in s for s in call.says)          # nothing offered while a valid path exists
    d = call.hear("I'd rather not say")
    assert "staff member" in d.say and "callback" in d.say and not verify_flow.active(state)


def test_unrelated_answer_at_ssn_step_is_asked_once_more(call, state):
    to_ssn(call)
    d = call.hear("why do you need that")
    assert "last four digits" in d.say and state.awaiting_ssn and state.unclear_count == 1


# ===== three-strikes handling of unclear answers (not the same as failed attempts) =====================================
def test_three_unusable_answers_in_a_row_offer_staff_but_not_earlier(call, state):
    d1, d2 = call.hear("why do you need that"), call.hear("the weather is nice today")
    assert not any("callback" in s.lower() for s in (d1.say, d2.say))       # not premature
    d3 = call.hear("what what what what")
    assert d3.say == STAFF_OFFER_SAY and not verify_flow.active(state)


# ===== failed attempts =======================================================================================
def attempt(call, name, dob, digits):
    call.hear(name)
    call.hear("yes, that's right")                               # confirm the name read-back
    call.hear(dob)
    return call.hear(digits)


def test_three_failed_attempts_lock_and_offer_staff_even_when_wording_changes(call, state):
    d = attempt(call, "Sarah Johnson", "March 14 1985", "3 3 4 5")
    assert d.say == RETRY_SAY and state.failed_attempts == 1 and state.vstep == "name"
    d = attempt(call, "this is Sarah Johnson", "the fifteenth of March nineteen eighty five", "three three four four")
    assert d.say == RETRY_SAY and state.failed_attempts == 2          # new wording does NOT reset the count
    d = attempt(call, "um, my name is Sarah Johnson", "03/14/1985", "1 1 1 1")
    assert d.say == LOCKED_SAY and state.failed_attempts == 3 and state.locked
    assert "staff member" in d.say and "callback" in d.say
    assert not any(s.count("?") > 1 for s in call.says)
    with pytest.raises(Exception) as e:                                # no more verification on this call
        verify_flow.start(state)
    assert "locked" in str(e.value)
    assert tools.get_balance_due(state).startswith("NOT ALLOWED (locked)")


def test_failure_message_never_says_which_field_was_wrong(call, state):
    outcomes = {
        attempt(call, "Sarah Smith", "March 14 1985", "3344").say,         # wrong name
    }
    call2 = Call(state.__class__(db_path=state.db_path, clock=state.clock))
    verify_flow.start(call2.state)
    outcomes.add(attempt(call2, "Sarah Johnson", "March 15 1985", "3344").say)   # wrong DOB
    call3 = Call(state.__class__(db_path=state.db_path, clock=state.clock))
    verify_flow.start(call3.state)
    outcomes.add(attempt(call3, "Sarah Johnson", "March 14 1985", "9999").say)   # wrong digits
    assert outcomes == {RETRY_SAY}


def test_third_attempt_can_still_succeed(call, state):
    attempt(call, "Sarah Johnson", "March 14 1985", "0 0 0 0")
    attempt(call, "Sarah Johnson", "March 14 1985", "1 1 1 1")
    d = attempt(call, "Sarah Johnson", "March 14 1985", "3 3 4 4")
    assert "VERIFIED" in d.note.upper() and state.verified and state.failed_attempts == 2


def test_each_retry_starts_over_at_the_name_question_once(call, state):
    attempt(call, "Sarah Johnson", "March 14 1985", "0 0 0 0")
    assert call.says[-1] == RETRY_SAY and RETRY_SAY.endswith(Q_NAME)
    assert call.says.count(Q_SSN) == 1                         # SSN question not repeated before DOB is re-given


# ===== no duplicate calls / state corruption ======================================================================
def test_duplicate_start_calls_change_nothing(call, state):
    to_dob(call)
    snapshot = (state.vstep, state.pending_name, state.failed_attempts, state.unclear_count)
    for _ in range(3):
        r = verify_flow.start(state, "Someone Else")
        assert r.silent and r.say is None
    assert (state.vstep, state.pending_name, state.failed_attempts, state.unclear_count) == snapshot
    assert "Say nothing" in tools.start_verification(state, "x")
    assert call.hear("March 14 1985").say == Q_SSN                # flow continues from where it was


def test_start_when_already_verified_says_nothing_new(state):
    call = Call(state)
    verify_flow.start(state)
    attempt(call, "Sarah Johnson", "March 14 1985", "3344")
    r = verify_flow.start(state)
    assert r.say is None and not r.silent and "already verified" in r.text.lower()


def test_audit_records_one_start_and_no_phi(call, state, conn):
    attempt(call, "Sarah Johnson", "March 14 1985", "3344")
    rows = conn.execute("SELECT action, outcome, detail FROM audit_logs").fetchall()
    assert [r["outcome"] for r in rows if r["action"] == "start_verification"] == ["started"]
    text = " ".join(" ".join(str(v) for v in r) for r in rows)
    for banned in ("3344", "1985", "03-14", "Sarah", "Johnson"):
        assert banned not in text


# ===== no patient data before verification ===================================================================================
def test_no_patient_data_during_or_before_verification(call, state):
    def refused():
        for out in (tools.get_next_appointment(state), tools.get_balance_due(state), tools.get_lab_status(state),
                    tools.prepare_refill(state, "lisinopril")):
            assert out.startswith("NOT ALLOWED (not_verified)"), out
            assert "45" not in out and "Lisinopril" not in out
    refused()
    for utterance in ("Sarah Johnson", "yes", "March 14 1985"):
        call.hear(utterance)
        refused()
    for s in call.says:                                         # scripted prompts contain no records
        for secret in ("45.00", "Lisinopril", "Ready", "CVS", "December", "Patel"):
            assert secret not in s
    call.hear("3 3 4 4")
    assert "$45.00" in tools.get_balance_due(state)


def test_clinic_closing_mid_verification_is_handled_without_error(call, state):
    to_dob(call)
    state.clock.set(CLOSED_EVENING)
    d = call.hear("March 14 1985")
    assert d.say == CLOSED_SAY and not verify_flow.active(state) and not state.verified


def test_verification_cannot_start_when_closed_or_during_emergency(make_state):
    st = make_state(CLOSED_EVENING)
    with pytest.raises(Exception, match="closed"):
        verify_flow.start(st)
    st2 = make_state()
    assess_turn(st2, "I think I'm having a heart attack")
    with pytest.raises(Exception, match="emergency"):
        verify_flow.start(st2)


# ===== duplicate workflow actions after a lost reply (e.g. TTS failure) ====================================================
def test_repeating_yes_after_a_silent_reply_never_saves_twice(state, conn):
    c = Call(state)
    verify_flow.start(state)
    attempt(c, "Sarah Johnson", "March 14 1985", "3344")
    tools.prepare_refill(state, "metformin")
    assert "doctor for approval" in c.hear("yes").say                         # saved; but suppose the spoken reply was lost
    again = c.hear("yes?? hello? are you there?")                # the caller repeats themselves
    assert "Saved" not in (again.note or "")
    # the LLM, confused, stages the same refill again and the caller says yes again
    tools.prepare_refill(state, "metformin")
    d = c.hear("yes")
    assert "NOT ALLOWED (already_saved)" in d.note and "Do not save it again" in d.note
    assert conn.execute("SELECT COUNT(*) FROM requests WHERE request_type='refill'").fetchone()[0] == 1


def test_repeated_callback_confirmation_never_duplicates(make_state, conn):
    st = make_state(CLOSED_EVENING)
    c = Call(st)
    tools.prepare_callback_message(st, "Jane Doe", "555-111-2222", "billing question")
    assert "Saved" in c.hear("yes").note
    tools.prepare_callback_message(st, "Jane Doe", "555-111-2222", "billing question")
    assert "already_saved" in c.hear("yes").note
    assert conn.execute("SELECT COUNT(*) FROM callback_messages").fetchone()[0] == 1


# ===== the real agent tool and hook ==========================================================================
class _Stub:
    def __init__(self):
        self.said: list[str] = []

    def say(self, text, **kw):
        self.said.append(text)


@pytest.mark.asyncio
async def test_agent_tool_speaks_the_first_question_stops_the_llm_and_ignores_duplicates(state, monkeypatch):
    import agent
    from livekit.agents import StopResponse
    stub = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    r = agent.Receptionist(state)
    tool = next(t for t in r.tools if t.id == "start_verification")
    with pytest.raises(StopResponse):                 # LLM must not add a second, overlapping prompt
        await tool()
    assert stub.said == [Q_NAME] and state.vstep == "name"
    for _ in range(3):                                # duplicate tool calls: silent no-ops
        with pytest.raises(StopResponse):
            await tool()
    assert stub.said == [Q_NAME] and state.vstep == "name"


@pytest.mark.asyncio
async def test_agent_tool_with_a_name_asks_only_for_the_date(state, monkeypatch):
    import agent
    from livekit.agents import StopResponse
    stub = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    tool = next(t for t in agent.Receptionist(state).tools if t.id == "start_verification")
    with pytest.raises(StopResponse):
        await tool(full_name="Sarah Johnson")
    assert stub.said == [q_name_confirm("Sarah Johnson")]


@pytest.mark.asyncio
async def test_agent_tool_reports_closed_or_locked_as_text_for_the_llm(make_state, monkeypatch):
    import agent
    stub = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    tool = next(t for t in agent.Receptionist(make_state(CLOSED_EVENING)).tools if t.id == "start_verification")
    out = await tool()
    assert out.startswith("NOT ALLOWED (closed)") and stub.said == []


@pytest.mark.asyncio
async def test_turn_hook_speaks_each_scripted_reply_once_and_skips_the_llm(state, monkeypatch):
    import agent
    from livekit.agents import ChatContext, ChatMessage, StopResponse
    stub = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    r = agent.Receptionist(state)
    verify_flow.start(state)
    for utterance, expected in [("Sarah Johnson", q_name_confirm("Sarah Johnson")), ("yes", Q_DOB),
                                ("March 14th, 1985", Q_SSN)]:
        with pytest.raises(StopResponse):
            await r.on_user_turn_completed(ChatContext(), ChatMessage(role="user", content=[utterance]))
        assert stub.said[-1] == expected
    assert len(stub.said) == 3                       # exactly one spoken prompt per caller answer


def test_typed_text_input_is_disabled_because_it_would_bypass_the_gates():
    import agent
    assert agent.ROOM_OPTIONS.get_text_input_options() is None
    assert "room_options=ROOM_OPTIONS" in open(agent.__file__, encoding="utf8").read()
