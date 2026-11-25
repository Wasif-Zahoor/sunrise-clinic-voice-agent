"""The booking conversation: one question per turn, name read-back, nothing asked twice, the LLM never
called during intake, real slots, explicit confirmation, atomic save. Synthetic callers only; no network."""

from __future__ import annotations

import pytest

from clinic import booking_flow, db, gate, tools, verify_flow
from clinic import services as svc
from clinic.turn import assess_turn
from clinic.verify_flow import Q_NAME, Q_NAME_SPELL, RETRY_SAY, STAFF_OFFER_SAY, WAIT_SAY, q_name_confirm
from conftest import CLOSED_EVENING

Q_FIRST = booking_flow.Q_FIRST_VISIT


class Call:
    def __init__(self, state):
        self.state, self.says, self.turns = state, [], []

    def start(self, first_visit=""):
        r = booking_flow.start(self.state, first_visit)
        if r.say:
            self.says.append(r.say)
        return r

    def hear(self, text):
        d = assess_turn(self.state, gate.scrub_transcript(self.state, text))
        self.turns.append((text, d))
        if d.say:
            self.says.append(d.say)
        return d


@pytest.fixture
def call(state):
    return Call(state)


def one_question(say: str) -> bool:
    return say.count("?") == 1


def new_patient_to_reason(call):
    call.start("yes")
    for text in ("My full name is John Smith.", "yes", "April ninth, nineteen ninety-two",
                 "five five five, one two three, four five six seven", "john dot smith at example dot com",
                 "Blue Shield", "B S one two three four five"):
        call.hear(text)


def new_patient_to_slots(call, preference="Tuesday"):
    new_patient_to_reason(call)
    call.hear("an annual check-up")
    return call.hear(preference)


# ===== regression: the real conversation that went wrong ========================================================
def test_replay_of_the_failed_voice_test_with_a_synthetic_caller(call, state):
    """'old patient' -> wrong details -> 'new patient': every reply is ONE question, the name is read back,
    spelled letters work, and nothing is dumped at once."""
    r = call.start()
    assert r.say == Q_FIRST
    d = call.hear("No, I'm an old patient.")
    assert d.say.endswith(Q_NAME) and state.after_verify == "booking" and verify_flow.active(state)
    d = call.hear("My full name is Jane Doe.")                      # 'my FULL name is' used to be rejected
    assert d.say == q_name_confirm("Jane Doe")                      # and the name is read back
    d = call.hear("Yeah, that's right.")
    assert d.say == verify_flow.Q_DOB
    d = call.hear("Yeah, it is 17th Feb 1973.")
    assert d.say == verify_flow.Q_SSN
    d = call.hear("four five six seven")
    assert d.say == RETRY_SAY and state.failed_attempts == 1        # not an existing patient: generic failure
    d = call.hear("I'm sorry, I forgot. I'm a new patient.")
    assert d.say.endswith(Q_NAME) and d.note is None and state.bstep == "name"     # handed to the scripted flow
    d = call.hear("I already gave you")
    assert d.say == f"I'm sorry about that. {Q_NAME}"               # not a list of seven questions
    d = call.hear("J-A-N-E")
    assert d.say == "Thanks, Jane. And your last name?"
    d = call.hear("D-O-E")
    assert d.say == q_name_confirm("Jane Doe")
    for say in call.says:
        assert one_question(say), say                               # never several questions at once
    assert not any(n for _, d in call.turns for n in [d.note] if n)  # the LLM was never involved


# ===== full new-patient intake ======================================================================================
def test_new_patient_intake_asks_one_question_at_a_time_in_order(call, state):
    r = call.start()
    assert r.say == Q_FIRST and state.bstep == "first_visit"
    expected = [
        ("Yes, it's my first visit.", "Welcome to Sunrise Family Clinic! " + Q_NAME),
        ("My full name is John Smith.", q_name_confirm("John Smith")),
        ("yes", booking_flow.QUESTIONS["dob"]),
        ("April ninth, nineteen ninety-two", booking_flow.QUESTIONS["phone"]),
        ("five five five, one two three, four five six seven", booking_flow.QUESTIONS["email"]),
        ("john dot smith at example dot com", booking_flow.QUESTIONS["insurance"]),
        ("Blue Shield", booking_flow.QUESTIONS["member"]),
        ("B S one two three four five", booking_flow.QUESTIONS["reason"]),
        ("an annual check-up", booking_flow.QUESTIONS["preference"]),
    ]
    for said, asked in expected:
        d = call.hear(said)
        assert d.say == asked and d.note is None, (said, d.say)
    d = call.hear("Tuesday")
    assert d.say.count("Dr. Patel") == 3 and "9:30 AM" in d.say and "11:00 AM" in d.say and "3:00 PM" in d.say
    assert d.say.endswith("Which would you like?")
    for say in call.says:
        assert one_question(say), say
        assert "YYYY" not in say and "callback" not in say.lower()
    assert state.intake["name"] == "John Smith" and state.intake["phone"] == "555-123-4567"
    assert state.intake["email"] == "john.smith@example.com" and state.intake["member"] == "BS12345"


def test_name_is_read_back_then_details_read_back_and_saved_after_yes(call, state, conn):
    new_patient_to_slots(call)
    d = call.hear("11 AM")
    assert d.say.startswith("Let me confirm. John Smith, Tuesday, December 1st at 11:00 AM, with Dr. Patel, for annual check-up.")
    assert "Born April 9th, 1992" in d.say and "Phone 555-123-4567" in d.say
    assert "j o h n dot s m i t h at e x a m p l e dot c o m" in d.say        # email spelled back
    assert d.say.endswith("Is all of that correct?") and state.pending is not None
    assert conn.execute("SELECT COUNT(*) FROM patients WHERE is_new=1").fetchone()[0] == 0        # nothing saved yet

    d = call.hear("Yes, that's all correct")
    assert d.say.startswith("You're all set for Tuesday, December 1st at 11:00 AM with Dr. Patel.")
    assert "arrive 15 minutes early" in d.say and "photo ID and your insurance card" in d.say
    assert d.say.endswith("Is there anything else I can help you with?")
    assert state.bstep == "idle" and state.pending is None

    p = conn.execute("SELECT * FROM patients WHERE is_new=1").fetchone()
    assert p["patient_id"] == "N001" and p["first_name"] == "John" and p["last_name"] == "Smith"
    assert p["dob"] == "1992-04-09" and p["phone"] == "555-123-4567" and p["email"] == "john.smith@example.com"
    assert p["insurance_provider"] == "Blue Shield" and p["member_id"] == "BS12345" and p["self_pay"] == 0
    a = conn.execute("SELECT * FROM appointments WHERE patient_id='N001'").fetchone()
    assert a["doctor"] == "Dr. Patel" and a["starts_at"] == "2026-12-01T11:00" and a["reason"] == "annual check-up"
    assert conn.execute("SELECT COUNT(*) FROM available_slots").fetchone()[0] == 6            # the slot is gone
    assert conn.execute("SELECT 1 FROM available_slots WHERE starts_at='2026-12-01T11:00'").fetchone() is None
    r = conn.execute("SELECT * FROM requests WHERE patient_id='N001'").fetchone()
    assert r["request_type"] == "booking" and r["status"] == "confirmed" and r["created_at"] and r["summary"]
    mail = conn.execute("SELECT * FROM email_outbox").fetchone()
    assert mail["to_email"] == "john.smith@example.com" and "15 minutes early" in mail["body"]


def test_after_the_booking_the_llm_takes_over_again(call, state):
    new_patient_to_slots(call)
    call.hear("11 AM")
    call.hear("yes")
    d = call.hear("No, that's everything. Goodbye.")
    assert d.say is None                                           # normal LLM conversation resumes


def test_self_pay_skips_the_provider_and_member_questions(call, state, conn):
    call.start("yes")
    for text in ("John Smith", "yes", "April 9 1992", "555 123 4567", "john.smith@example.com"):
        call.hear(text)
    d = call.hear("I'll self-pay")
    assert d.say == booking_flow.QUESTIONS["reason"]
    call.hear("annual check-up")
    call.hear("no preference")
    d = call.hear("the second one")
    assert "Let me confirm" in d.say
    call.hear("yes")
    p = conn.execute("SELECT * FROM patients WHERE patient_id='N001'").fetchone()
    assert p["self_pay"] == 1 and p["member_id"] is None


# ===== name read-back and corrections ===========================================================================================
def test_wrong_name_read_back_can_be_corrected_by_spelling(call, state):
    call.start("yes")
    call.hear("Jon Smyth")
    d = call.hear("No")
    assert d.say == Q_NAME_SPELL
    assert call.hear("J-O-H-N").say == "Thanks, John. And your last name?"
    assert call.hear("S-M-I-T-H").say == q_name_confirm("John Smith")
    assert call.hear("yes").say == booking_flow.QUESTIONS["dob"]


def test_correction_in_one_breath(call, state):
    call.start("yes")
    call.hear("Jon Smyth")
    assert call.hear("No, it's John Smith").say == q_name_confirm("John Smith")


# ===== existing patients ======================================================================================================
def existing(call, who=("Michael Brown", "November 8 1972", "1 0 2 0")):
    call.start()
    call.hear("No, I'm an old patient.")
    call.hear(who[0])
    call.hear("yes")
    call.hear(who[1])
    return call.hear(who[2])


def test_old_patient_is_verified_then_booking_continues_without_repeating_details(call, state, conn):
    d = existing(call)
    assert d.say == "Thank you, you're verified. And your email address?"        # no 'Thank you' twice, name/DOB not re-asked
    assert state.verified and state.bstep == "email" and not verify_flow.active(state)
    assert call.hear("mike at example dot com").say == booking_flow.QUESTIONS["reason"]
    call.hear("blood pressure follow-up")
    d = call.hear("Dr. Lee")
    assert d.say.count("Dr. Lee") == 2 and "Wednesday, December 2nd at 10:00 AM" in d.say
    d = call.hear("the 1:30")
    assert "Michael Brown" in d.say and "1:30 PM" in d.say and "m i k e at e x a m p l e dot c o m" in d.say
    assert "Born" not in d.say                                                    # details on file are not recited
    d = call.hear("yes please")
    assert "You're all set for Wednesday, December 2nd at 1:30 PM with Dr. Lee" in d.say
    a = conn.execute("SELECT * FROM appointments WHERE patient_id='P002'").fetchone()
    assert a["doctor"] == "Dr. Lee" and a["starts_at"] == "2026-12-02T13:30"
    assert conn.execute("SELECT COUNT(*) FROM patients WHERE is_new=1").fetchone()[0] == 0        # no duplicate patient


def test_already_verified_caller_skips_straight_to_what_is_missing(call, state):
    existing(call)
    other = Call(state.__class__(db_path=state.db_path, clock=state.clock))
    other.state.verified_patient_id = "P001"                  # Sarah has no email on file either
    r = other.start()
    assert r.say == booking_flow.QUESTIONS["email"] and other.state.bstep == "email"


def test_failed_verification_does_not_start_a_booking(call, state):
    call.start("no")
    d = call.hear("Jane Doe")
    d = call.hear("yes")
    d = call.hear("February 17 1973")
    d = call.hear("1 1 1 1")
    assert d.say == RETRY_SAY and not state.verified and state.bstep == "idle" and state.after_verify == "booking"


# ===== changing details ==========================================================================================================
def test_saying_no_to_the_read_back_lets_the_caller_fix_one_detail(call, state, conn):
    new_patient_to_slots(call)
    call.hear("11 AM")
    d = call.hear("No, the email is wrong")
    assert d.say == "No problem. " + booking_flow.QUESTIONS["email"] and state.pending is None
    d = call.hear("john dot smith at gmail dot com")
    assert d.say.startswith("Let me confirm.") and "j o h n dot s m i t h at gmail dot com" in d.say
    call.hear("yes")
    assert conn.execute("SELECT email FROM patients WHERE patient_id='N001'").fetchone()[0] == "john.smith@gmail.com"


def test_changing_the_time_goes_back_to_the_openings(call, state, conn):
    new_patient_to_slots(call)
    call.hear("11 AM")
    d = call.hear("No")
    assert d.say.startswith("What would you like to change?")
    d = call.hear("the time")
    assert d.say.startswith("No problem. ") and booking_flow.QUESTIONS["preference"] in d.say
    call.hear("Dr. Lee")
    d = call.hear("the first one")
    assert "Dr. Lee" in d.say and "Let me confirm" in d.say
    call.hear("yes")
    assert conn.execute("SELECT doctor FROM appointments WHERE patient_id='N001'").fetchone()[0] == "Dr. Lee"


def test_none_of_those_asks_for_a_new_preference(call, state):
    new_patient_to_slots(call)
    d = call.hear("none of those, something else")
    assert booking_flow.QUESTIONS["preference"] in d.say


# ===== unclear answers, waits, interruptions =====================================================================================
@pytest.mark.parametrize("filler", ["Sorry", "hold on a second", "one moment please", "wait"])
def test_waits_never_advance_or_repeat_the_question(call, state, filler):
    new_patient_to_reason(call)
    before = (state.bstep, dict(state.intake), state.unclear_count)
    assert call.hear(filler).say == WAIT_SAY
    assert (state.bstep, state.intake, state.unclear_count) == before


@pytest.mark.parametrize("step_text,bad,fragment", [
    ("phone", "five five five one two three", "all ten digits"),
    ("email", "no idea what that is", "email"),
    ("reason", "x", "few words"),
    ("dob", "February 30 1990", "date"),
])
def test_unusable_answers_are_asked_again_briefly(call, state, step_text, bad, fragment):
    call.start("yes")
    path = {"phone": ["John Smith", "yes", "April 9 1992"], "email": ["John Smith", "yes", "April 9 1992", "555 123 4567"],
            "reason": ["John Smith", "yes", "April 9 1992", "555 123 4567", "j@example.com", "self-pay"],
            "dob": ["John Smith", "yes"]}[step_text]
    for t in path:
        call.hear(t)
    assert state.bstep == step_text
    d = call.hear(bad)
    assert fragment in d.say and state.bstep == step_text and state.unclear_count == 1
    assert "callback" not in d.say.lower()                              # not premature


def test_three_unusable_answers_offer_staff_and_leave_the_flow(call, state):
    call.start("yes")
    for _ in range(2):
        assert "callback" not in call.hear("what what what").say.lower()
    d = call.hear("what what what")
    assert d.say == STAFF_OFFER_SAY and not booking_flow.active(state)


def test_ambiguous_birth_date_is_clarified(call, state):
    call.start("yes")
    call.hear("John Smith")
    call.hear("yes")
    d = call.hear("04/09/1992")
    assert "April 9th, 1992" in d.say and "September 4th, 1992" in d.say and state.bstep == "dob_clarify"
    assert call.hear("September fourth").say == booking_flow.QUESTIONS["phone"]
    assert state.intake["dob"] == "1992-09-04"


@pytest.mark.parametrize("text,fragment", [("Can I talk to a real person?", "request_staff_transfer"),
                                           ("Never mind, forget it", "cancelled")])
def test_caller_can_leave_the_booking(call, state, text, fragment):
    new_patient_to_reason(call)
    d = call.hear(text)
    assert d.say is None and fragment in d.note and not booking_flow.active(state) and state.intake == {}


def test_emergency_mid_booking_stops_everything(call, state):
    new_patient_to_reason(call)
    d = call.hear("I'm having chest pain")
    assert "911" in d.say and state.emergency and not booking_flow.active(state)


def test_duplicate_start_booking_changes_nothing(call, state):
    call.start("yes")
    call.hear("John Smith")
    before = (state.bstep, dict(state.intake))
    for _ in range(3):
        r = booking_flow.start(state)
        assert r.silent and r.say is None
    assert (state.bstep, state.intake) == before
    assert "Say nothing" in tools.start_booking(state)


def test_a_slot_taken_by_someone_else_is_re_offered_not_double_booked(call, state, make_state, conn):
    new_patient_to_slots(call)
    call.hear("11 AM")                                              # John is at the read-back for Dec 1 11:00
    rival = make_state()
    svc.find_slots(rival)
    sid = next(i for i, s in rival.offered_slots.items() if s["starts_at"] == "2026-12-01T11:00")
    svc.prepare_booking_new(rival, sid, "Ann Lee", "1980-01-01", "555-222-3333", "ann@example.com", "self-pay", "", "checkup")
    rival.register_user_turn("yes")
    svc.confirm_booking(rival)                                      # the rival takes the slot first
    d = call.hear("yes")
    assert "no longer available" in d.say and "Which would you like?" in d.say and "11:00 AM" not in d.say
    assert conn.execute("SELECT COUNT(*) FROM appointments WHERE starts_at='2026-12-01T11:00'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM patients WHERE patient_id='N002'").fetchone()[0] == 0


def test_new_patient_matching_an_existing_record_is_sent_to_staff_without_saying_why(call, state):
    call.start("yes")
    for t in ("Sarah Johnson", "yes", "March 14 1985", "555 201 3344", "sarah@example.com", "self-pay", "check-up", "no preference"):
        call.hear(t)
    d = call.hear("the first one")
    assert d.say == booking_flow.CANNOT_COMPLETE_SAY and not booking_flow.active(state)
    assert "exist" not in d.say.lower() and "record" not in d.say.lower()


def test_no_openings_offers_a_callback(call, state, conn):
    conn.execute("DELETE FROM available_slots")
    new_patient_to_reason(call)
    call.hear("check-up")
    d = call.hear("no preference")
    assert d.say == booking_flow.NO_SLOTS_SAY and not booking_flow.active(state)


def test_no_preference_match_is_said_honestly(call, state):
    new_patient_to_reason(call)
    call.hear("check-up")
    d = call.hear("Friday afternoon with Dr. Lee")
    assert "I don't have an exact match" in d.say


# ===== guards ================================================================================================================
def test_booking_cannot_start_when_the_clinic_is_closed(make_state):
    st = make_state(CLOSED_EVENING)
    assert tools.start_booking(st).startswith("NOT ALLOWED (closed)")
    assert st.bstep == "idle"


def test_clinic_closing_mid_booking_is_handled_without_error(call, state):
    new_patient_to_slots(call)
    state.clock.set(CLOSED_EVENING)
    d = call.hear("11 AM")
    assert "closed" in d.say.lower() and not booking_flow.active(state)
    assert db.connect(state.db_path).execute("SELECT COUNT(*) FROM appointments WHERE patient_id LIKE 'N%'").fetchone()[0] == 0


def test_the_llm_is_never_called_during_intake(call, state):
    call.start()
    for text in ("Yes it's my first visit", "John Smith", "yes", "April 9 1992", "555 123 4567", "j@example.com",
                 "Blue Shield", "A B C 1 2 3 4", "check-up", "no preference"):
        d = call.hear(text)
        assert d.say is not None, text                              # every turn answered by code
    assert state.bstep == "slot"


# ===== the real agent tool and the LLM safety net ==============================================================================
class _Stub:
    def __init__(self):
        self.said = []

    def say(self, text, **kw):
        self.said.append(text)


@pytest.mark.asyncio
async def test_start_booking_tool_speaks_once_stops_the_llm_and_ignores_duplicates(state, monkeypatch):
    import agent
    from livekit.agents import StopResponse
    stub = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    tool = next(t for t in agent.Receptionist(state).tools if t.id == "start_booking")
    with pytest.raises(StopResponse):
        await tool()
    assert stub.said == [Q_FIRST]
    for _ in range(3):
        with pytest.raises(StopResponse):
            await tool()
    assert stub.said == [Q_FIRST]


@pytest.mark.asyncio
async def test_an_llm_failure_gives_a_short_apology_instead_of_ending_the_call(state, monkeypatch):
    import agent
    from livekit.agents import APIConnectionError, ChatContext

    async def failing(self, chat_ctx, tools_, settings):
        raise APIConnectionError("rate limited")
        yield  # pragma: no cover

    monkeypatch.setattr(agent.Agent.default, "llm_node", staticmethod(failing))
    out = [c async for c in agent.Receptionist(state).llm_node(ChatContext(), [], None)]
    assert len(out) == 1 and "technical issue" in out[0] and "staff member" in out[0] and "callback" in out[0]
    assert state.choice == "staff_or_callback"          # the caller's answer will be handled in code


@pytest.mark.asyncio
async def test_llm_requests_are_kept_small_by_truncating_history(state, monkeypatch):
    import agent
    from livekit.agents import ChatContext
    seen = {}

    async def spy(self, chat_ctx, tools_, settings):
        seen["items"] = len(chat_ctx.items)
        yield "ok"

    monkeypatch.setattr(agent.Agent.default, "llm_node", staticmethod(spy))
    ctx = ChatContext()
    for i in range(40):
        ctx.add_message(role="user" if i % 2 == 0 else "assistant", content=[f"message {i}"])
    [c async for c in agent.Receptionist(state).llm_node(ctx, [], None)]
    assert seen["items"] <= 15 and len(ctx.items) == 40            # the caller's context is untouched
