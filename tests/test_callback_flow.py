"""Email dictation (real-call regressions), the scripted callback conversation, and the staff-or-callback choice.
Synthetic callers only; no network."""

from __future__ import annotations

import pytest

from clinic import booking_flow, callback_flow, gate, speech, tools
from clinic.turn import assess_turn
from clinic.verify_flow import LOCKED_SAY, STAFF_OFFER_SAY, WAIT_SAY
from conftest import CLOSED_EVENING, CLOSED_SATURDAY


class Call:
    def __init__(self, state):
        self.state, self.says = state, []

    def hear(self, text):
        d = assess_turn(self.state, gate.scrub_transcript(self.state, text))
        if d.say:
            self.says.append(d.say)
        return d


def one_question(say: str) -> bool:
    return say.count("?") == 1


# ===== email dictation: what real speech-to-text produced ===================================================
@pytest.mark.parametrize("said,expected", [
    ("jane at gmail.com", "jane@gmail.com"),                              # STT writes the real dot (was rejected)
    ("jane at the rate gmail.com", "jane@gmail.com"),                     # "at the rate" = @ (Indian English)
    ("jane at the rate of gmail dot com", "jane@gmail.com"),
    ("jane dot doe at example dot com", "jane.doe@example.com"),
    ("my email is jane.doe@example.com", "jane.doe@example.com"),
    ("j a n e nine at g mail dot com", "jane9@gmail.com"),
    ("J-A-N-E at yahoo com", "jane@yahoo.com"),
    ("jane at gmail", "jane@gmail.com"),
    ("jane doe nine at outlook dot com", "janedoe9@outlook.com"),
])
def test_email_phrasings_that_real_callers_use(said, expected):
    assert speech.email_from_speech(said) == expected


@pytest.mark.parametrize("said", ["no idea what that is", "hello there how are you", "at", "dot com"])
def test_non_emails_are_not_accepted(said):
    assert speech.email_from_speech(said) is None


def test_halves_are_reported_separately():
    assert speech.email_parts("J-A-N-E.") == ("jane", "")
    assert speech.email_parts("at gmail.com") == ("", "gmail.com")
    assert speech.email_parts("gmail.com") == ("", "gmail.com")


def to_email(call):
    call.state.__class__  # (readability)
    booking_flow.start(call.state, "yes")
    for t in ("My full name is Jane Doe.", "yes", "April 9 1992", "7328735133"):
        call.hear(t)
    assert call.state.bstep == "email"


@pytest.fixture
def call(state):
    return Call(state)


def test_the_exact_failure_from_the_voice_test_now_works(call, state):
    to_email(call)
    d = call.hear("jane at gmail.com")                                   # previously: "Sorry, I didn't catch that email"
    assert d.say == booking_flow.QUESTIONS["insurance"] and state.intake["email"] == "jane@gmail.com"


def test_email_said_in_two_turns_is_put_together(call, state):
    to_email(call)
    assert "after the at sign" in call.hear("J-A-N-E").say and state.intake["email_local"] == "jane"
    d = call.hear("at gmail.com")
    assert d.say == booking_flow.QUESTIONS["insurance"] and state.intake["email"] == "jane@gmail.com"
    assert "email_local" not in state.intake


def test_provider_first_then_name_is_also_put_together(call, state):
    to_email(call)
    assert "before the at sign" in call.hear("gmail.com").say
    call.hear("j a n e")
    assert state.intake["email"] == "jane@gmail.com"


def test_several_misheard_emails_do_not_throw_the_booking_away(call, state):
    to_email(call)
    for _ in range(4):
        d = call.hear("what is that what is that")
        assert d.say != STAFF_OFFER_SAY and booking_flow.active(state), d.say
    assert "two parts" in call.says[-1] or "email" in call.says[-1].lower()
    assert call.hear("what is that what is that").say == STAFF_OFFER_SAY       # only the fifth miss gives up
    assert state.intake == {} and not booking_flow.active(state)


def test_the_second_miss_asks_for_the_spelled_first_half(call, state):
    to_email(call)
    assert "Could you say it again" in call.hear("blah blah blah blah").say
    assert "spell the part before the at sign" in call.hear("blah blah blah blah").say


# ===== callback conversation ===============================================================================
def test_after_hours_callback_is_one_question_at_a_time_and_saved(make_state, conn):
    st = make_state(CLOSED_EVENING)
    c = Call(st)
    r = callback_flow.start(st)
    assert r.say == callback_flow.QUESTIONS["name"] and callback_flow.active(st)
    assert c.hear("Jane Doe").say == callback_flow.QUESTIONS["phone"]
    assert c.hear("five five five, one one one, two two two two").say == callback_flow.QUESTIONS["reason"]
    d = c.hear("a billing question")
    assert d.say == "Let me confirm. Jane Doe, callback number 555-111-2222, regarding billing question. Is that correct?"
    assert conn.execute("SELECT COUNT(*) FROM callback_messages").fetchone()[0] == 0          # nothing saved yet
    d = c.hear("Yes, that's correct")
    assert d.say == ("Thank you, Jane. Your message has been recorded, and our team will get back to you on the next "
                     "business day. If this is an emergency, please hang up and call 911. Have a good night.")
    row = conn.execute("SELECT * FROM callback_messages").fetchone()
    assert row["caller_name"] == "Jane Doe" and row["callback_number"] == "555-111-2222" and row["followup_date"] == "2026-12-01"
    req = conn.execute("SELECT * FROM requests").fetchone()
    assert (req["patient_id"], req["request_type"], req["status"]) == ("UNVERIFIED", "callback", "open")
    assert not callback_flow.active(st)
    assert d.say.count("?") == 0 and all(one_question(s) for s in c.says if "?" in s)


def test_callback_during_open_hours_has_a_daytime_closing_and_is_not_an_emergency_script(state, conn):
    c = Call(state)
    callback_flow.start(state, "Jane")
    c.hear("555 111 2222")
    c.hear("a question for the doctor")
    d = c.hear("yes")
    assert "next business day" in d.say and "911" not in d.say and d.say.endswith("Is there anything else I can help you with?")


def test_weekend_day_closing_says_good_day_and_follows_up_monday(make_state, conn):
    st = make_state(CLOSED_SATURDAY)
    c = Call(st)
    callback_flow.start(st, "Jane Doe")
    c.hear("5551112222")
    c.hear("billing")
    assert c.hear("yes").say.endswith("Have a good day.")
    assert conn.execute("SELECT followup_date FROM callback_messages").fetchone()[0] == "2026-11-30"


def test_name_hint_from_the_llm_skips_the_name_question(make_state):
    st = make_state(CLOSED_EVENING)
    assert callback_flow.start(st, "Jane Doe").say == callback_flow.QUESTIONS["phone"]


def test_verified_patient_is_not_asked_for_their_name_again(state):
    state.verified_patient_id = "P001"
    assert callback_flow.start(state).say == callback_flow.QUESTIONS["phone"]


def test_callback_read_back_can_be_corrected(state, conn):
    c = Call(state)
    callback_flow.start(state, "Jane Doe")
    c.hear("5551112222")
    c.hear("billing")
    assert c.hear("No").say.startswith("No problem. What would you like to change")
    assert c.hear("the number").say == callback_flow.QUESTIONS["phone"]
    d = c.hear("555 333 4444")
    assert "555-333-4444" in d.say and "Let me confirm" in d.say
    c.hear("yes")
    assert conn.execute("SELECT callback_number FROM callback_messages").fetchone()[0] == "555-333-4444"


@pytest.mark.parametrize("filler", ["hold on a second", "sorry, one moment"])
def test_waits_do_not_advance_a_callback(state, filler):
    c = Call(state)
    callback_flow.start(state)
    assert c.hear(filler).say == WAIT_SAY and state.cstep == "name"


def test_duplicate_start_callback_does_nothing(state):
    callback_flow.start(state, "Jane")
    before = (state.cstep, dict(state.callback))
    for _ in range(3):
        assert callback_flow.start(state).silent
    assert (state.cstep, state.callback) == before
    assert "Say nothing" in tools.start_callback(state)


def test_callback_can_be_stopped_and_emergencies_win(state):
    c = Call(state)
    callback_flow.start(state, "Jane")
    d = c.hear("never mind")
    assert d.say is None and "cancelled" in d.note and not callback_flow.active(state)
    callback_flow.start(state, "Jane")
    d = c.hear("I am having chest pain")
    assert "911" in d.say and state.emergency and not callback_flow.active(state)


def test_the_llm_is_not_called_during_a_callback(state):
    c = Call(state)
    callback_flow.start(state)
    for text in ("Jane Doe", "5551112222", "billing question", "yes"):
        assert c.hear(text).say is not None, text


# ===== "staff member or callback?" is answered in code ============================================================
def offer_after_three_unclear_answers(state):
    c = Call(state)
    booking_flow.start(state, "yes")
    for _ in range(3):
        c.hear("what what what")
    assert state.choice == "staff_or_callback"
    return c


def test_callback_answer_after_the_offer_starts_the_scripted_callback(state, conn):
    """Replay of the real call: 'Call back please' used to make the LLM ask three questions at once."""
    c = offer_after_three_unclear_answers(state)
    d = c.hear("Call back please")
    assert d.say == callback_flow.QUESTIONS["name"] and one_question(d.say) and callback_flow.active(state)
    c.hear("Jane Doe")
    c.hear("5551112222")
    c.hear("I could not finish booking")
    d = c.hear("yes")
    assert "recorded" in d.say and conn.execute("SELECT COUNT(*) FROM callback_messages").fetchone()[0] == 1


def test_staff_answer_creates_a_simulated_transfer_request(state, conn):
    c = offer_after_three_unclear_answers(state)
    d = c.hear("a staff member please")
    assert "asked a staff member" in d.say and state.choice is None
    r = conn.execute("SELECT * FROM requests WHERE request_type='handoff'").fetchone()
    assert r["status"] == "transfer_simulated"


def test_unclear_answer_to_the_offer_is_asked_once_then_handed_back(state):
    c = offer_after_three_unclear_answers(state)
    assert c.hear("why is that").say == callback_flow.STAFF_OR_CALLBACK_REASK
    assert c.hear("banana").say is None and state.choice is None            # no nagging


def test_okay_to_the_offer_is_waited_on_not_acted_on(state):
    c = offer_after_three_unclear_answers(state)
    assert c.hear("hold on").say == WAIT_SAY and state.choice == "staff_or_callback"


def test_after_a_lockout_the_same_choice_works(state):
    c = Call(state)
    from clinic import verify_flow
    verify_flow.start(state)
    for _ in range(3):
        for t in ("Jane Doe", "yes", "February 17 1973", "1 1 1 1"):
            d = c.hear(t)
    assert d.say == LOCKED_SAY and state.choice == "staff_or_callback"
    # the name was already confirmed during verification, so the callback goes straight to the number
    assert c.hear("leave a message").say == callback_flow.QUESTIONS["phone"]
    assert state.callback["name"] == "Jane Doe"


def test_no_openings_offer_accepts_yes_as_a_callback(state, conn):
    conn.execute("DELETE FROM available_slots")
    c = Call(state)
    booking_flow.start(state, "yes")
    for t in ("Jane Doe", "yes", "April 9 1992", "5551112222", "jane@example.com", "self-pay", "checkup", "no preference"):
        d = c.hear(t)
    assert d.say == booking_flow.NO_SLOTS_SAY and state.choice == "callback_yes"
    assert c.hear("yes please").say == callback_flow.QUESTIONS["name"]


def test_no_openings_offer_accepts_no(state, conn):
    conn.execute("DELETE FROM available_slots")
    c = Call(state)
    booking_flow.start(state, "yes")
    for t in ("Jane Doe", "yes", "April 9 1992", "5551112222", "jane@example.com", "self-pay", "checkup", "no preference"):
        c.hear(t)
    assert "anything else" in c.hear("no thanks").say and state.choice is None


# ===== the agent tool ======================================================================================================
class _Stub:
    def __init__(self):
        self.said = []

    def say(self, text, **kw):
        self.said.append(text)


@pytest.mark.asyncio
async def test_start_callback_tool_speaks_once_and_stops_the_llm(make_state, monkeypatch):
    import agent
    from livekit.agents import StopResponse
    st = make_state(CLOSED_EVENING)
    stub = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    tool = next(t for t in agent.Receptionist(st).tools if t.id == "start_callback")
    with pytest.raises(StopResponse):
        await tool()
    assert stub.said == [callback_flow.QUESTIONS["name"]]
    for _ in range(2):
        with pytest.raises(StopResponse):
            await tool()
    assert len(stub.said) == 1
    st2 = make_state(CLOSED_EVENING)
    stub2 = _Stub()
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub2))
    tool2 = next(t for t in agent.Receptionist(st2).tools if t.id == "start_callback")
    with pytest.raises(StopResponse):
        await tool2(caller_name="Jane Doe")
    assert stub2.said == [callback_flow.QUESTIONS["phone"]]
