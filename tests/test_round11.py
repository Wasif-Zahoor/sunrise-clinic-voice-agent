"""Round 11 (fourth live call, 2026-11-25): after a NEW-patient booking, "what is my lab status" restarted verification (name, date of
birth, last four digits) although the caller had just given those details, and a new patient can never pass it (no last four digits
are collected). Also: "book 1 for 10am" booked 1:30 PM, and "Sophie" (speech-to-text for "self-pay") was saved as the insurer. Offline."""

from __future__ import annotations

import pytest

from clinic import booking_flow, intent, speech, tools, transcript, verify_flow
from clinic import services as svc
from conftest import CLOSED_EVENING
from test_booking_flow import Call, new_patient_to_slots
from test_round9 import verify_as


@pytest.fixture
def call(state):
    return Call(state)


def booked_new_patient(call):
    """A new patient books Tuesday 11:00 with Dr. Patel and says yes to the read-back."""
    new_patient_to_slots(call)
    call.hear("11 AM")
    d = call.hear("Yes, that's all correct")
    assert d.say.startswith("You're all set")
    return d


# ---- the call that went wrong -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("asked,expected", [
    ("what is my lab status", "Since this is your first visit, I don't have any lab results on file for you yet."),
    ("can you tell me my due balance", "Since this is your first visit, there is no balance on your account."),
    ("when is my next appointment", "Your next appointment is Tuesday, December 1st at 11:00 AM with Dr. Patel."),
    ("I would like a prescription refill", "I don't have any medications on file for you yet."),
])
def test_a_new_patient_is_answered_from_this_call_and_never_asked_to_verify_again(call, state, conn, asked, expected):
    booked_new_patient(call)
    assert state.new_patient_id == "N001"
    d = call.hear(asked)
    assert expected in d.say and d.say.endswith("Is there anything else I can help you with?")
    assert not verify_flow.active(state) and state.vstep == "idle" and not state.awaiting_ssn
    assert not state.verified and state.failed_attempts == 0 and not state.locked
    assert "Social Security" not in d.say and "full name" not in d.say
    topic = {"lab": "lab", "balance": "balance", "appointment": "appointment", "refill": "refill"}[intent.detect(asked)]
    row = conn.execute("SELECT outcome, patient_id FROM audit_logs WHERE action=? ORDER BY id DESC LIMIT 1",
                       (f"new_patient_{topic}",)).fetchone()
    assert row["outcome"] == "answered" and row["patient_id"] == "N001"


def test_the_replay_of_the_real_wording(call, state):
    booked_new_patient(call)
    assert "lab results" in call.hear("Yes, can you tell me what is my lab status.").say


def test_a_new_patient_is_still_not_verified_so_every_verified_only_tool_refuses_them(call, state):
    booked_new_patient(call)
    with pytest.raises(svc.ServiceError) as e:
        svc.get_lab_status(state)
    assert e.value.code == "not_verified"
    assert tools.speak_balance(state)[0] is None and tools.speak_next_appointment(state)[0] is None


def test_a_caller_who_did_not_register_in_this_call_still_has_to_verify(make_state):
    s = make_state()
    assert intent.route(s, "what is my lab status") == verify_flow.Q_NAME
    assert s.new_patient_id is None


def test_the_new_patient_answer_is_refused_without_a_booking_in_this_call(make_state):
    s = make_state()
    with pytest.raises(svc.ServiceError) as e:
        svc.get_new_patient_info(s, "lab")
    assert e.value.code == "not_verified"


def test_the_new_patient_answer_follows_the_after_hours_and_emergency_rules(call, state, make_state):
    booked_new_patient(call)
    state.emergency = True
    with pytest.raises(svc.ServiceError) as e:
        svc.get_new_patient_info(state, "lab")
    assert e.value.code == "emergency"
    closed = make_state(CLOSED_EVENING)
    closed.new_patient_id = "N001"
    with pytest.raises(svc.ServiceError) as e:
        svc.get_new_patient_info(closed, "lab")
    assert e.value.code == "closed"


def test_a_question_about_someone_else_is_not_answered_from_the_new_patient_record(call, state):
    booked_new_patient(call)
    state.register_user_turn("what does my husband owe")
    state.third_party_turn = state.turn
    assert intent.route(state, "what does my husband owe") is None
    with pytest.raises(svc.ServiceError) as e:
        svc.get_new_patient_info(state, "balance")
    assert e.value.code == "third_party"


def test_a_new_patient_who_then_verifies_as_an_existing_patient_gets_the_normal_answers(call, state):
    booked_new_patient(call)
    verify_as(call)                                           # Sarah Johnson, with her real date of birth and last four digits
    assert state.verified
    say = intent.route(state, "what is my lab status")
    assert say and "first visit" not in say


def test_scheduling_another_appointment_is_not_taken_over_by_the_new_patient_answers(call, state):
    booked_new_patient(call)
    assert intent.route(state, "I would like to schedule another appointment") == booking_flow.Q_FIRST_VISIT


# ---- "book 1 for 10am" ---------------------------------------------------------------------------------------------------------------
SLOTS = [{"slot_id": 1, "doctor": "Dr. Lee", "starts_at": "2026-12-02T10:00"},
         {"slot_id": 2, "doctor": "Dr. Lee", "starts_at": "2026-12-02T13:30"}]


@pytest.mark.parametrize("said,slot", [
    ("Yes, book 1 for 10am.", 1), ("book 1 for 10 am", 1), ("Yes, book one for ten a m", 1), ("the 10am", 1), ("ten am", 1),
    ("10:00 am please", 1), ("book 2 for 1:30pm", 2), ("book 1 for 1:30 pm", 2), ("1:30 pm", 2), ("one thirty", 2),
    ("book one thirty", 2), ("one thirty pm", 2),
    ("book 1", 1), ("book one", 1), ("take two", 2), ("go with number 2", 2), ("the second one", 2), ("the first", 1)])
def test_slot_selection(said, slot):
    assert speech.choose_slot(said, SLOTS) == ("ok", slot), said


def test_a_time_that_was_not_offered_is_not_guessed():
    assert speech.choose_slot("book 1 for 3pm", SLOTS)[0] == "none"
    assert speech.choose_slot("yes", SLOTS) == ("none", None)


def test_the_flow_books_the_time_that_was_said_not_a_stray_digit(call, state):
    new_patient_to_slots(call)                                # Tuesday: Dr. Patel at 9:30, 11:00 and 3:00
    offers = state.intake["offers"]
    d = call.hear("Yes, book 1 for 11am.")
    assert d.say.startswith("Let me confirm.") and "11:00 AM" in d.say, offers


# ---- "Sophie" was saved as the insurance provider ---------------------------------------------------------------------------------------
def to_insurance(call):
    call.start("yes")
    for text in ("John Smith", "yes", "April 9 1992", "555 123 4567", "john.smith@example.com"):
        call.hear(text)


def test_an_unknown_insurer_is_read_back_and_a_known_one_is_not(call, state):
    to_insurance(call)
    d = call.hear("Sophie")
    assert d.say == "I heard Sophie. Is that your insurance provider, or would you like to self-pay?" and state.bstep == "provider_confirm"
    assert "provider" not in state.intake                         # nothing is kept until the caller says yes


def test_a_well_known_insurer_goes_straight_on(call, state):
    to_insurance(call)
    assert call.hear("Blue Cross Blue Shield").say == booking_flow.QUESTIONS["member"]
    assert state.intake["provider"] == "Blue Cross Blue Shield"


def test_yes_keeps_the_provider_and_asks_for_the_member_id(call, state):
    to_insurance(call)
    call.hear("Sophie")
    assert call.hear("yes").say == booking_flow.QUESTIONS["member"] and state.intake["provider"] == "Sophie"


def test_self_pay_after_the_read_back_skips_insurance(call, state):
    to_insurance(call)
    call.hear("Sophie")
    d = call.hear("No, self pay")
    assert d.say == booking_flow.QUESTIONS["reason"] and state.intake["insurance_done"] and not state.intake.get("provider")


def test_no_asks_for_the_provider_again_and_self_pay_is_accepted_there_too(call, state):
    to_insurance(call)
    call.hear("Sophie")
    assert call.hear("no").say == "No problem. " + booking_flow.QUESTIONS["provider"]
    assert state.bstep == "provider" and "provider_heard" not in state.intake
    assert call.hear("I'll self-pay").say == booking_flow.QUESTIONS["reason"]


def test_a_different_name_instead_of_yes_is_read_back_in_turn(call, state):
    to_insurance(call)
    call.hear("Sophie")
    assert call.hear("Aetna").say == booking_flow.QUESTIONS["member"]


def test_wait_does_not_count_and_unclear_answers_end_with_staff(call, state):
    to_insurance(call)
    call.hear("Sophie")
    assert call.hear("hold on a second").say == verify_flow.WAIT_SAY
    assert state.bstep == "provider_confirm"


def test_the_caller_can_be_asked_the_provider_question_again_after_a_silence(call, state):
    to_insurance(call)
    call.hear("Sophie")
    assert "Sophie" in booking_flow._question_for(state)


def test_the_provider_answers_are_not_stored_in_the_transcript(call, state):
    to_insurance(call)
    call.hear("Sophie")
    assert transcript.step_tag(state, "yes") == "insurance details"
    names, _, _ = transcript.known_terms(state)
    assert "Sophie" in names                                      # the agent's own read-back is masked too
    assert "Sophie" not in transcript.redact("I heard Sophie.", names)


@pytest.mark.parametrize("name,known", [("Aetna", True), ("Blue Cross Blue Shield", True), ("United Healthcare", True),
                                        ("Sophie", False), ("Banana Insurance", False), ("", False)])
def test_known_insurers(name, known):
    assert speech.is_known_insurer(name) is known
