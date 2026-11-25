"""Templated spoken answers: the LLM picks the tool, code words the answer (no second LLM request). Offline."""

from __future__ import annotations

import pytest

from clinic import gate, tools, verify_flow
from clinic.turn import assess_turn
from conftest import CLOSED_EVENING, MARIA, MICHAEL, SARAH


def verified(st, who=SARAH):
    verify_flow.start(st)
    for t in (who[0], "yes", "March 14 1985" if who is SARAH else {"Michael Brown": "November 8 1972", "Maria Garcia": "July 22 1990"}[who[0]],
              " ".join(who[2])):
        assess_turn(st, gate.scrub_transcript(st, t))
    assert st.verified


@pytest.mark.parametrize("status,sentence", [
    ("Ready - doctor will call to discuss", "Your lab results are ready. Doctor will call to discuss."),
    ("Pending", "Your lab results are pending."),
    ("No recent labs", "I don't see any recent lab results on file.")])
def test_lab_wording_never_interprets_results(status, sentence):
    assert tools.lab_sentence(status) == sentence


@pytest.mark.parametrize("who,expected", [(SARAH, "Your balance due is $45.00."), (MICHAEL, "You have no balance due."),
                                          (MARIA, "Your balance due is $120.50.")])
def test_balance_is_spoken_from_a_template(state, who, expected):
    verified(state, who)
    say, text = tools.speak_balance(state)
    assert say == expected + tools.MORE and text == ""


def test_lab_and_appointment_templates(state):
    verified(state)
    assert tools.speak_lab_status(state)[0] == "Your lab results are ready. Doctor will call to discuss." + tools.MORE
    assert tools.speak_next_appointment(state)[0] == "Your next appointment is Thursday, December 3rd at 10:30 AM with Dr. Patel." + tools.MORE


def test_no_appointment_offers_booking_or_callback_in_code(state):
    verified(state, MICHAEL)
    say, text = tools.speak_next_appointment(state)
    assert say == tools.NO_APPOINTMENT_SAY and text == "" and state.choice == "book_or_callback"
    assert "book" in say and "callback" in say


def test_refill_readback_and_confirmation_are_scripted_and_never_say_approved(state, conn):
    verified(state)
    say, _ = tools.speak_refill_readback(state, "metformin")
    assert say == "Metformin 500mg at CVS Main St. There are no refills left, so this would go to the doctor for approval. Is that right?"
    d = assess_turn(state, "yes")
    assert d.note is None and d.say == tools.refill_done_say(True)
    assert "approved" not in d.say.replace("approval", "") and "doctor for approval" in d.say
    assert conn.execute("SELECT status FROM requests").fetchone()[0] == "pending_doctor_approval"


def test_refill_with_refills_left_is_submitted_not_approved(state):
    verified(state)
    assert tools.speak_refill_readback(state, "lisinopril")[0] == "Lisinopril 10mg at CVS Main St. Is that right?"
    say = assess_turn(state, "yes please").say
    assert "submitted" in say and "approved" not in say


@pytest.mark.parametrize("fn,args", [(tools.speak_balance, ()), (tools.speak_lab_status, ()), (tools.speak_next_appointment, ()),
                                     (tools.speak_refill_readback, ("lisinopril",))])
def test_every_template_still_refuses_before_verification(state, fn, args):
    say, text = fn(state, *args)
    assert say is None and text.startswith("NOT ALLOWED (not_verified)")


def test_templates_refuse_when_the_clinic_is_closed(make_state):
    st = make_state(CLOSED_EVENING)
    st.verified_patient_id = "P001"
    say, text = tools.speak_balance(st)
    assert say is None and text.startswith("NOT ALLOWED (closed)") and "45" not in text


@pytest.mark.asyncio
async def test_agent_tools_speak_the_answer_and_stop_the_llm(state, monkeypatch):
    import agent
    from livekit.agents import StopResponse

    class Stub:
        said: list[str] = []

        def say(self, text, **kw):
            self.said.append(text)

    stub = Stub()
    stub.said = []
    monkeypatch.setattr(agent.Receptionist, "session", property(lambda self: stub))
    verified(state)
    r = agent.Receptionist(state)
    for name, args in (("get_balance_due", {}), ("get_lab_status", {}), ("get_next_appointment", {}),
                       ("prepare_refill", {"medication_name": "lisinopril"})):
        tool = next(t for t in r.tools if t.id == name)
        with pytest.raises(StopResponse):
            await tool(**args)
    assert stub.said[0].startswith("Your balance due is $45.00") and stub.said[3].startswith("Lisinopril 10mg at CVS")
    # a refusal is returned as text for the LLM to relay, not spoken
    other = agent.Receptionist(state.__class__(db_path=state.db_path, clock=state.clock))
    out = await next(t for t in other.tools if t.id == "get_balance_due")()
    assert out.startswith("NOT ALLOWED (not_verified)") and len(stub.said) == 4


# ===== guards added after the live scenario matrix ==================================================================
MORE_EMERGENCIES = ["I feel really dizzy and my left arm has gone numb and I think I'm going to faint.",
                    "my face is drooping and my speech is slurred", "I think my dad collapsed", "she's having an allergic reaction",
                    "my throat is swelling up", "he is vomiting blood", "the bleeding won't stop", "I can't feel my legs",
                    "I'm going to pass out"]
NOT_EMERGENCIES = ["my foot fell asleep a bit", "my arm is sore from the gym", "I fainted last year, can I book a check-up",
                   "what time do you open on Friday", "I need to update my emergency contact"]


@pytest.mark.parametrize("text", MORE_EMERGENCIES)
def test_plain_word_emergencies_are_caught_by_code_not_by_the_model(state, text):
    d = assess_turn(state, text)
    assert d.say and "911" in d.say and state.emergency


@pytest.mark.parametrize("text", NOT_EMERGENCIES)
def test_everyday_sentences_do_not_trigger_the_emergency_script(state, text):
    d = assess_turn(state, text)
    assert not state.emergency and "911" not in (d.say or "")      # (booking requests now start the booking flow in code)


def test_naming_another_person_blocks_every_patient_data_tool_for_that_turn(state):
    """A model once called get_balance_due for 'what does my husband owe' and code then spoke the CALLER's own balance."""
    verified(state)
    assess_turn(state, "What does my husband Michael Brown owe on his account?")
    for fn, args in ((tools.speak_balance, ()), (tools.speak_lab_status, ()), (tools.speak_next_appointment, ()),
                     (tools.speak_refill_readback, ("lisinopril",))):
        say, text = fn(state, *args)
        assert say is None and text.startswith("NOT ALLOWED (third_party)"), fn.__name__
        assert "45" not in text and "Johnson" not in text
    assert tools.get_balance_due(state).startswith("NOT ALLOWED (third_party)")


def test_the_third_party_block_lasts_one_turn_only(state):
    verified(state)
    assess_turn(state, "What does my wife owe?")
    assert tools.speak_balance(state)[0] is None
    assess_turn(state, "Okay, and what do I owe myself?")
    assert tools.speak_balance(state)[0].startswith("Your balance due is $45.00")


def test_the_block_is_audited_without_phi(state, conn):
    verified(state)
    assess_turn(state, "What does my mother owe?")
    tools.speak_balance(state)
    row = conn.execute("SELECT action, outcome, detail FROM audit_logs WHERE outcome='denied:third_party'").fetchone()
    assert row is not None and row["action"] == "get_balance" and not row["detail"]
