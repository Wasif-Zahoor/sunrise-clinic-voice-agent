"""What the agent says when the caller goes quiet. Pure functions plus one small async loop; no LLM, no audio.

Only a few reprompts are spoken (each costs TTS quota), then the agent stops talking and waits. During an emergency
nothing is added to the emergency instructions."""

from __future__ import annotations

import asyncio
from typing import Any

from . import booking_flow, callback_flow, verify_flow
from .state import CallState

MAX_REPROMPTS = 3
REPROMPT_WAIT_SECONDS = 15.0
OFFER_SAY = ("I'm not hearing a response. I can connect you with a staff member, "
             "or take a message for a callback. Which would you prefer?")


def current_question(state: CallState) -> str:
    """The question the caller still owes an answer to, or '' when nothing is pending."""
    if verify_flow.active(state):
        return verify_flow._current_question(state)
    if booking_flow.active(state):
        return booking_flow._question_for(state)
    if callback_flow.active(state):
        step = state.cstep
        return callback_flow.QUESTIONS.get(step) or {"confirm": "Is that correct?",
                                                      "change": "Which one: your name, your number, or the reason?"}.get(step, "")
    if state.choice == "book_or_callback":
        return "Would you like to book an appointment, or leave a callback message?"
    if state.choice:
        return callback_flow.STAFF_OR_CALLBACK_REASK
    if state.pending is not None:
        return "Shall I go ahead?"
    return ""


def reprompt(state: CallState) -> str | None:
    """The next thing to say after the caller was silent for a while, or None to stay quiet."""
    if state.emergency or state.silence_count >= MAX_REPROMPTS:
        return None
    state.silence_count += 1
    state.dob_frag, state.frag_count = "", 0
    if state.awaiting_ssn:
        state.clear_ssn()                       # half-heard digits are dropped: the caller starts that answer again
    question = current_question(state)
    if state.silence_count == 1:
        return f"Are you still there? {question}".strip() if question else "Are you still there? How can I help you today?"
    if state.silence_count == 2:
        return f"I'm still here whenever you're ready. {question}".strip() if question else \
            "I'm still here whenever you're ready. Just tell me what you need."
    # third time: stop the open question and offer a person instead of waiting forever
    if verify_flow.active(state):
        verify_flow.abort(state, "caller_silent")
    if booking_flow.active(state):
        booking_flow.abort(state, "caller_silent")
    if callback_flow.active(state):
        callback_flow.abort(state, "caller_silent")
    state.choice, state.unclear_count = "staff_or_callback", 0
    return OFFER_SAY


async def watch(session: Any, state: CallState, wait: float = REPROMPT_WAIT_SECONDS) -> int:
    """Speak up to MAX_REPROMPTS reprompts, `wait` seconds apart, stopping as soon as the caller says something.
    Returns how many were spoken. Never raises into the call."""
    spoken, turn_at_start = 0, state.turn
    try:
        while state.turn == turn_at_start:
            text = reprompt(state)
            if text is None:
                break
            handle = session.say(text, allow_interruptions=True)
            spoken += 1
            await handle
            await asyncio.sleep(wait)
    except asyncio.CancelledError:
        raise
    except Exception:                                              # noqa: BLE001  (a failed reprompt must not end the call)
        pass
    return spoken
