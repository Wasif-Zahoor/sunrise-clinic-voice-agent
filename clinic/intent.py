"""Recognise a clear request from the caller's first words and start the right conversation in code.

Why: a garbled "I would like to get a prescription refill" was answered by the LLM with "I can't handle prescription requests", and
"can you schedule an appointment" took three LLM round trips before anything happened ("Hello?"). Opening requests must not depend on
the model's judgement or its latency.

  * UNVERIFIED caller, account request (refill, labs, balance, my appointment): verification starts; the request is remembered
    (`state.after_verify`) and continued right after verification (clinic.verify_flow.continue_intent).
  * VERIFIED caller: the answer is given immediately (templates), or the refill / booking conversation starts.
  * Booking (new or existing patient): the booking conversation starts; it asks "first visit?" and verifies existing patients itself.

Nothing here relaxes a rule: every step still goes through the backend checks. AFTER HOURS the same requests get one fixed answer
(`closed_say`): closed, the hours, the saved message or an offer to take one; nothing is looked up. Not routed at all: a locked
caller, an emergency, a turn already flagged as an injection attempt or a third-party request (the LLM must see that note), or while
another conversation is running.
"""

from __future__ import annotations

import re

from . import booking_flow, gate, refill_flow, tools, verify_flow
from .clock import HOURS_TEXT
from .state import CallState

# order matters: the first match wins. Lookups of the caller's OWN appointment come before booking.
INTENTS: list[tuple[str, re.Pattern]] = [
    ("refill", re.compile(r"\b(refill|re-fill|prescription|prescriptions|medication|medications|medicine|meds|pharmacy)\b", re.I)),
    # "last status" / "lap results": what speech-to-text made of "lab status" in a real call
    ("lab", re.compile(r"\b(labs?|lab results?|test results?|blood ?work|my results|(?:last|lap) (?:status|results?))\b", re.I)),
    ("balance", re.compile(r"\b(balance|owe|bill|billing|how much do i)\b", re.I)),
    ("appointment", re.compile(r"\b(my (next |upcoming |current |existing )?appointments?|when is my|appointment (time|date|details)|"
                               r"do i have an appointment|check (on )?my appointment)\b", re.I)),
    ("booking", re.compile(r"\b(schedule|book|make|set up|setup|need|want|get)\b[^.?!]{0,30}\bappointments?\b|\bnew appointment\b|"
                           r"\bbook (me|with|an?)\b", re.I)),
]
_ABOUT_SERVICES = re.compile(r"\b(what (services|can you|do you)|do you (offer|provide|handle|do)|which services)\b", re.I)


def detect(text: str) -> str | None:
    if _ABOUT_SERVICES.search(text):
        return None
    for name, pattern in INTENTS:
        if pattern.search(text):
            return name
    return None


def _busy(state: CallState) -> bool:
    from . import callback_flow
    return (state.locked or state.emergency or bool(state.choice) or state.pending is not None
            or verify_flow.active(state) or booking_flow.active(state) or callback_flow.active(state) or refill_flow.active(state))


def _verified(state: CallState, name: str, text: str) -> str | None:
    if name == "refill":
        return refill_flow.begin(state, text).say
    if name == "booking":
        return booking_flow.start(state).say
    say, _ = {"lab": tools.speak_lab_status, "balance": tools.speak_balance, "appointment": tools.speak_next_appointment}[name](state)
    return say                                          # None when refused: the LLM and its note take over


def closed_say(state: CallState) -> str:
    """After hours, a request for patient information or a booking is answered in code, the same way every time: the clinic is
    closed, the hours, and either the message already recorded or an offer to take one. Nothing is looked up or changed."""
    base = f"The clinic is closed, so I can't share or change patient information right now. Our hours are {HOURS_TEXT}."
    if "callback message saved" in state.completed:
        return f"{base} Your message is recorded, and our team will follow up on the next business day. Is there anything else I can help you with?"
    state.choice, state.unclear_count = "callback_yes", 0                     # a "yes" starts the message in code
    return f"{base} I can take a message for our team. Would you like that?"


def route(state: CallState, text: str) -> str | None:
    """The first thing to say when `text` is a clear request this module handles, else None (the LLM answers as before)."""
    if _busy(state) or state.third_party_turn == state.turn:
        return None
    name = detect(text)
    if name is None:
        return None
    if not state.clock.is_open():
        return closed_say(state)
    try:
        if state.verified:
            return _verified(state, name, text)
        if state.new_patient_id and name != "booking":      # registered as a new patient in this call: nothing to verify, nothing on file
            return tools.speak_new_patient(state, name)[0]
        if name == "booking":
            return booking_flow.start(state).say
        state.after_verify = name
        started = verify_flow.start(state)
    except Exception:                                  # noqa: BLE001  (closed, locked, refused...: leave it to the normal path)
        state.after_verify = None
        return None
    if not started.say:
        state.after_verify = None
        return None
    return started.say
