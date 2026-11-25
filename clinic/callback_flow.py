"""The callback-message conversation as a code-owned state machine (name -> number -> reason -> read-back -> saved).

Used after hours, when verification fails, when staff is requested but unavailable, and whenever the agent cannot
safely finish a request. In a real call the LLM answered "call back please" with three questions in one reply, so
this is scripted exactly like verification and booking: one question per turn, the LLM is not called until saved.

The read-back and closing follow the task document: name, callback number and a few words of reason are read back,
and the closing tells the caller the team will follow up on the next business day (with the 911 reminder when closed).
"""

from __future__ import annotations

import re

from . import db, speech
from . import services as svc
from .security import audit
from .state import CallState
from .verify_flow import MAX_UNCLEAR, STAFF_OFFER_SAY, WAIT_SAY, FlowStart, FlowTurn

QUESTIONS = {
    "name": "May I have your name, please?",
    "phone": "Thank you. What is the best number to call you back on?",
    "reason": "And in a few words, what is the call about? Please leave out any health details.",
}
ORDER = ["name", "phone", "reason"]


def active(state: CallState) -> bool:
    return state.cstep != "idle"


def _audit(state: CallState, action: str, outcome: str) -> None:
    conn = db.connect(state.db_path)
    try:
        audit(conn, state.call_id, action, outcome, state.verified_patient_id)
    finally:
        conn.close()


def abort(state: CallState, why: str) -> None:
    state.cstep, state.callback, state.unclear_count, state.choice = "idle", {}, 0, None
    if state.pending and state.pending.kind == "callback":
        state.pending = None
    _audit(state, "callback_stopped", why)


def _verified_name(state: CallState) -> str | None:
    if not state.verified:
        return None
    conn = db.connect(state.db_path)
    try:
        p = conn.execute("SELECT first_name, last_name FROM patients WHERE patient_id=?",
                         (state.verified_patient_id,)).fetchone()
    finally:
        conn.close()
    return f"{p['first_name']} {p['last_name']}" if p else None


def _next(state: CallState) -> FlowTurn:
    for step in ORDER:
        if not state.callback.get(step):
            state.cstep, state.unclear_count = step, 0
            return FlowTurn(say=QUESTIONS[step])
    return _finish(state)


def begin(state: CallState, caller_name: str = "") -> str:
    """Start the conversation; returns the first question to speak."""
    state.callback, state.unclear_count, state.choice = {}, 0, None
    name = _verified_name(state) or state.confirmed_name
    if not name and caller_name:
        status, value = speech.extract_name(caller_name)
        name = value if status in ("ok", "partial") else None
    if name:
        state.callback["name"] = name
    return _next(state).say or ""


def start(state: CallState, caller_name: str = "") -> FlowStart:
    """Called by the start_callback tool. Idempotent: a duplicate call changes nothing."""
    from . import booking_flow, verify_flow
    if active(state) or booking_flow.active(state) or verify_flow.active(state):
        return FlowStart(silent=True, text="A conversation is already in progress. Say nothing.")
    with svc.operation(state, "start_callback", need_open=False) as conn:      # only the emergency lock applies
        audit(conn, state.call_id, "start_callback", "started", state.verified_patient_id)
    return FlowStart(say=begin(state, caller_name))


def _finish(state: CallState) -> FlowTurn:
    c = state.callback
    try:
        r = svc.prepare_callback(state, c["name"], c["phone"], c["reason"])
    except svc.ServiceError as e:
        fix = {"bad_name": "name", "bad_phone": "phone", "bad_reason": "reason"}.get(e.code)
        if fix is None:
            raise
        c.pop(fix, None)
        return _next(state)
    state.cstep = "confirm"
    return FlowTurn(say=f"Let me confirm. {r['readback']}. Is that correct?")


def _closing(state: CallState, name: str) -> str:
    first = name.split()[0]
    base = f"Thank you, {first}. Your message has been recorded, and our team will get back to you on the next business day."
    if state.clock.is_open():
        return f"{base} Is there anything else I can help you with?"
    evening = state.clock.now().hour >= 17 or state.clock.now().hour < 5
    return f"{base} If this is an emergency, please hang up and call 911. Have a good {'night' if evening else 'day'}."


def _retry(state: CallState, text: str, reprompt: str) -> FlowTurn:
    if speech.is_wait(text):
        return FlowTurn(say=WAIT_SAY)
    state.unclear_count += 1
    if state.unclear_count >= MAX_UNCLEAR:
        abort(state, "too_many_unclear_answers")
        return FlowTurn(say=STAFF_OFFER_SAY)
    return FlowTurn(say=reprompt)


def handle(state: CallState, text: str) -> FlowTurn | None:
    if not active(state):
        return None
    try:
        return _handle(state, text)
    except svc.ServiceError as e:            # e.g. an emergency was flagged: never crash the turn
        abort(state, f"service_error:{e.code}")
        return FlowTurn(say=STAFF_OFFER_SAY)


def _handle(state: CallState, text: str) -> FlowTurn | None:
    step, c = state.cstep, state.callback
    if speech.WANT_STAFF.search(text):
        abort(state, "caller_asked_for_staff")
        return FlowTurn(note="SYSTEM: The caller asked for a person. The message was stopped. Call request_staff_transfer "
                             "(or say the clinic is closed and offer to continue the message).")
    if speech.CANCEL.search(text):
        abort(state, "caller_cancelled")
        return FlowTurn(note="SYSTEM: The caller cancelled the message. Ask what else they need.")
    if speech.ALREADY.search(text) and step in QUESTIONS:
        return FlowTurn(say=f"I'm sorry about that. {QUESTIONS[step]}")

    if step == "name":
        status, value = speech.extract_name(text)
        if status in ("ok", "partial"):                         # a first name alone is enough for a message
            c["name"] = value
            return _next(state)
        return _retry(state, text, "Sorry, I didn't catch your name. Could you say it again?")
    if step == "phone":
        status, number = speech.phone_from_speech(text)
        if status == "ok":
            c["phone"] = number
            return _next(state)
        return _retry(state, text, "I need all ten digits. What is the best number to call you back on?"
                      if status in ("short", "long") else "Sorry, I didn't catch that. What is your phone number?")
    if step == "reason":
        if (reason := speech.reason_from_speech(text)):
            c["reason"] = reason
            return _next(state)
        return _retry(state, text, "Just a few words please. What is the call about?")

    if step == "confirm":
        answer = speech.yes_no(text)
        if answer == "yes":
            svc.confirm_callback(state)                       # saves the request + message (read-back already done)
            closing = _closing(state, c["name"])
            state.cstep, state.callback, state.choice = "idle", {}, None
            return FlowTurn(say=closing)
        if answer == "no":
            state.cstep = "change"
            state.pending = None
            return FlowTurn(say="No problem. What would you like to change: your name, your number, or the reason?")
        return _retry(state, text, "Is that correct, yes or no?")
    if step == "change":
        if re.match(r"\W*(nothing|none|nope|no changes?|no thanks|it'?s fine|that'?s (fine|ok|okay|right|correct)|all good|looks good)\b",
                    text, re.I):
            return _finish(state)                         # nothing to change: read it back again and ask to confirm
        low = text.lower()
        for pattern, field in (("name", "name"), ("number|phone", "phone"), ("reason|about", "reason")):
            if re.search(rf"\b({pattern})\b", low):
                c.pop(field, None)
                return _next(state)
        return _retry(state, text, "Which one: your name, your number, or the reason?")
    return None


# ---- "staff member or callback?" ---------------------------------------------------------------------------------
_WANT_CALLBACK = re.compile(r"call ?back|call me|message|leave|voicemail|ring me", re.I)
_WANT_STAFF = re.compile(r"staff|person|human|transfer|someone|agent|representative|connect", re.I)
STAFF_OR_CALLBACK_REASK = "Would you like to speak with a staff member, or leave a callback message?"


def handle_choice(state: CallState, text: str) -> FlowTurn | None:
    """The agent just offered staff or a callback (verification failed, a booking could not be completed...).
    Act on the caller's answer in code: no LLM improvisation, no list of questions."""
    kind = state.choice
    if not kind:
        return None
    if speech.is_wait(text):
        return FlowTurn(say=WAIT_SAY)
    yes_no = speech.yes_no(text)
    if kind == "book_or_callback":                               # a verified caller has no appointment on file
        if _WANT_CALLBACK.search(text):
            return FlowTurn(say=begin(state))
        if re.search(r"book|schedule|appointment|make one|set one up|yes", text, re.I) and yes_no != "no":
            from . import booking_flow
            state.choice = None
            try:
                return FlowTurn(say=booking_flow.start(state).say or STAFF_OR_CALLBACK_REASK)
            except svc.ServiceError:                              # e.g. the clinic closed meanwhile
                return FlowTurn(say=begin(state))
        if yes_no == "no":
            state.choice = None
            return FlowTurn(say="No problem. Is there anything else I can help you with?")
    elif _WANT_CALLBACK.search(text) or (kind == "callback_yes" and yes_no == "yes"):
        return FlowTurn(say=begin(state))
    if kind == "callback_yes" and yes_no == "no":
        state.choice = None
        return FlowTurn(say="No problem. Is there anything else I can help you with?")
    if kind == "staff_or_callback" and _WANT_STAFF.search(text):
        state.choice = None
        try:
            svc.request_handoff(state, "verification_failed" if state.locked else "cannot_complete")
        except svc.ServiceError:                                 # closed: a live transfer is not possible
            return FlowTurn(say="The clinic is closed, so I can't transfer you, but I can take a message. " + begin(state))
        return FlowTurn(say="I've asked a staff member to help you, and they will pick this up. "
                            "Is there anything else I can help you with?")
    state.unclear_count += 1
    if state.unclear_count >= 2:                                 # do not nag: hand back to normal conversation
        state.choice, state.unclear_count = None, 0
        return None
    return FlowTurn(say={"staff_or_callback": STAFF_OR_CALLBACK_REASK,
                         "book_or_callback": "Would you like to book an appointment, or leave a callback message?"}
                    .get(kind, "Would you like me to take a message?"))
