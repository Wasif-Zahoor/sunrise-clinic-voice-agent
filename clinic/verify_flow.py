"""The verification conversation as an explicit, code-owned state machine.

    idle -> name -> name_confirm (-> name_spell) -> dob (-> dob_clarify) -> ssn (state.awaiting_ssn)
         -> verified | retry | locked

While a step is active the LLM is skipped: every reply is scripted, one question at a time. A step
advances only when its answer is usable. The name is read back for confirmation (a misheard name would
otherwise burn one of the three attempts). Unusable answers ("um", "wait", a half-heard name, an
impossible date) re-ask the SAME question briefly and are counted; they never touch the failed-
attempt counter (only a complete name + date + last-four mismatch does), and wording changes can
never reset that counter because it lives in CallState.failed_attempts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import dates, db, gate, speech
from . import services as svc
from .security import audit
from .speech import extract_name  # noqa: F401  (re-exported)
from .state import CallState

Q_NAME = "May I have your full name, please?"
Q_DOB = "Thank you. May I have your date of birth, please?"
Q_SSN = "Thank you. And the last four digits of your Social Security number?"
Q_NAME_SPELL = "Sorry about that. Could you spell your first and last name for me?"
WAIT_SAY = "Of course, take your time."
STAFF_OFFER_SAY = ("I'm having trouble with that. I can connect you with a staff member, "
                   "or take a message for a callback. Which would you prefer?")
RETRY_SAY = "I'm sorry, I couldn't verify those details. Let's try once more. " + Q_NAME
LOCKED_SAY = ("I'm sorry, I couldn't verify those details, and I can't continue verification on this call. "
              "I can connect you with a staff member, or take a message for a callback. Which would you prefer?")
CLOSED_SAY = ("The clinic is now closed, so I can't access patient information. "
              "I can take a message for the team. May I have your name?")
DECLINED_SSN_SAY = ("That's okay. I can connect you with a staff member, or take a message for a callback. "
                    "Which would you prefer?")
MAX_UNCLEAR = 3


def q_name_confirm(name: str) -> str:
    return f"Thank you. I have {name}. Is that right?"


@dataclass
class FlowStart:
    say: str | None = None     # scripted first question (spoken by code)
    text: str = ""             # instruction for the LLM when nothing is spoken by code
    silent: bool = False       # duplicate call: say and do nothing


@dataclass
class FlowTurn:
    say: str | None = None
    note: str | None = None
    silent: bool = False       # say nothing and do not call the LLM: the caller is probably still talking


# ---- helpers --------------------------------------------------------------------------------------
def _audit(state: CallState, action: str, outcome: str) -> None:
    conn = db.connect(state.db_path)
    try:
        audit(conn, state.call_id, action, outcome, state.verified_patient_id)
    finally:
        conn.close()


def active(state: CallState) -> bool:
    return state.vstep != "idle" or state.awaiting_ssn


def abort(state: CallState, why: str) -> None:
    state.vstep, state.awaiting_ssn = "idle", False
    state.pending_name = state.pending_dob = state.partial_first = state.dob_hint = None
    state.after_verify = None
    state.dob_candidates = []
    state.unclear_count = 0
    state.dob_frag, state.frag_count = "", 0
    state.clear_ssn()
    _audit(state, "verification_stopped", why)


def _unclear(state: CallState, reprompt: str) -> FlowTurn:
    state.unclear_count += 1
    if state.unclear_count >= MAX_UNCLEAR:
        abort(state, "too_many_unclear_answers")
        return FlowTurn(say=STAFF_OFFER_SAY)
    return FlowTurn(say=reprompt)


def _wait_or_unclear(state: CallState, text: str, reprompt: str) -> FlowTurn:
    if speech.is_wait(text):
        return FlowTurn(say=WAIT_SAY)            # not counted as an unclear answer
    return _unclear(state, reprompt)


def _current_question(state: CallState) -> str:
    if state.awaiting_ssn:
        return "What are the last four digits of your Social Security number?"
    return {"name": Q_NAME, "name_confirm": q_name_confirm(state.pending_name or ""), "name_spell": Q_NAME_SPELL,
            "dob": "May I have your date of birth, please?"}.get(state.vstep, "")


# ---- entry ----------------------------------------------------------------------------------------
def start(state: CallState, name_hint: str = "") -> FlowStart:
    """Called by the start_verification tool. Idempotent: a duplicate call changes nothing."""
    if state.verified:
        return FlowStart(text="The caller is already verified. Continue with their request.")
    if active(state):
        return FlowStart(silent=True, text="Verification is already in progress. Say nothing.")
    with svc.operation(state, "start_verification") as conn:       # emergency / closed / locked checks
        if state.locked:
            audit(conn, state.call_id, "start_verification", "denied:locked")
            raise svc.ServiceError("locked")
        audit(conn, state.call_id, "start_verification", "started")
    state.unclear_count = 0
    state.partial_first = state.dob_hint = None
    if name_hint:
        status, value = speech.extract_name(name_hint)
        if status == "ok":
            state.pending_name, state.vstep = value, "name_confirm"
            return FlowStart(say=q_name_confirm(value))
    state.vstep = "name"
    return FlowStart(say=Q_NAME)


# ---- per-turn handling -------------------------------------------------------------------------------
def _after_dob(state: CallState, iso: str) -> FlowTurn:
    r = svc.begin_verification(state, state.pending_name or "", iso)
    state.vstep = "idle"
    state.unclear_count = 0
    state.dob_frag, state.frag_count = "", 0
    if r["status"] == "awaiting_ssn":
        return FlowTurn(say=Q_SSN)
    if r["status"] in ("verified", "failed", "locked"):         # digits were volunteered earlier
        return _result(state, r)
    abort(state, "bad_input_after_dob")
    state.vstep = "name"
    return FlowTurn(say=Q_NAME)


def _result(state: CallState, r: dict) -> FlowTurn:
    s = r["status"]
    if s == "verified":
        state.vstep = "idle"
        if state.after_verify == "booking":                     # the caller came here to book: carry on
            state.after_verify = None
            from . import booking_flow
            nxt = booking_flow.begin_existing(state)
            if nxt:
                return FlowTurn(say=f"Thank you, you're verified. {nxt.removeprefix('Thank you. ')}")
        if state.after_verify in INTENT_STEPS:                  # the caller's first request was recognised in code: carry on with it
            intent, state.after_verify = state.after_verify, None
            say, text = continue_intent(state, intent)
            if say:
                return FlowTurn(say=f"Thank you, you're verified. {say}")
            return FlowTurn(note=f"SYSTEM: Identity verified by the backend. {text}")
        return FlowTurn(note="SYSTEM: Identity verified by the backend. Continue with the caller's original request.")
    if s == "locked":
        state.vstep, state.after_verify = "idle", None
        return FlowTurn(say=LOCKED_SAY)
    state.vstep, state.partial_first, state.unclear_count = "name", None, 0     # failed: start over, same attempt count
    return FlowTurn(say=RETRY_SAY)


INTENT_STEPS = {"refill", "lab", "balance", "appointment"}
_DATEISH = re.compile(r"\d|birth|born|dob", re.I)


def continue_intent(state: CallState, intent: str) -> tuple[str | None, str]:
    """What to say right after verification for the request the caller opened with -> (spoken text, note for the LLM)."""
    from . import tools
    if intent == "refill":
        from . import refill_flow
        return refill_flow.begin(state).say, ""
    return {"lab": tools.speak_lab_status, "balance": tools.speak_balance,
            "appointment": tools.speak_next_appointment}[intent](state)


def dob_with_fragments(state: CallState, text: str) -> tuple[str, dates.DateResult, bool]:
    """A date often arrives in pieces (\"My date of birth is\" / \"March\" / \"14, 1985\"). Join them rather than counting each
    piece as an unclear answer. -> (text to use, parse result, hold). hold=True: still incomplete, stay quiet for now."""
    combined = f"{state.dob_frag} {text}".strip() if state.dob_frag else text
    res = dates.parse_spoken_date(combined)
    if res.status in ("ok", "ambiguous"):
        state.dob_frag, state.frag_count = "", 0
        return combined, res, False
    alone = dates.parse_spoken_date(text)
    if _is_wait(text, alone):
        return text, alone, False
    words = re.findall(r"[a-z]+", text.lower())
    dateish = bool(_DATEISH.search(text)) or any(w in dates._MONTHS or w in dates._WORDNUM for w in words)
    if res.status in ("none", "incomplete") and dateish and state.frag_count < 2 and len(combined.split()) <= 20:
        state.dob_frag, state.frag_count = combined, state.frag_count + 1
        return combined, res, True
    state.dob_frag, state.frag_count = "", 0
    return combined, res, False


def _is_wait(text: str, res: dates.DateResult) -> bool:
    """"Hold on a second" is a wait, not the day number 2 of a half-said date."""
    has_month = any(w in dates._MONTHS for w in re.findall(r"[a-z]+", text.lower()))
    return speech.is_wait(text) and not has_month and res.status != "ok"


def _ask_dob(state: CallState, res: dates.DateResult, text: str) -> FlowTurn:
    if _is_wait(text, res):
        return FlowTurn(say=WAIT_SAY)
    if res.status == "ok":
        return _after_dob(state, res.iso)
    if res.status == "ambiguous":
        a, b = (dates.speak_date(c) for c in res.candidates)
        state.vstep, state.dob_candidates = "dob_clarify", res.candidates
        return FlowTurn(say=f"Just to be sure, is that {a}, or {b}?")
    if res.status == "invalid":
        return _unclear(state, "I'm not sure I got that date. Could you say it again with the month, day and year?")
    if res.status == "incomplete":
        return _unclear(state, "I missed part of that. Could you give me the month, day and year?")
    return _wait_or_unclear(state, text, "Sorry, I didn't catch that. What is your date of birth?")


def handle(state: CallState, text: str) -> FlowTurn | None:
    """Handle one caller turn while verification is in progress; None when it is not."""
    if not active(state):
        return None
    try:
        return _handle(state, text)
    except svc.ServiceError as e:          # e.g. the clinic closed mid-conversation: never crash the turn
        abort(state, f"service_error:{e.code}")
        return FlowTurn(say={"closed": CLOSED_SAY, "locked": LOCKED_SAY}.get(e.code, STAFF_OFFER_SAY))


def _set_name(state: CallState, name: str, text: str) -> FlowTurn:
    """A name was heard: read it back and ask the caller to confirm it."""
    state.pending_name, state.partial_first, state.unclear_count = name, None, 0
    res = dates.parse_spoken_date(text)              # the caller may have said both at once
    state.dob_hint = res.iso if res.status == "ok" else None
    state.vstep = "name_confirm"
    return FlowTurn(say=q_name_confirm(name))


def _handle(state: CallState, text: str) -> FlowTurn | None:
    # the caller changed their mind: leave verification, let the LLM (or the booking flow) take over
    if speech.WANT_STAFF.search(text):
        abort(state, "caller_asked_for_staff")
        return FlowTurn(note="SYSTEM: The caller asked for a person. Verification was stopped. Call request_staff_transfer "
                             "(or take a callback message if the clinic is closed).")
    if speech.NEW_PATIENT.search(text):
        abort(state, "caller_is_new_patient")
        from . import booking_flow
        return FlowTurn(say=booking_flow.begin_new(state, welcome="No problem, welcome to Sunrise Family Clinic!"))
    if speech.CANCEL.search(text) and not state.awaiting_ssn:
        abort(state, "caller_cancelled")
        return FlowTurn(note="SYSTEM: The caller cancelled verification. Ask what else they need.")
    if speech.ALREADY.search(text) and (q := _current_question(state)):
        return FlowTurn(say=f"I'm sorry about that. {q}")        # never argue; ask the same thing once more

    # ---- step: last four digits ------------------------------------------------------------------
    if state.awaiting_ssn:
        status = gate.ssn_input_status(state)
        if status == "ready":
            state.frag_count = 0
            try:
                result = svc.complete_verification(state)
            except svc.ServiceError as e:
                abort(state, f"service_error:{e.code}")
                return FlowTurn(say=CLOSED_SAY if e.code == "closed" else LOCKED_SAY)
            return _result(state, result)
        if status == "short":
            if state.frag_count < 3:                    # digits come in pieces (even one at a time): keep them, wait
                state.frag_count += 1
                return FlowTurn(silent=True)
            state.clear_ssn()
            state.frag_count = 0
            return _unclear(state, "I only caught part of that. Could you say all four digits?")
        if status == "long":
            state.clear_ssn()
            return _unclear(state, "Please give me only the last four digits.")
        if speech.REFUSE.search(text):
            abort(state, "caller_declined_ssn")
            return FlowTurn(say=DECLINED_SSN_SAY)
        return _wait_or_unclear(state, text, "Could you tell me just the last four digits of your Social Security number?")

    # ---- step: name -----------------------------------------------------------------------------------
    if state.vstep in ("name", "name_spell"):
        status, value = speech.extract_name(text, state.partial_first)
        if status == "ok":
            return _set_name(state, value, text)
        if status == "partial":
            state.partial_first = value
            return FlowTurn(say=f"Thanks, {value}. And your last name?")
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        return _unclear(state, "Sorry, I didn't catch your name. Could you tell me your first and last name?"
                        if state.vstep == "name" else Q_NAME_SPELL)

    # ---- step: confirm the name that was read back ----------------------------------------------------
    if state.vstep == "name_confirm":
        answer = speech.yes_no(text)
        if answer == "yes":
            state.unclear_count = 0
            state.confirmed_name = state.pending_name
            if state.dob_hint:
                iso, state.dob_hint = state.dob_hint, None
                return _after_dob(state, iso)
            state.vstep = "dob"
            return FlowTurn(say=Q_DOB)
        status, value = speech.extract_name(text)
        if status == "ok" and value != state.pending_name:      # "no, it's Jane Doe": take the correction
            return _set_name(state, value, text)
        if answer == "no":
            state.vstep = "name_spell"
            state.partial_first = None
            return _unclear(state, Q_NAME_SPELL)
        if status == "ok":                                      # said the same name again: that is a yes
            state.vstep = "dob"
            return FlowTurn(say=Q_DOB)
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        return _unclear(state, f"Is {state.pending_name} correct? Please say yes or no.")

    # ---- step: date of birth ---------------------------------------------------------------------
    if state.vstep == "dob":
        text, res, hold = dob_with_fragments(state, text)
        if hold:
            return FlowTurn(silent=True)
        return _ask_dob(state, res, text)

    # ---- step: clarify an ambiguous date --------------------------------------------------------
    if state.vstep == "dob_clarify":
        iso = dates.resolve_ambiguity(text, state.dob_candidates)
        if iso is None:
            res = dates.parse_spoken_date(text)
            if res.status == "ok":
                iso = res.iso
        if iso:
            return _after_dob(state, iso)
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        a, b = (dates.speak_date(c) for c in state.dob_candidates)
        return _unclear(state, f"Sorry, which was it: {a}, or {b}?")
    return None
