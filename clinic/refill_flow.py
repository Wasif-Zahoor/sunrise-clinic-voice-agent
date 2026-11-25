"""The prescription-refill conversation as a code-owned state machine (which medication -> read back -> yes -> request saved).

Why: in a real call the caller's drug name reached the LLM as "Lisey Nuperell", then "L-I-S-I-L, N-O-P-R-I-L"; the model never called
the tool and said "I can't help with that". Speech-to-text mangles drug names, so the match is made in code, fuzzily, but ONLY among the
verified caller's own medications (nothing else is ever compared or disclosed), and the caller always confirms the read-back
("Lisinopril 10mg at CVS Main St. Is that right?"). The save after a clear yes is done by the turn hook, as before.
"""

from __future__ import annotations

import re

from . import db, speech, tools
from . import services as svc
from .security import audit
from .state import CallState
from .verify_flow import STAFF_OFFER_SAY, WAIT_SAY, FlowTurn

Q_MED = "Which medication would you like to refill?"
MAX_TRIES = 3


def active(state: CallState) -> bool:
    return state.refill_step != "idle"


def abort(state: CallState, why: str) -> None:
    state.refill_step, state.refill_tries = "idle", 0
    conn = db.connect(state.db_path)
    try:
        audit(conn, state.call_id, "refill_stopped", why, state.verified_patient_id)
    finally:
        conn.close()


def begin(state: CallState, text: str = "") -> FlowTurn:
    """Start (the caller is verified). If the opening sentence already names the medication, go straight to the read-back."""
    state.refill_tries = 0
    try:
        if text and svc.has_medication(state, text):
            return _try(state, text)
    except svc.ServiceError:
        pass
    state.refill_step = "ask"
    return FlowTurn(say=Q_MED)


def handle(state: CallState, text: str) -> FlowTurn | None:
    if not active(state):
        return None
    if speech.WANT_STAFF.search(text):
        abort(state, "caller_asked_for_staff")
        return FlowTurn(note="SYSTEM: The caller asked for a person. The refill was stopped. Call request_staff_transfer.")
    if speech.CANCEL.search(text):
        abort(state, "caller_cancelled")
        return FlowTurn(note="SYSTEM: The caller cancelled the refill. Ask what else they need.")
    if speech.is_wait(text):
        return FlowTurn(say=WAIT_SAY)
    if says_no_medication(text):
        return FlowTurn(say=Q_MED)                                 # "Thank you", "Yeah", "I'd like a refill": not a miss
    return _try(state, text)


# Words that can appear without a medication name in them. A turn made only of these (a thank-you, a "yeah", the request said again)
# is the caller talking, not a failed attempt at the name: in a real call three of them used up all the tries before "Metformin".
_NO_MEDICATION_WORDS = frozenset("""yes yeah yep yup yea okay ok sure right fine alright thanks thank you hello hi hey um uh er hmm
oh well so please no nope i id i'd i'm im would like to get a an the my want need needs refill refills refilled prescription
prescriptions rifle medication medicine medicines meds one also just again can could for it that is s ll d as of me""".split())


def says_no_medication(text: str) -> bool:
    words = re.findall(r"[a-z']+", text.lower())
    return bool(words) and all(w in _NO_MEDICATION_WORDS for w in words)


def _try(state: CallState, text: str) -> FlowTurn:
    try:
        r = svc.prepare_refill(state, text)
    except svc.ServiceError as e:
        if e.code == "ambiguous":
            return FlowTurn(say="More than one of your medications sounds like that. Could you say the full name?")
        if e.code in ("not_on_file", "need_medication"):
            state.refill_tries += 1
            if state.refill_tries >= MAX_TRIES:
                state.refill_step, state.refill_tries = "idle", 0
                return FlowTurn(say=STAFF_OFFER_SAY)               # the turn hook remembers the offer, so the answer is handled in code
            return FlowTurn(say="Sorry, I couldn't find that one. Could you say the medication name again, or spell it for me?")
        abort(state, f"service_error:{e.code}")                    # closed, emergency, not verified...: never crash the turn
        return FlowTurn(say=STAFF_OFFER_SAY)
    state.refill_step, state.refill_tries = "idle", 0
    return FlowTurn(say=tools.refill_readback_say(r))              # a clear yes on the next turn saves it (turn._auto_confirm)
