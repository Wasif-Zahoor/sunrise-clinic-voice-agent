"""The appointment-booking conversation as a code-owned state machine (like verification).

New patient:   first_visit -> name -> name_confirm -> dob -> phone -> email -> insurance (-> provider -> member)
               -> reason -> preference -> slot -> confirm -> saved
Verified:      (details on file are skipped) -> [email] -> reason -> preference -> slot -> confirm -> saved

Why code and not the LLM: in a real call the LLM listed all seven intake questions in one reply, forgot
answers it had already been given, and never read the name back. Here every turn asks exactly ONE
question, answers are parsed by deterministic code (clinic/speech.py), the name is read back, nothing is
asked twice, and the LLM is not called at all until the booking is finished. That also removes most LLM
token use during a booking.

Saving still goes through clinic.services (read-back, explicit yes, atomic slot removal).
"""

from __future__ import annotations

import re

from . import dates, db, speech
from . import services as svc
from . import verify_flow as vf
from .security import audit
from .state import CallState
from .verify_flow import (CLOSED_SAY, MAX_UNCLEAR, Q_NAME, Q_NAME_SPELL, STAFF_OFFER_SAY, WAIT_SAY, FlowStart, FlowTurn,
                          q_name_confirm)

Q_FIRST_VISIT = "Is this your first visit with Sunrise Family Clinic?"
QUESTIONS = {
    "name": Q_NAME,
    "dob": "Thank you. May I have your date of birth, please?",
    "phone": "Thank you. What is the best phone number to reach you?",
    "email": "Thank you. And your email address?",
    "insurance": "Do you have insurance, or would you like to self-pay?",
    "provider": "Who is your insurance provider?",
    "member": "And your member ID?",
    "reason": "In a few words, what is the visit for?",
    "preference": "Do you have a preferred doctor or day? Dr. Patel and Dr. Lee are both available.",
}
NEW_ORDER = ["name", "dob", "phone", "email", "insurance", "reason", "preference", "slot"]
EXISTING_ORDER = ["email", "reason", "preference", "slot"]
CANNOT_COMPLETE_SAY = ("I'm not able to complete this booking by phone. I can connect you with a staff member, "
                       "or take a message for a callback. Which would you prefer?")
NO_SLOTS_SAY = "I'm sorry, I don't have any openings right now. I can have a staff member call you back. Would you like that?"
_CHANGE_FIELDS = [("email", "email"), ("phone|number", "phone"), ("name", "name"), ("birth|birthday|dob", "dob"),
                  ("reason|visit", "reason"), ("time|date|day|slot|doctor|appointment|opening|different|other", "preference")]


def active(state: CallState) -> bool:
    return state.bstep != "idle"


def _audit(state: CallState, action: str, outcome: str) -> None:
    conn = db.connect(state.db_path)
    try:
        audit(conn, state.call_id, action, outcome, state.verified_patient_id)
    finally:
        conn.close()


def abort(state: CallState, why: str) -> None:
    state.bstep, state.intake, state.unclear_count, state.partial_first = "idle", {}, 0, None
    if state.pending and state.pending.kind == "booking":
        state.pending = None
    _audit(state, "booking_stopped", why)


def _retry(state: CallState, text: str, reprompt: str, limit: int = MAX_UNCLEAR) -> FlowTurn:
    """An unusable answer: wait gracefully, otherwise re-ask the same question (`limit` in a row -> staff)."""
    if speech.is_wait(text):
        return FlowTurn(say=WAIT_SAY)
    state.unclear_count += 1
    if state.unclear_count >= limit:
        abort(state, "too_many_unclear_answers")
        return FlowTurn(say=STAFF_OFFER_SAY)
    return FlowTurn(say=reprompt)


# ---- entry points ---------------------------------------------------------------------------------------
def begin_new(state: CallState, welcome: str = "Welcome to Sunrise Family Clinic!") -> str:
    state.intake, state.bstep, state.unclear_count, state.partial_first = {"new": True}, "name", 0, None
    return f"{welcome} {Q_NAME}"


def begin_existing(state: CallState) -> str | None:
    """Verified patient: details on file are not asked again. Returns the first question."""
    conn = db.connect(state.db_path)
    try:
        p = conn.execute("SELECT * FROM patients WHERE patient_id=?", (state.verified_patient_id,)).fetchone()
    finally:
        conn.close()
    if p is None:
        return None
    state.intake = {"new": False, "name": f"{p['first_name']} {p['last_name']}", "dob": p["dob"], "phone": p["phone"],
                    "email": p["email"] or None, "insurance_done": True}
    state.unclear_count = 0
    return _after(state, "start").say


def _verify_first(state: CallState) -> FlowStart:
    state.after_verify = "booking"
    r = vf.start(state)
    if r.say:
        return FlowStart(say=f"Let me verify your identity first. {r.say}")
    return r


def start(state: CallState, first_visit: str = "") -> FlowStart:
    """Called by the start_booking tool. Idempotent: a duplicate call changes nothing."""
    if active(state) or vf.active(state):
        return FlowStart(silent=True, text="A booking or verification is already in progress. Say nothing.")
    with svc.operation(state, "start_booking") as conn:          # emergency / closed checks
        audit(conn, state.call_id, "start_booking", "started")
    if state.verified:
        return FlowStart(say=begin_existing(state), text="Could not load the patient record. Offer a staff callback.")
    fv = (first_visit or "").strip().lower()
    if fv in ("yes", "true", "new", "first", "first visit"):
        return FlowStart(say=begin_new(state))
    if fv in ("no", "false", "existing", "old", "returning"):
        return _verify_first(state)
    state.bstep, state.intake, state.unclear_count = "first_visit", {"new": None}, 0
    return FlowStart(say=Q_FIRST_VISIT)


# ---- moving between steps ----------------------------------------------------------------------------------
def _have(state: CallState, step: str) -> bool:
    i = state.intake
    return {"name": bool(i.get("name")), "dob": bool(i.get("dob")), "phone": bool(i.get("phone")),
            "email": bool(i.get("email")), "insurance": bool(i.get("insurance_done")), "reason": bool(i.get("reason")),
            "preference": "pref" in i, "slot": bool(i.get("slot_id"))}.get(step, False)


def _goto(state: CallState, step: str, prefix: str = "") -> FlowTurn:
    state.bstep, state.unclear_count = step, 0
    if step == "slot":
        return _offer_slots(state, prefix)
    return FlowTurn(say=prefix + QUESTIONS[step])


def _after(state: CallState, done: str) -> FlowTurn:
    """The step `done` is complete: ask the next missing thing, or read everything back."""
    if state.intake.pop("return", False):
        return _finish(state)
    order = NEW_ORDER if state.intake.get("new") else EXISTING_ORDER
    start = order.index(done) + 1 if done in order else 0
    for step in order[start:]:
        if not _have(state, step):
            return _goto(state, step)
    return _finish(state)


def _offer_slots(state: CallState, prefix: str = "") -> FlowTurn:
    pref = state.intake.get("pref") or {}
    r = svc.find_slots(state, pref.get("doctor", ""), pref.get("weekday", ""), pref.get("time_of_day", ""))
    if not r["slots"]:
        abort(state, "no_slots")
        return FlowTurn(say=NO_SLOTS_SAY)
    offers = [{"slot_id": s["slot_id"], "doctor": s["doctor"], "when": s["when"],
               "starts_at": state.offered_slots[s["slot_id"]]["starts_at"]} for s in r["slots"]]
    state.intake["offers"] = offers
    state.bstep, state.unclear_count = "slot", 0
    lead = "I have these openings." if r["matched_preference"] else "I don't have an exact match, but I do have these."
    return FlowTurn(say=f"{prefix}{lead} " + " ".join(f"{o['doctor']}, {o['when']}." for o in offers) + " Which would you like?")


def _finish(state: CallState) -> FlowTurn:
    i = state.intake
    try:
        if i.get("new"):
            r = svc.prepare_booking_new(state, i["slot_id"], i["name"], i["dob"], i["phone"], i["email"],
                                        i.get("provider") or "self-pay", i.get("member") or "", i["reason"])
        else:
            r = svc.prepare_booking_existing(state, i["slot_id"], i["reason"], i.get("email") or "")
    except svc.ServiceError as e:
        return _prepare_failed(state, e)
    state.bstep = "confirm"
    extra = f"Born {dates.speak_date(i['dob'])}. Phone {i['phone']}. " if i.get("new") else ""
    return FlowTurn(say=f"Let me confirm. {r['readback']}. {extra}Your email is {r['email_spelled']}. "
                        "Is all of that correct?")


def _provider_heard(state: CallState, provider: str) -> FlowTurn:
    """A well-known insurer goes straight on to the member ID. Anything else is read back first: speech-to-text turns "self-pay"
    into things like "Sophie", and a made-up insurer would be saved on the patient's record."""
    i = state.intake
    if speech.is_known_insurer(provider):
        i["provider"] = provider
        return _goto(state, "member")
    i["provider_heard"] = provider
    state.bstep, state.unclear_count = "provider_confirm", 0
    return FlowTurn(say=f"I heard {provider}. Is that your insurance provider, or would you like to self-pay?")


def _prepare_failed(state: CallState, e: svc.ServiceError) -> FlowTurn:
    i = state.intake
    if e.code == "cannot_complete":
        abort(state, "cannot_complete")
        return FlowTurn(say=CANNOT_COMPLETE_SAY)
    if e.code in ("slot_taken", "slot_not_offered"):
        i.pop("slot_id", None)
        i["return"] = True
        return _offer_slots(state, "Sorry, that time is no longer available. ")
    fix = {"bad_email": "email", "need_email": "email", "bad_phone": "phone", "bad_dob": "dob", "bad_name": "name",
           "need_insurance": "insurance", "need_reason": "reason", "reason_too_long": "reason"}.get(e.code)
    if fix is None:
        raise e
    for k in {"email": ["email"], "phone": ["phone"], "dob": ["dob"], "name": ["name"],
              "insurance": ["insurance_done", "provider", "member"], "reason": ["reason"]}[fix]:
        i.pop(k, None)
    i["return"] = True
    return _goto(state, fix, "Sorry, I need that again. ")


# ---- per-turn handling ------------------------------------------------------------------------------------------
def handle(state: CallState, text: str) -> FlowTurn | None:
    if not active(state):
        return None
    try:
        return _handle(state, text)
    except svc.ServiceError as e:            # e.g. the clinic closed mid-booking: never crash the turn
        abort(state, f"service_error:{e.code}")
        return FlowTurn(say=CLOSED_SAY if e.code == "closed" else STAFF_OFFER_SAY)


def _question_for(state: CallState) -> str:
    step = state.bstep
    if step == "name_confirm":
        return q_name_confirm(state.intake.get("name", ""))
    if step == "name_spell":
        return Q_NAME_SPELL
    if step == "first_visit":
        return Q_FIRST_VISIT
    if step == "provider_confirm":
        return f"Is {state.intake.get('provider_heard', '')} your insurance provider, or would you like to self-pay?"
    if step == "slot":
        return "Which of those openings would you like?"
    if step == "confirm":
        return "Is all of that correct?"
    return QUESTIONS.get(step, "")


def _handle(state: CallState, text: str) -> FlowTurn | None:
    step, i = state.bstep, state.intake
    if speech.WANT_STAFF.search(text):
        abort(state, "caller_asked_for_staff")
        return FlowTurn(note="SYSTEM: The caller asked for a person. The booking was stopped. Call request_staff_transfer "
                             "(or take a callback message if the clinic is closed).")
    if speech.CANCEL.search(text):
        abort(state, "caller_cancelled")
        return FlowTurn(note="SYSTEM: The caller cancelled the booking. Ask what else they need.")
    if speech.ALREADY.search(text) and (q := _question_for(state)):
        return FlowTurn(say=f"I'm sorry about that. {q}")

    # ---- first visit? ----------------------------------------------------------------------------------
    if step == "first_visit":
        if speech.EXISTING_PATIENT.search(text) or (speech.yes_no(text) == "no" and not speech.NEW_PATIENT.search(text)):
            state.bstep, state.intake = "idle", {}
            return FlowTurn(say=_verify_first(state).say)
        if speech.NEW_PATIENT.search(text) or speech.yes_no(text) == "yes":
            return FlowTurn(say=begin_new(state))
        return _retry(state, text, "Is this your first visit with us, yes or no?")

    # ---- name (read back and confirmed) --------------------------------------------------------------------
    if step in ("name", "name_spell"):
        status, value = speech.extract_name(text, state.partial_first)
        if status == "ok":
            i["name"], state.partial_first, state.unclear_count, state.bstep = value, None, 0, "name_confirm"
            return FlowTurn(say=q_name_confirm(value))
        if status == "partial":
            state.partial_first = value
            return FlowTurn(say=f"Thanks, {value}. And your last name?")
        return _retry(state, text, "Sorry, I didn't catch your name. Could you tell me your first and last name?"
                      if step == "name" else Q_NAME_SPELL)
    if step == "name_confirm":
        answer = speech.yes_no(text)
        if answer == "yes":
            state.unclear_count = 0
            return _after(state, "name")
        status, value = speech.extract_name(text)
        if status == "ok" and value != i.get("name"):           # "no, it's Jane Doe"
            i["name"], state.unclear_count = value, 0
            return FlowTurn(say=q_name_confirm(value))
        if answer == "no":
            state.bstep, state.partial_first = "name_spell", None
            return _retry(state, "", Q_NAME_SPELL)
        if status == "ok":
            return _after(state, "name")                        # said the same name again: that is a yes
        return _retry(state, text, f"Is {i.get('name')} correct? Please say yes or no.")

    # ---- date of birth -----------------------------------------------------------------------------------------
    if step == "dob":
        text, res, hold = vf.dob_with_fragments(state, text)
        if hold:
            return FlowTurn(silent=True)
        if vf._is_wait(text, res):
            return FlowTurn(say=WAIT_SAY)
        if res.status == "ok":
            i["dob"] = res.iso
            return _after(state, "dob")
        if res.status == "ambiguous":
            a, b = (dates.speak_date(c) for c in res.candidates)
            state.bstep, state.dob_candidates = "dob_clarify", res.candidates
            return FlowTurn(say=f"Just to be sure, is that {a}, or {b}?")
        if res.status == "invalid":
            return _retry(state, text, "I'm not sure I got that date. Could you say it again with the month, day and year?")
        if res.status == "incomplete":
            return _retry(state, text, "I missed part of that. Could you give me the month, day and year?")
        return _retry(state, text, "Sorry, I didn't catch that. What is your date of birth?")
    if step == "dob_clarify":
        iso = dates.resolve_ambiguity(text, state.dob_candidates) or (
            lambda r: r.iso if r.status == "ok" else None)(dates.parse_spoken_date(text))
        if iso:
            i["dob"] = iso
            return _after(state, "dob")
        a, b = (dates.speak_date(c) for c in state.dob_candidates)
        return _retry(state, text, f"Sorry, which was it: {a}, or {b}?")

    # ---- contact details ----------------------------------------------------------------------------------------
    if step == "phone":
        status, number = speech.phone_from_speech(text)
        if status == "ok":
            i["phone"] = number
            return _after(state, "phone")
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        return _retry(state, text, "I need all ten digits. What is the best phone number to reach you?"
                      if status in ("short", "long") else "Sorry, I didn't catch that. What is your phone number?")
    if step == "email":
        local, domain = speech.email_parts(text)
        local, domain = local or i.get("email_local", ""), domain or i.get("email_domain", "")
        if local and domain and svc.normalize_email(f"{local}@{domain}"):
            i["email"] = f"{local}@{domain}"
            i.pop("email_local", None)
            i.pop("email_domain", None)
            return _after(state, "email")
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        if local and not domain:                         # only the part before the "at" was heard: keep it
            i["email_local"] = local
            return FlowTurn(say="Thanks. And the part after the at sign, like gmail dot com?")
        if domain and not local:                         # only the provider was heard: keep it
            i["email_domain"] = domain
            return FlowTurn(say="Thanks. And the part before the at sign? You can spell it letter by letter.")
        tries = state.unclear_count
        return _retry(state, text, "Sorry, I didn't catch that email. Could you say it again?" if tries == 0 else
                      "Let's do it in two parts. Please spell the part before the at sign, letter by letter.", limit=5)

    # ---- insurance ----------------------------------------------------------------------------------------------
    if step == "insurance":
        answer = speech.yes_no(text)
        if speech.SELF_PAY.search(text) or answer == "no":
            i.update(insurance_done=True, provider=None, member=None)
            return _after(state, "insurance")
        if answer == "yes":
            return _goto(state, "provider")
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        if (provider := speech.provider_from_speech(text)):
            return _provider_heard(state, provider)
        return _retry(state, text, "Sorry, do you have insurance, or would you like to self-pay?")
    if step == "provider":
        if speech.SELF_PAY.search(text):
            i.update(insurance_done=True, provider=None, member=None)
            return _after(state, "insurance")
        if (provider := speech.provider_from_speech(text)) and not speech.is_wait(text):
            return _provider_heard(state, provider)
        return _retry(state, text, "Sorry, who is your insurance provider?")
    if step == "provider_confirm":
        heard, answer = i.get("provider_heard", ""), speech.yes_no(text)
        if speech.SELF_PAY.search(text):
            i.pop("provider_heard", None)
            i.update(insurance_done=True, provider=None, member=None)
            return _after(state, "insurance")
        if answer == "yes":
            i["provider"] = i.pop("provider_heard", heard)
            return _goto(state, "member")
        if answer == "no":
            i.pop("provider_heard", None)
            return _goto(state, "provider", "No problem. ")
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        if (other := speech.provider_from_speech(text)) and other.lower() != heard.lower():      # said a different name instead
            return _provider_heard(state, other)
        return _retry(state, text, f"Is {heard} your insurance provider? Please say yes, or say self-pay.")
    if step == "member":
        if (member := speech.member_id_from_speech(text)):
            i["member"], i["insurance_done"] = member, True
            return _after(state, "insurance")
        return _retry(state, text, "Sorry, I didn't catch that. Could you say your member ID again, one character at a time?")

    # ---- reason and preferences ------------------------------------------------------------------------------------
    if step == "reason":
        if (reason := speech.reason_from_speech(text)):
            i["reason"] = reason
            return _after(state, "reason")
        return _retry(state, text, "Just a few words please. What is the visit for?")
    if step == "preference":
        if (pref := speech.preferences_from_speech(text)) is not None:
            i["pref"] = pref
            return _after(state, "preference")
        return _retry(state, text, "Would you like Dr. Patel or Dr. Lee, or a particular day? You can also say no preference.")

    # ---- choose one of the offered slots ------------------------------------------------------------------------------
    if step == "slot":
        offers = i.get("offers", [])
        if speech.wants_other_slots(text):
            i.pop("pref", None)
            return _goto(state, "preference", "No problem. ")
        kind, slot_id = speech.choose_slot(text, offers)
        if kind == "ok":
            i["slot_id"] = slot_id
            return _after(state, "slot")
        if speech.is_wait(text):
            return FlowTurn(say=WAIT_SAY)
        again = " ".join(f"{o['doctor']}, {o['when']}." for o in offers)
        return _retry(state, text, ("Sorry, which one? " if kind == "ambiguous" else "Sorry, I didn't catch which. ") + again)

    # ---- read-back confirmation ---------------------------------------------------------------------------------------
    if step == "confirm":
        answer = speech.yes_no(text)
        if answer == "yes":
            try:
                r = svc.confirm_booking(state)
            except svc.ServiceError as e:
                if e.code in ("slot_taken", "not_confirmed", "nothing_pending", "awaiting_caller_reply"):
                    i.pop("slot_id", None)
                    i["return"] = True
                    return _offer_slots(state, "Sorry, that time is no longer available. ")
                raise
            state.bstep, state.intake = "idle", {}
            return FlowTurn(say=f"You're all set for {r['when']} with {r['doctor']}. {svc.ARRIVAL_INSTRUCTIONS} "
                                "Is there anything else I can help you with?")
        if answer == "no":
            return _change(state, text)
        return _retry(state, text, "Is all of that correct, yes or no?")
    if step == "change":
        return _change(state, text)
    return None


def _change(state: CallState, text: str) -> FlowTurn:
    """The caller said something was wrong: jump straight to that detail, then read everything back again."""
    i = state.intake
    allowed = {"email", "reason", "preference"} if not i.get("new") else {s for _, s in _CHANGE_FIELDS}
    for pattern, step in _CHANGE_FIELDS:
        if step in allowed and re.search(rf"\b({pattern})\b", text.lower()):
            state.pending = None
            if step == "preference":                  # pick again: preference -> slot -> read-back (normal order)
                i.pop("pref", None)
                i.pop("slot_id", None)
            else:                                     # one detail: fix it, then read everything back
                i.pop(step, None)
                i["return"] = True
                if step == "email":
                    i.pop("email", None)
            return _goto(state, step, "No problem. ")
    state.bstep = "change"
    return _retry(state, text if state.unclear_count else "", "What would you like to change? "
                  + ("You can say name, phone, email, reason, or the appointment time."
                     if i.get("new") else "You can say email, reason, or the appointment time."))
