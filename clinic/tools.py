"""Agent-facing tool functions. Each takes the CallState first, calls the backend service layer
and returns a short instruction string for the LLM. None takes a patient id: tools can only
ever act on the caller verified in `state`.

Backend refusals are returned as "NOT ALLOWED: ..." text (the LLM relays them politely).
"""

from __future__ import annotations

from typing import Callable

from . import services as svc
from . import booking_flow, callback_flow, verify_flow
from .clock import HOURS_TEXT
from .state import CallState

DOCTORS = "Dr. Patel (Family Medicine) and Dr. Lee (Internal Medicine)"


def _safe(fn: Callable[[], str]) -> str:
    try:
        return fn()
    except svc.ServiceError as e:
        return f"NOT ALLOWED ({e.code}): {e.message}"


def _already_saved(state: CallState) -> str | None:
    if state.pending is None and state.saved_turn == state.turn:
        return "Already saved by the system this turn. Just confirm it to the caller; do not call this again."
    return None


def check_clinic_hours(state: CallState) -> str:
    now = state.clock.now()
    status = "OPEN" if state.clock.is_open() else "CLOSED"
    return (f"The clinic is {status} right now ({now:%A %I:%M %p} Eastern). Hours: {HOURS_TEXT}. "
            f"Doctors: {DOCTORS}. Address: {svc.CLINIC_ADDRESS}.")


def start_verification(state: CallState, full_name: str = "") -> str:
    """Text-only variant (used by tests and tool-level callers). The live agent speaks flow.say itself."""
    def run() -> str:
        r = verify_flow.start(state, full_name)
        return r.text if r.silent or not r.say else f"Verification started. The system asks: {r.say}"
    return _safe(run)


def start_booking(state: CallState, first_visit: str = "") -> str:
    """Text-only variant (tests / tool-level callers). The live agent speaks the first question itself."""
    def run() -> str:
        r = booking_flow.start(state, first_visit)
        return r.text if r.silent or not r.say else f"Booking started. The system asks: {r.say}"
    return _safe(run)


def start_callback(state: CallState, caller_name: str = "") -> str:
    """Text-only variant (tests / tool-level callers). The live agent speaks the first question itself."""
    def run() -> str:
        r = callback_flow.start(state, caller_name)
        return r.text if r.silent or not r.say else f"Message started. The system asks: {r.say}"
    return _safe(run)


def get_next_appointment(state: CallState) -> str:
    def run() -> str:
        r = svc.get_next_appointment(state)
        if r["found"]:
            return f"Next appointment: {r['when']} with {r['doctor']}. Tell the caller."
        return "No upcoming appointment on file. Offer to book one, or a staff callback."
    return _safe(run)


def prepare_refill(state: CallState, medication_name: str) -> str:
    def run() -> str:
        r = svc.prepare_refill(state, medication_name)
        extra = (" NO refills remain: tell the caller this will be sent to the doctor for approval. Never say it is approved."
                 if r["needs_doctor_approval"] else "")
        return (f"Read back: {r['medication']} at {r['pharmacy']}. Ask 'Is that right?' and wait for a yes, "
                f"then call confirm_refill.{extra}")
    return _safe(run)


def confirm_refill(state: CallState) -> str:
    if (done := _already_saved(state)):
        return done

    def run() -> str:
        r = svc.confirm_refill(state)
        if r["needs_doctor_approval"]:
            return (f"Saved (request {r['request_id']}). Say: no refills remain, so a request has been sent to the doctor "
                    "for approval. Never say it is approved. Ask if there is anything else.")
        return (f"Saved (request {r['request_id']}). Say the refill request has been submitted and the clinic will process "
                "it. Never say it is approved or ready. Ask if there is anything else.")
    return _safe(run)


def get_lab_status(state: CallState) -> str:
    def run() -> str:
        r = svc.get_lab_status(state)
        return (f"Lab status: {r['status']}. Tell the caller ONLY this status. Never read, explain or interpret results. "
                "For clinical questions offer to send a message to the doctor.")
    return _safe(run)


def get_balance_due(state: CallState) -> str:
    def run() -> str:
        b = svc.get_balance(state)["balance"]
        return ("Balance due: no balance. Tell the caller." if b == 0
                else f"Balance due: ${b:,.2f}. Tell the caller this amount only.")
    return _safe(run)


def find_appointment_slots(state: CallState, preferred_doctor: str = "", preferred_weekday: str = "",
                           preferred_time_of_day: str = "") -> str:
    def run() -> str:
        r = svc.find_slots(state, preferred_doctor, preferred_weekday, preferred_time_of_day)
        if not r["slots"]:
            return "No open slots. Say so and offer a staff callback."
        lines = [f"Option {i} (slot_id {s['slot_id']}): {s['doctor']}, {s['when']}" for i, s in enumerate(r["slots"], 1)]
        note = "" if r["matched_preference"] else " None matched the caller's preference: say so, then offer these nearest ones."
        return "Offer ONLY these (never invent others):\n" + "\n".join(lines) + note
    return _safe(run)


def prepare_booking_for_verified_patient(state: CallState, slot_id: int, reason_for_visit: str, email: str = "") -> str:
    def run() -> str:
        r = svc.prepare_booking_existing(state, int(slot_id), reason_for_visit, email)
        return (f"Read back: {r['readback']}. Then spell the email: {r['email_spelled']}. Ask 'Is all of that correct?' "
                "and wait for a yes, then call confirm_booking.")
    return _safe(run)


def prepare_new_patient_booking(state: CallState, slot_id: int, full_name: str, date_of_birth: str, phone: str,
                                email: str, insurance_provider: str, member_id: str, reason_for_visit: str) -> str:
    def run() -> str:
        r = svc.prepare_booking_new(state, int(slot_id), full_name, date_of_birth, phone, email,
                                    insurance_provider, member_id, reason_for_visit)
        return (f"Read back: {r['readback']}. Then spell the email: {r['email_spelled']}. Ask 'Is all of that correct?' "
                "and wait for a yes, then call confirm_booking.")
    return _safe(run)


def confirm_booking(state: CallState) -> str:
    if (done := _already_saved(state)):
        return done

    def run() -> str:
        r = svc.confirm_booking(state)
        return (f"BOOKED: {r['when']} with {r['doctor']}. Tell the caller it is confirmed. Then say: {svc.ARRIVAL_INSTRUCTIONS} "
                "Ask if there is anything else.")
    return _safe(run)


def prepare_callback_message(state: CallState, caller_name: str, callback_number: str, reason: str) -> str:
    def run() -> str:
        r = svc.prepare_callback(state, caller_name, callback_number, reason)
        return (f"Read back: {r['readback']}. Ask 'Is that correct?' and wait for a yes, then call confirm_callback_message.")
    return _safe(run)


def confirm_callback_message(state: CallState) -> str:
    if (done := _already_saved(state)):
        return done

    def run() -> str:
        r = svc.confirm_callback(state)
        if state.clock.is_open():
            closing = "Say: your message has been recorded and the team will follow up on the next business day."
        else:
            closing = ("Say: Your message has been recorded, and our team will get back to you on the next business day. "
                       "If this is an emergency, please hang up and call 911.")
        return f"Saved (request {r['request_id']}, follow-up {r['followup_date']}). {closing}"
    return _safe(run)


def request_staff_transfer(state: CallState, reason: str) -> str:
    def run() -> str:
        svc.request_handoff(state, reason)
        return ("Staff assistance requested (this prototype simulates the transfer). Tell the caller a staff member has been "
                "asked to help and will pick this up. Do not promise a time.")
    return _safe(run)


def cancel_pending_request(state: CallState) -> str:
    def run() -> str:
        svc.cancel_pending(state)
        return "Cancelled. Nothing was saved. Ask what the caller would like to do."
    return _safe(run)


def report_emergency(state: CallState) -> str:
    svc.flag_emergency(state, "llm")
    return ("Emergency flagged. Tell the caller to hang up and call 911 now or go to the nearest emergency room. "
            "Do not continue any other request.")


# ---- templated spoken answers ------------------------------------------------------------------------------------------
# The LLM decides WHICH tool to call; for simple facts the sentence is written by code. That halves the tokens for these
# turns (no second LLM request) and the model can never misread an amount, status or date.
MORE = " Is there anything else I can help you with?"
NO_APPOINTMENT_SAY = ("I don't see an upcoming appointment on file for you. "
                      "Would you like to book one, or would you rather leave a callback message?")


def _denied(e: svc.ServiceError) -> tuple[None, str]:
    return None, f"NOT ALLOWED ({e.code}): {e.message}"


def lab_sentence(status: str) -> str:
    if status.lower().startswith("no "):
        return "I don't see any recent lab results on file."
    head, _, rest = status.partition(" - ")
    sentence = f"Your lab results are {head.strip().lower()}."
    return f"{sentence} {rest.strip()[:1].upper()}{rest.strip()[1:]}." if rest.strip() else sentence


def speak_balance(state: CallState) -> tuple[str | None, str]:
    try:
        b = svc.get_balance(state)["balance"]
    except svc.ServiceError as e:
        return _denied(e)
    return ("You have no balance due." if b == 0 else f"Your balance due is ${b:,.2f}.") + MORE, ""


def speak_lab_status(state: CallState) -> tuple[str | None, str]:
    try:
        status = svc.get_lab_status(state)["status"]
    except svc.ServiceError as e:
        return _denied(e)
    return lab_sentence(status) + MORE, ""


def speak_next_appointment(state: CallState) -> tuple[str | None, str]:
    try:
        r = svc.get_next_appointment(state)
    except svc.ServiceError as e:
        return _denied(e)
    if r["found"]:
        return f"Your next appointment is {r['when']} with {r['doctor']}." + MORE, ""
    state.choice, state.unclear_count = "book_or_callback", 0         # the answer is handled in code (callback_flow.handle_choice)
    return NO_APPOINTMENT_SAY, ""


def speak_new_patient(state: CallState, topic: str) -> tuple[str | None, str]:
    """A caller who registered as a new patient earlier in this call asks about labs / balance / appointment / refills. No
    verification is asked for again (it could never succeed: no last-four digits are collected from new patients) and nothing
    about anyone else is read."""
    try:
        r = svc.get_new_patient_info(state, topic)
    except svc.ServiceError as e:
        return _denied(e)
    if topic == "appointment" and r.get("found"):
        return f"Your next appointment is {r['when']} with {r['doctor']}." + MORE, ""
    say = {"lab": "Since this is your first visit, I don't have any lab results on file for you yet.",
           "balance": "Since this is your first visit, there is no balance on your account.",
           "appointment": "I don't see an upcoming appointment for you.",
           "refill": "Since this is your first visit, I don't have any medications on file for you yet. "
                     "Your doctor can go over that with you at your appointment."}[topic]
    return say + MORE, ""


def speak_refill_readback(state: CallState, medication_name: str) -> tuple[str | None, str]:
    try:
        r = svc.prepare_refill(state, medication_name)
    except svc.ServiceError as e:
        return _denied(e)
    return refill_readback_say(r), ""


def refill_readback_say(r: dict) -> str:
    note = " There are no refills left, so this would go to the doctor for approval." if r["needs_doctor_approval"] else ""
    return f"{r['medication']} at {r['pharmacy']}.{note} Is that right?"


def refill_done_say(needs_doctor_approval: bool) -> str:
    if needs_doctor_approval:
        return "Done. There are no refills left, so I've sent the request to your doctor for approval." + MORE
    return "Done. Your refill request has been submitted, and the clinic will process it." + MORE
