"""Backend service layer. EVERY rule that protects patient data is enforced here, in code:
verification state, after-hours lockout, emergency lockout, explicit confirmation, slot
integrity. Nothing here trusts the LLM's claims.

Public functions take the per-call CallState and return plain dicts, or raise ServiceError.
None of them accepts a patient id from the caller: they act only on state.verified_patient_id,
so one caller cannot reach another patient's record.
"""

from __future__ import annotations

import difflib
import re
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterator

from . import dates, db, gate
from .clock import HOURS_TEXT
from .security import audit, last4_matches
from .state import MAX_VERIFY_ATTEMPTS, CallState, PendingAction

CLINIC_ADDRESS = "120 Main Street, Springfield"
ARRIVAL_INSTRUCTIONS = (
    "Please arrive 15 minutes early and bring a photo ID and your insurance card. "
    "You'll receive a confirmation by email."
)
HANDOFF_REASONS = {"requested_human", "upset", "confused", "verification_failed", "cannot_complete"}

_MESSAGES = {
    "emergency": "Emergency flagged. Tell the caller to hang up and call 911 now. Do nothing else.",
    "closed": f"The clinic is closed (hours: {HOURS_TEXT}). Do not share or act on any patient information. "
              "Offer to take a callback message instead.",
    "locked": "Verification failed three times and is locked for this call. Do not retry it. "
              "Offer a staff transfer or a callback message.",
    "third_party": "The caller is asking about someone else. Share nothing, not even the caller's own details: say that person "
                   "must call the clinic themselves, or offer a staff callback.",
    "not_verified": "The caller is NOT verified. Share nothing. Collect full name and date of birth, "
                    "call start_verification, then ask for the last four SSN digits.",
}


class ServiceError(Exception):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or _MESSAGES.get(code, code)
        super().__init__(f"{code}: {self.message}")


# ---- guard / connection -------------------------------------------------------------------------
@contextmanager
def operation(state: CallState, action: str, *, need_open: bool = True,
              need_verified: bool = False) -> Iterator[sqlite3.Connection]:
    """Open the DB, enforce emergency / after-hours / verification rules, audit denials."""
    conn = db.connect(state.db_path)
    try:
        code = None
        if state.emergency:
            code = "emergency"
        elif need_open and not state.clock.is_open():
            code = "closed"
        elif need_verified and state.locked:
            code = "locked"
        elif need_verified and not state.verified:
            code = "not_verified"
        elif need_verified and state.third_party_turn == state.turn:
            code = "third_party"
        if code:
            audit(conn, state.call_id, action, f"denied:{code}", state.verified_patient_id)
            raise ServiceError(code)
        yield conn
    finally:
        conn.close()


# ---- parsing helpers ----------------------------------------------------------------------------
def parse_dob(text: str | None) -> str | None:
    """ISO date for any unambiguous natural / numeric date; None otherwise."""
    r = dates.parse_spoken_date(text)
    return r.iso if r.status == "ok" else None


def _name_tokens(name: str | None) -> list[str]:
    return re.sub(r"[^a-z\s'\-]", " ", (name or "").lower()).split()


def normalize_phone(text: str | None) -> str | None:
    d = re.sub(r"\D", "", text or "")
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return f"{d[:3]}-{d[3:6]}-{d[6:]}" if len(d) == 10 else None


def normalize_email(text: str | None) -> str | None:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+at\s+", "@", t)
    t = re.sub(r"\s+dot\s+", ".", t).replace(" ", "")
    return t if re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", t) else None


_COMMON_DOMAINS = {"gmail.com": "gmail dot com", "yahoo.com": "yahoo dot com", "outlook.com": "outlook dot com",
                   "hotmail.com": "hotmail dot com", "icloud.com": "i cloud dot com", "aol.com": "a o l dot com"}


def spell_email(email: str) -> str:
    """Letter-by-letter for the name part; well-known domains are spoken as words."""
    names = {"@": "at", ".": "dot", "-": "dash", "_": "underscore", "+": "plus"}
    local, _, domain = email.partition("@")
    spoken = " ".join(names.get(c, c) for c in local)
    if domain.lower() in _COMMON_DOMAINS:
        return f"{spoken} at {_COMMON_DOMAINS[domain.lower()]}"
    return " ".join(names.get(c, c) for c in email)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def spoken_when(starts_at: str) -> str:
    dt = datetime.fromisoformat(starts_at)
    return f"{dt:%A, %B} {_ordinal(dt.day)} at {dt.strftime('%I:%M %p').lstrip('0')}"


def _insert_request(conn: sqlite3.Connection, patient_id: str, rtype: str, summary: str, status: str) -> int:
    return conn.execute(
        "INSERT INTO requests (patient_id, request_type, created_at, summary, status) VALUES (?,?,?,?,?)",
        (patient_id, rtype, db.now_iso(), summary, status),
    ).lastrowid


# ---- verification ---------------------------------------------------------------------------------
def begin_verification(state: CallState, full_name: str, dob_text: str) -> dict[str, Any]:
    """Store name + DOB and wait for the SSN digits (captured by the STT gate, not the LLM)."""
    with operation(state, "begin_verification") as conn:
        if state.locked:
            audit(conn, state.call_id, "begin_verification", "denied:locked")
            raise ServiceError("locked")
        if state.verified:
            return {"status": "already_verified"}
        dob = parse_dob(dob_text)
        if len(_name_tokens(full_name)) < 2 or not dob:
            audit(conn, state.call_id, "begin_verification", "bad_input")
            return {"status": "bad_input"}
        state.pending_name, state.pending_dob, state.awaiting_ssn = full_name, dob, True
        state.vstep, state.unclear_count = "idle", 0
        audit(conn, state.call_id, "begin_verification", "awaiting_ssn")
    # the caller may already have said the digits ("...my social ends in 3344")
    if len(state.ssn_buffer) == 4:
        return complete_verification(state)
    state.clear_ssn()
    return {"status": "awaiting_ssn"}


def complete_verification(state: CallState) -> dict[str, Any]:
    """Match name + DOB + last 4 against ONE patient record. Runs in code only."""
    last4, name, dob = state.ssn_buffer, state.pending_name, state.pending_dob
    state.clear_ssn()  # digits are used once and discarded
    state.awaiting_ssn = False
    state.pending_name = state.pending_dob = None
    with operation(state, "verify_identity", need_open=True) as conn:
        if state.locked:
            raise ServiceError("locked")
        tokens = _name_tokens(name)
        matched = None
        if len(last4) == 4 and tokens and dob:
            for p in conn.execute("SELECT * FROM patients WHERE is_new = 0 AND dob = ?", (dob,)):
                if (tokens[0] == p["first_name"].lower() and tokens[-1] == p["last_name"].lower()
                        and last4_matches(last4, p["ssn_last4_hash"])):
                    matched = p["patient_id"]
        if matched:
            state.verified_patient_id = matched
            audit(conn, state.call_id, "verify_identity", "success", matched)
            return {"status": "verified"}
        state.failed_attempts += 1
        if state.failed_attempts >= MAX_VERIFY_ATTEMPTS:
            state.locked = True
        # never record which field was wrong, and never a patient id on failure
        audit(conn, state.call_id, "verify_identity", f"failed (attempt {state.failed_attempts})")
        return {"status": "locked" if state.locked else "failed",
                "attempts_left": max(0, MAX_VERIFY_ATTEMPTS - state.failed_attempts)}


# ---- confirmation ---------------------------------------------------------------------------------
def _stage(state: CallState, kind: str, payload: dict[str, Any], readback: str) -> None:
    state.pending = PendingAction(kind, payload, state.turn, readback)


def _require_confirmation(state: CallState, kind: str) -> PendingAction:
    p = state.pending
    if p is None or p.kind != kind:
        raise ServiceError("nothing_pending", "Nothing is waiting for confirmation. Start over with the prepare step.")
    if state.turn <= p.prepared_turn:
        raise ServiceError("awaiting_caller_reply",
                           "The caller has not replied since the read-back. Read it back and wait for their answer.")
    if not gate.is_affirmative(state.last_user_text):
        raise ServiceError("not_confirmed",
                           "The caller did not clearly say yes. Ask what to change, or cancel_pending_request.")
    return p


def cancel_pending(state: CallState) -> dict[str, Any]:
    with operation(state, "cancel_pending", need_open=False) as conn:
        kind = state.pending.kind if state.pending else None
        state.pending = None
        audit(conn, state.call_id, "cancel_pending", kind or "nothing_pending", state.verified_patient_id)
    return {"cancelled": kind}


# ---- verified-patient lookups -----------------------------------------------------------------------
def get_next_appointment(state: CallState) -> dict[str, Any]:
    with operation(state, "get_next_appointment", need_verified=True) as conn:
        now = state.clock.now().replace(tzinfo=None).isoformat(timespec="minutes")
        row = conn.execute(
            "SELECT doctor, starts_at FROM appointments WHERE patient_id=? AND status='scheduled'"
            " AND starts_at >= ? ORDER BY starts_at LIMIT 1", (state.verified_patient_id, now)).fetchone()
        audit(conn, state.call_id, "get_next_appointment", "found" if row else "none", state.verified_patient_id)
        return {"found": True, "doctor": row["doctor"], "when": spoken_when(row["starts_at"])} if row else {"found": False}


def get_lab_status(state: CallState) -> dict[str, Any]:
    with operation(state, "get_lab_status", need_verified=True) as conn:
        row = conn.execute("SELECT lab_status FROM patients WHERE patient_id=?", (state.verified_patient_id,)).fetchone()
        audit(conn, state.call_id, "get_lab_status", "allowed", state.verified_patient_id)
        return {"status": row["lab_status"] or "No lab information on file"}


def get_balance(state: CallState) -> dict[str, Any]:
    with operation(state, "get_balance", need_verified=True) as conn:
        row = conn.execute("SELECT balance_due FROM patients WHERE patient_id=?", (state.verified_patient_id,)).fetchone()
        audit(conn, state.call_id, "get_balance", "allowed", state.verified_patient_id)
        return {"balance": float(row["balance_due"] or 0.0)}


# ---- refills ---------------------------------------------------------------------------------------
MED_MATCH_MIN = 0.55       # speech-to-text mangles drug names ("Lisey Nuperell"); the caller confirms the read-back anyway
MED_MATCH_MARGIN = 0.15    # the best match must beat the runner-up by this much, otherwise it is ambiguous


def _med_score(hint: str, name: str) -> float:
    # similarity (0..1) of what was heard, spoken or spelled letter by letter, to a medication's first word
    first = re.sub(r"[^a-z]", "", name.split()[0].lower())
    candidates = [re.sub(r"[^a-z]", "", hint)] + [re.sub(r"[^a-z]", "", w) for w in hint.split()]
    return max((difflib.SequenceMatcher(None, c, first).ratio() for c in candidates if len(c) >= 3), default=0.0)


def _select_medications(hint: str, meds: list) -> list:
    """The caller's own medications that match `hint`: exact / prefix first, then a fuzzy match. Never other patients' data."""
    first = hint.split()[0]
    hits = [m for m in meds if hint in m["name"].lower() or m["name"].lower().startswith(first)]
    if hits:
        return hits
    scored = sorted(((_med_score(hint, m["name"]), m) for m in meds), key=lambda x: -x[0])
    if not scored or scored[0][0] < MED_MATCH_MIN:
        return []
    if len(scored) > 1 and scored[0][0] - scored[1][0] < MED_MATCH_MARGIN:
        return [scored[0][1], scored[1][1]]
    return [scored[0][1]]


def has_medication(state: CallState, medication_hint: str) -> bool:
    """True when exactly one of the verified caller's medications matches (no audit row, nothing staged)."""
    with operation(state, "check_medication", need_verified=True) as conn:
        hint = (medication_hint or "").strip().lower()
        meds = conn.execute("SELECT * FROM medications WHERE patient_id=?", (state.verified_patient_id,)).fetchall()
        return bool(hint) and len(_select_medications(hint, meds)) == 1


def prepare_refill(state: CallState, medication_hint: str) -> dict[str, Any]:
    with operation(state, "prepare_refill", need_verified=True) as conn:
        hint = (medication_hint or "").strip().lower()
        if not hint:
            raise ServiceError("need_medication", "Ask which medication they need refilled. Do not list their medications.")
        meds = conn.execute("SELECT * FROM medications WHERE patient_id=?", (state.verified_patient_id,)).fetchall()
        hits = _select_medications(hint, meds)
        if not hits:
            audit(conn, state.call_id, "prepare_refill", "not_on_file", state.verified_patient_id)
            raise ServiceError("not_on_file", "That medication is not on file. Do not guess. Offer a staff callback.")
        if len(hits) > 1:
            raise ServiceError("ambiguous", "More than one medication matches. Ask the caller to say the full name.")
        m = hits[0]
        needs_approval = m["refills_remaining"] <= 0
        readback = f"{m['name']} at {m['pharmacy']}"
        _stage(state, "refill", {"med_id": m["id"], "name": m["name"], "pharmacy": m["pharmacy"],
                                 "needs_approval": needs_approval, "refills": m["refills_remaining"]}, readback)
        audit(conn, state.call_id, "prepare_refill", "staged", state.verified_patient_id)
        return {"medication": m["name"], "pharmacy": m["pharmacy"], "needs_doctor_approval": needs_approval}


def confirm_refill(state: CallState) -> dict[str, Any]:
    with operation(state, "confirm_refill", need_verified=True) as conn:
        p = _require_confirmation(state, "refill")
        d = p.payload
        key = f"refill:{state.verified_patient_id}:{d['med_id']}"
        if key in state.saved_keys:
            state.pending = None
            raise ServiceError("already_saved", "This refill was already requested in this call. Do not save it again; "
                                                "just tell the caller it is done.")
        status = "pending_doctor_approval" if d["needs_approval"] else "submitted_to_pharmacy"
        summary = f"Refill request: {d['name']} at {d['pharmacy']} ({d['refills']} refills remaining)"
        with db.transaction(conn):
            rid = _insert_request(conn, state.verified_patient_id, "refill", summary, status)
            audit(conn, state.call_id, "confirm_refill", status, state.verified_patient_id, f"request {rid}")
        state.pending = None
        state.saved_keys.add(key)
        state.completed.append("refill requested")
        return {"request_id": rid, "needs_doctor_approval": d["needs_approval"], "medication": d["name"]}


# ---- booking ----------------------------------------------------------------------------------------
def find_slots(state: CallState, doctor: str = "", weekday: str = "", time_of_day: str = "") -> dict[str, Any]:
    """Real open slots from the database, never invented. Returns at most three."""
    with operation(state, "find_slots") as conn:
        now = state.clock.now().replace(tzinfo=None).isoformat(timespec="minutes")
        rows = conn.execute("SELECT * FROM available_slots WHERE starts_at > ? ORDER BY starts_at", (now,)).fetchall()

        def ok(r: sqlite3.Row) -> bool:
            dt = datetime.fromisoformat(r["starts_at"])
            return ((not doctor or doctor.lower().replace("dr.", "").strip() in r["doctor"].lower())
                    and (not weekday or dt.strftime("%A").lower() == weekday.strip().lower())
                    and (not time_of_day or (time_of_day.lower().startswith("morn") and dt.hour < 12)
                         or (time_of_day.lower().startswith("after") and dt.hour >= 12)))

        matches = [r for r in rows if ok(r)]
        matched = bool(matches)
        chosen = (matches or rows)[:3]
        state.offered_slots = {r["id"]: dict(r) for r in chosen}
        audit(conn, state.call_id, "find_slots", f"{len(chosen)} offered", state.verified_patient_id)
        return {"matched_preference": matched, "slots": [
            {"slot_id": r["id"], "doctor": r["doctor"], "when": spoken_when(r["starts_at"])} for r in chosen]}


def _slot(conn: sqlite3.Connection, state: CallState, slot_id: int) -> dict[str, Any]:
    if slot_id not in state.offered_slots:
        raise ServiceError("slot_not_offered", "That slot was not offered in this call. Call find_appointment_slots first.")
    row = conn.execute("SELECT * FROM available_slots WHERE id=?", (slot_id,)).fetchone()
    if row is None:
        state.offered_slots.pop(slot_id, None)
        raise ServiceError("slot_taken", "That slot was just taken. Call find_appointment_slots again.")
    return dict(row)


def _short_reason(reason: str) -> str:
    r = (reason or "").strip()
    if not r:
        raise ServiceError("need_reason", "Ask for a short reason for the visit (a few words).")
    if len(r) > 80:
        raise ServiceError("reason_too_long", "Ask for the reason in just a few words. No symptoms or history.")
    return r


def prepare_booking_existing(state: CallState, slot_id: int, reason: str, email: str = "") -> dict[str, Any]:
    with operation(state, "prepare_booking_existing", need_verified=True) as conn:
        slot = _slot(conn, state, slot_id)
        reason = _short_reason(reason)
        pat = conn.execute("SELECT first_name, last_name, email FROM patients WHERE patient_id=?",
                           (state.verified_patient_id,)).fetchone()
        mail = normalize_email(email) if email else pat["email"]
        if not mail:
            raise ServiceError("need_email", "No valid email on file. Ask for the caller's email address.")
        name = f"{pat['first_name']} {pat['last_name']}"
        readback = f"{name}, {spoken_when(slot['starts_at'])}, with {slot['doctor']}, for {reason}"
        _stage(state, "booking", {"new": False, "slot_id": slot_id, "email": mail, "reason": reason, "name": name}, readback)
        audit(conn, state.call_id, "prepare_booking_existing", "staged", state.verified_patient_id)
        return {"readback": readback, "email_spelled": spell_email(mail)}


def prepare_booking_new(state: CallState, slot_id: int, full_name: str, dob_text: str, phone: str, email: str,
                        insurance_provider: str, member_id: str, reason: str) -> dict[str, Any]:
    with operation(state, "prepare_booking_new") as conn:
        if state.verified:
            raise ServiceError("already_verified", "This caller is a verified patient. Use prepare_booking_for_verified_patient.")
        slot = _slot(conn, state, slot_id)
        reason = _short_reason(reason)
        dres = dates.parse_spoken_date(dob_text)
        dob, ph, mail = dres.iso if dres.status == "ok" else None, normalize_phone(phone), normalize_email(email)
        if len(_name_tokens(full_name)) < 2:
            raise ServiceError("bad_name", "Ask for the caller's first and last name.")
        if not dob:
            hint = ("The date can be read two ways. Ask the caller to say the month by name."
                    if dres.status == "ambiguous" else
                    "Ask for the date of birth again, with the month, day and year. Never mention a date format.")
            raise ServiceError("bad_dob", hint)
        if not ph:
            raise ServiceError("bad_phone", "Ask for a 10-digit phone number.")
        if not mail:
            raise ServiceError("bad_email", "Ask for the email address again and spell it back.")
        self_pay = (insurance_provider or "").strip().lower() in {"self-pay", "self pay", "selfpay", "none"} and not member_id
        if not self_pay and not (insurance_provider.strip() and (member_id or "").strip()):
            raise ServiceError("need_insurance", "Ask for insurance provider and member ID, or say self-pay.")
        tokens = _name_tokens(full_name)
        dup = conn.execute("SELECT 1 FROM patients WHERE dob=? AND lower(first_name)=? AND lower(last_name)=?",
                           (dob, tokens[0], tokens[-1])).fetchone()
        if dup:  # do not reveal that a record exists
            audit(conn, state.call_id, "prepare_booking_new", "duplicate_blocked")
            raise ServiceError("cannot_complete", "This booking cannot be completed by phone. Offer a staff callback. "
                                                  "Do not say why.")
        name = " ".join(w.capitalize() for w in tokens)
        readback = f"{name}, {spoken_when(slot['starts_at'])}, with {slot['doctor']}, for {reason}"
        _stage(state, "booking", {"new": True, "slot_id": slot_id, "name": name, "dob": dob, "phone": ph, "email": mail,
                                  "insurance": None if self_pay else insurance_provider.strip(),
                                  "member_id": None if self_pay else member_id.strip(), "self_pay": self_pay,
                                  "reason": reason}, readback)
        audit(conn, state.call_id, "prepare_booking_new", "staged")
        return {"readback": readback, "email_spelled": spell_email(mail), "self_pay": self_pay}


def confirm_booking(state: CallState) -> dict[str, Any]:
    with operation(state, "confirm_booking", need_verified=state.pending is not None and not state.pending.payload.get("new", True)) as conn:
        p = _require_confirmation(state, "booking")
        d = p.payload
        with db.transaction(conn):
            slot = conn.execute("SELECT * FROM available_slots WHERE id=?", (d["slot_id"],)).fetchone()
            if slot is None or conn.execute("DELETE FROM available_slots WHERE id=?", (d["slot_id"],)).rowcount != 1:
                state.pending = None
                state.offered_slots.pop(d["slot_id"], None)
                raise ServiceError("slot_taken", "That slot was just taken. Call find_appointment_slots again.")
            if d["new"]:
                n = conn.execute("SELECT COUNT(*) FROM patients WHERE is_new=1").fetchone()[0] + 1
                pid = f"N{n:03d}"
                first, _, last = d["name"].partition(" ")
                conn.execute(
                    "INSERT INTO patients (patient_id, first_name, last_name, dob, phone, email, insurance_provider,"
                    " member_id, self_pay, is_new, created_at) VALUES (?,?,?,?,?,?,?,?,?,1,?)",
                    (pid, first, last.strip() or first, d["dob"], d["phone"], d["email"], d["insurance"],
                     d["member_id"], int(d["self_pay"]), db.now_iso()))
            else:
                pid = state.verified_patient_id
                conn.execute("UPDATE patients SET email=? WHERE patient_id=? AND (email IS NULL OR email='')",
                             (d["email"], pid))
            appt = conn.execute(
                "INSERT INTO appointments (patient_id, doctor, starts_at, reason, created_at) VALUES (?,?,?,?,?)",
                (pid, slot["doctor"], slot["starts_at"], d["reason"], db.now_iso())).lastrowid
            when = spoken_when(slot["starts_at"])
            _insert_request(conn, pid, "booking", f"Appointment with {slot['doctor']} on {slot['starts_at']} ({d['reason']})",
                            "confirmed")
            conn.execute(
                "INSERT INTO email_outbox (to_email, subject, body, created_at, appointment_id) VALUES (?,?,?,?,?)",
                (d["email"], "Your appointment at Sunrise Family Clinic",
                 f"Hello {d['name']},\n\nYour appointment is confirmed: {when} with {slot['doctor']}.\n"
                 f"Location: {CLINIC_ADDRESS}.\n{ARRIVAL_INSTRUCTIONS}\n\nSunrise Family Clinic",
                 db.now_iso(), appt))
            audit(conn, state.call_id, "confirm_booking", "booked", pid, f"appointment {appt}")
        state.pending = None
        state.offered_slots.pop(d["slot_id"], None)
        state.completed.append("appointment booked")
        if d["new"]:
            state.new_patient_id = pid
        return {"patient_id": pid, "appointment_id": appt, "when": when, "doctor": slot["doctor"], "new_patient": d["new"]}


def get_new_patient_info(state: CallState, topic: str) -> dict[str, Any]:
    """What a caller who registered as a NEW patient in this call may be told. Only the record created in this call is ever read
    (never another patient's), and a new patient has no labs, balance or medications on file yet, so nothing sensitive is disclosed.
    The caller is not marked as verified: every verified-only tool still refuses them."""
    with operation(state, f"new_patient_{topic}") as conn:                 # emergency / after-hours rules still apply
        if state.verified or not state.new_patient_id or topic not in ("lab", "balance", "appointment", "refill"):
            audit(conn, state.call_id, f"new_patient_{topic}", "denied:not_a_new_patient_in_this_call")
            raise ServiceError("not_verified")
        if state.third_party_turn == state.turn:
            audit(conn, state.call_id, f"new_patient_{topic}", "denied:third_party", state.new_patient_id)
            raise ServiceError("third_party")
        result: dict[str, Any] = {"topic": topic}
        if topic == "appointment":
            now = state.clock.now().replace(tzinfo=None).isoformat(timespec="minutes")
            row = conn.execute("SELECT doctor, starts_at FROM appointments WHERE patient_id=? AND status='scheduled' AND starts_at >= ? "
                               "ORDER BY starts_at LIMIT 1", (state.new_patient_id, now)).fetchone()
            if row:
                result.update(found=True, doctor=row["doctor"], when=spoken_when(row["starts_at"]))
            else:
                result["found"] = False
        audit(conn, state.call_id, f"new_patient_{topic}", "answered", state.new_patient_id)
        return result


# ---- callbacks, handoff, emergency ---------------------------------------------------------------
def prepare_callback(state: CallState, caller_name: str, callback_number: str, reason: str) -> dict[str, Any]:
    with operation(state, "prepare_callback", need_open=False) as conn:
        num = normalize_phone(callback_number)
        reason = (reason or "").strip()
        if len(_name_tokens(caller_name)) < 1:
            raise ServiceError("bad_name", "Ask for the caller's name.")
        if not num:
            raise ServiceError("bad_phone", "Ask for a 10-digit callback number.")
        if not reason or len(reason) > 120:
            raise ServiceError("bad_reason", "Ask for the reason in a few words. No detailed health information.")
        readback = f"{caller_name.strip()}, callback number {num}, regarding {reason}"
        _stage(state, "callback", {"name": caller_name.strip(), "number": num, "reason": reason}, readback)
        audit(conn, state.call_id, "prepare_callback", "staged", state.verified_patient_id)
        return {"readback": readback}


def confirm_callback(state: CallState) -> dict[str, Any]:
    with operation(state, "confirm_callback", need_open=False) as conn:
        p = _require_confirmation(state, "callback")
        d = p.payload
        pid = state.verified_patient_id or "UNVERIFIED"
        key = f"callback:{d['number']}:{d['reason']}"
        if key in state.saved_keys:
            state.pending = None
            raise ServiceError("already_saved", "This message was already saved in this call. Do not save it again; "
                                                "just tell the caller it is done.")
        follow = state.clock.next_business_day().isoformat()
        with db.transaction(conn):
            rid = _insert_request(conn, pid, "callback", f"Callback requested: {d['reason']}", "open")
            conn.execute(
                "INSERT INTO callback_messages (request_id, patient_id, caller_name, callback_number, reason, created_at,"
                " followup_date) VALUES (?,?,?,?,?,?,?)", (rid, pid, d["name"], d["number"], d["reason"], db.now_iso(), follow))
            audit(conn, state.call_id, "confirm_callback", "saved", state.verified_patient_id, f"request {rid}")
        state.pending = None
        state.saved_keys.add(key)
        state.completed.append("callback message saved")
        return {"request_id": rid, "followup_date": follow}


def request_handoff(state: CallState, reason: str) -> dict[str, Any]:
    """Simulated transfer: records a staff-assist request. No live transfer infrastructure exists."""
    code = reason if reason in HANDOFF_REASONS else "cannot_complete"
    with operation(state, "request_handoff") as conn:
        pid = state.verified_patient_id or "UNVERIFIED"
        with db.transaction(conn):
            rid = _insert_request(conn, pid, "handoff", f"Staff assistance requested ({code}); simulated transfer",
                                  "transfer_simulated")
            audit(conn, state.call_id, "request_handoff", code, state.verified_patient_id, f"request {rid}")
        return {"request_id": rid}


def flag_emergency(state: CallState, source: str) -> None:
    """Lock the call out of every normal workflow. source: 'gate' (regex) or 'llm'."""
    first = not state.emergency
    state.emergency = True
    state.pending = None
    if first:
        conn = db.connect(state.db_path)
        try:
            audit(conn, state.call_id, "emergency_detected", source, state.verified_patient_id)
        finally:
            conn.close()
