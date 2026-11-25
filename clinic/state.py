"""Per-call state. Verification state lives ONLY here and is changed only by backend code
(clinic.services). The LLM can neither read nor set it directly."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock

MAX_VERIFY_ATTEMPTS = 3


@dataclass
class PendingAction:
    kind: str                     # 'refill' | 'booking' | 'callback'
    payload: dict[str, Any]
    prepared_turn: int
    readback: str


@dataclass
class CallState:
    db_path: str | None = None
    clock: Clock = field(default_factory=Clock)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # verification (backend-owned)
    verified_patient_id: str | None = None
    failed_attempts: int = 0
    locked: bool = False
    pending_name: str | None = None
    pending_dob: str | None = None
    awaiting_ssn: bool = False
    ssn_buffer: str = ""          # digits captured by the STT gate; never sent to the LLM

    # verification conversation (code-owned state machine, see clinic/verify_flow.py)
    vstep: str = "idle"           # idle | name | dob | dob_clarify  (the SSN step is awaiting_ssn)
    partial_first: str | None = None   # first name heard when the last name is still missing
    dob_candidates: list[str] = field(default_factory=list)
    unclear_count: int = 0        # unusable answers at the current step (never resets failed_attempts)

    dob_hint: str | None = None   # a date said together with the name; used after the name is confirmed
    after_verify: str | None = None    # what to continue with once verified: booking | refill | lab | balance | appointment
    dob_frag: str = ""            # a date said in pieces ("My date of birth is" / "March" / "14, 1985"), joined here
    frag_count: int = 0           # pieces held so far at the current step (the agent stays quiet while the caller is mid-answer)
    confirmed_name: str | None = None   # a name the caller confirmed (unverified); a callback does not ask for it again
    refill_step: str = "idle"     # refill conversation (code-owned, see clinic/refill_flow.py): idle | ask
    refill_tries: int = 0         # medication names that could not be matched in a row

    # booking conversation (code-owned state machine, see clinic/booking_flow.py)
    bstep: str = "idle"
    intake: dict[str, Any] = field(default_factory=dict)

    # callback-message conversation (code-owned, see clinic/callback_flow.py)
    cstep: str = "idle"
    callback: dict[str, Any] = field(default_factory=dict)
    choice: str | None = None     # a staff-or-callback offer was just spoken: "staff_or_callback" | "callback_yes"

    # The record created by a NEW-patient booking in THIS call. A new patient has no stored last-four digits, so they can never be
    # "verified"; they may only be told about the record they just created (see services.get_new_patient_info).
    new_patient_id: str | None = None

    emergency: bool = False
    turn: int = 0
    last_user_text: str = ""
    third_party_turn: int = -1    # the turn on which the caller asked about someone else
    pending: PendingAction | None = None
    saved_turn: int = -1          # turn on which the backend auto-saved a confirmed action
    offered_slots: dict[int, dict[str, Any]] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)   # e.g. 'refill requested' (no identifiers)
    saved_keys: set[str] = field(default_factory=set)   # guards against saving the same thing twice
    silence_count: int = 0        # reprompts spoken since the caller last said something (see clinic/silence.py)

    @property
    def verified(self) -> bool:
        return self.verified_patient_id is not None

    def register_user_turn(self, text: str) -> None:
        self.turn += 1
        self.silence_count = 0
        self.last_user_text = text

    def capture_ssn_digits(self, digits: str) -> None:
        self.ssn_buffer += digits

    def clear_ssn(self) -> None:
        self.ssn_buffer = ""
