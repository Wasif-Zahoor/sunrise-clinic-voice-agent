"""Per-call text transcripts for reviewing the agent after a test call.  Not a medical record.

One readable .txt file per call in transcripts/ (git-ignored): call id, start and end times, and timestamped CALLER / AGENT
lines in order. Interruptions and errors are marked. The file is flushed after every line, so a call that ends abnormally
still leaves a readable (unfinished) file, and nothing here can raise into the voice session.

PRIVACY (healthcare demo, synthetic patients only). Two independent layers:
  1. Step-based: while the agent is collecting a sensitive answer (name, date of birth, last four SSN digits, phone, email,
     insurance, reason) the caller's words are NOT stored at all; the line says "[date of birth provided - not stored]".
     The credentials never reach the file, whatever the speech-to-text wrote.
  2. Pattern-based redaction of everything else (agent lines and free-form caller lines): SSNs, phone numbers, emails (written
     or spelled out), dates that include a year, dollar amounts, long digit runs, insurance-style IDs, medication names and doses,
     and every patient name known to the database or captured during the call.
LIMITS: a name or detail said in free-form speech BEFORE the agent has captured it cannot be recognised (for example a new
patient's name in an opening sentence). The audio itself and the LLM/STT/TTS vendors still receive the real words; this file is
a redacted copy for debugging, not a privacy boundary for the call.
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from . import config, db, gate, speech

logger = logging.getLogger("clinic.transcript")

# ---- what the caller is about to say, from the conversation state --------------------------------------------------------
_VERIFY_STEPS = {"name": "name", "name_spell": "name", "dob": "date of birth", "dob_clarify": "date of birth"}
_BOOKING_STEPS = {"name": "name", "name_spell": "name", "dob": "date of birth", "dob_clarify": "date of birth",
                  "phone": "phone number", "email": "email address", "insurance": "insurance details",
                  "provider": "insurance details", "provider_confirm": "insurance details", "member": "insurance details",
                  "reason": "reason for visit"}
_CALLBACK_STEPS = {"name": "name", "phone": "phone number", "reason": "reason for the call"}


def step_tag(state, text: str = "") -> str | None:
    """Which sensitive answer the caller's NEXT words will be, or None. Call this BEFORE the turn is processed."""
    if state.awaiting_ssn:
        return "last four digits"
    if "name_confirm" in (state.vstep, state.bstep) and speech.yes_no(text) not in ("yes", "no"):
        return "name"                                   # a correction of the name read back ("no, it's ...")
    return (_VERIFY_STEPS.get(state.vstep) or _BOOKING_STEPS.get(state.bstep) or _CALLBACK_STEPS.get(state.cstep))


# ---- pattern redaction -------------------------------------------------------------------------------------------------
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+"), "[EMAIL]"),
    (re.compile(r"(?i)(email is )(.+?)(?=\. (?:Is|Thank)|\?|$)"), r"\1[EMAIL]"),                  # the spelled-out read-back
    # an address said or spelled as words: "j o h n at gmail dot com", "john at the rate e x a m p l e dot c o m"
    (re.compile(r"(?i)(?:\b[a-z0-9]\s+){0,40}[\w.\-]*\s*\bat\s+(?:the\s+rate\s+)?(?:[\w\-]+\s+){1,12}?dot\s+"
                r"(?:(?:[a-z]\s+){2}[a-z]\b|[a-z]{2,4}\b)(?:\s+dot\s+(?:(?:[a-z]\s+){0,2}[a-z]\b|[a-z]{2,4}\b))*"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"), "[SSN]"),
    (re.compile(r"(?<!\d)(?:\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}(?!\d)"), "[PHONE]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "[DATE]"),
    (re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"), "[DATE]"),
    (re.compile(rf"(?i)\b{_MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b"), "[DATE]"),
    (re.compile(rf"(?i)\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?{_MONTH}\.?,?\s+\d{{4}}\b"), "[DATE]"),
    (re.compile(r"\$\s?\d[\d,]*(?:\.\d{1,2})?"), "$[amount]"),
    (re.compile(r"(?i)\b\d+\s?(?:mg|mcg|ml)\b"), "[dose]"),
    (re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6,20}\b"), "[ID]"),
    (re.compile(r"(?<!\d)(?!911(?!\d))\d{3,}"), "[digits]"),                     # 911 is the one number worth keeping
]


def _spoken_digit_runs(text: str) -> str:
    """'five five five one two three...' -> '[digits]' (only runs of three or more digits)."""
    def repl(m: re.Match) -> str:
        if m.group(0).strip() == "911":                         # the one number worth keeping
            return m.group(0)
        return "[digits]" if len(gate._run_to_digits(m.group(0))) >= 3 else m.group(0)
    return gate._RUN.sub(repl, text)


def redact(text: str, known: Iterable[str] = (), emails: Iterable[str] = (), reasons: Iterable[str] = ()) -> str:
    """Remove sensitive details from free text. `known` are names/medications/IDs to mask; `emails` are known addresses
    (masked plain and as the agent spells them back); `reasons` are the visit/call reasons the caller gave."""
    from .services import spell_email
    out = text or ""
    for email in emails:
        for form in {email, spell_email(email)}:
            out = re.sub(re.escape(form), "[EMAIL]", out, flags=re.I)
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    out = _spoken_digit_runs(out)
    for phrase in sorted({r for r in reasons if r and len(r) >= 4}, key=len, reverse=True):
        out = re.sub(rf"(?i)(?<!\w){re.escape(phrase)}(?!\w)", "[REASON]", out)
    for term in sorted({t for t in known if t and len(t) >= 3}, key=len, reverse=True):
        out = re.sub(rf"(?i)\b{re.escape(term)}\b", "[NAME]" if term[0].isalpha() else "[ID]", out)
    return out


def known_terms(state) -> tuple[list[str], list[str], list[str]]:
    """(names / medications / ids, emails, visit or call reasons) to mask, from the database and what this call captured."""
    names: set[str] = set()
    emails: set[str] = set()
    reasons: set[str] = set()
    try:
        conn = db.connect(state.db_path)
        try:
            for r in conn.execute("SELECT first_name, last_name, email FROM patients"):
                names.update((r["first_name"], r["last_name"]))
                if r["email"]:
                    emails.add(r["email"])
            for r in conn.execute("SELECT name, pharmacy FROM medications"):
                names.add(r["name"].split()[0])
        finally:
            conn.close()
    except Exception:                                                   # noqa: BLE001  (a missing DB must not stop transcripts)
        pass
    for source in (state.intake, state.callback):
        for key in ("name", "partial", "provider", "provider_heard", "member"):
            if source.get(key):
                names.update(str(source[key]).split() if key in ("name", "partial") else [str(source[key])])
        for key in ("email",):
            if source.get(key):
                emails.add(source[key])
        if source.get("reason"):
            reasons.add(str(source["reason"]))
    for v in (state.pending_name, state.partial_first):
        if v:
            names.update(v.split())
    return sorted(names), sorted(emails), sorted(reasons)


# ---- the writer --------------------------------------------------------------------------------------------------------------
class TranscriptWriter:
    """Append-only, never raises. `enabled=False` makes every method a no-op (no file is created)."""

    def __init__(self, state, directory: Path | str | None = None, enabled: bool = True,
                 now: Callable[[], datetime] = datetime.now) -> None:
        self.state, self._now, self.enabled = state, now, enabled
        self.directory = Path(directory or config.transcript_dir())
        self.path: Path | None = None
        self._fh = None
        self._closed = False
        self._started_at: datetime | None = None
        self._counts = {"caller": 0, "agent": 0, "events": 0}
        self._warned = False
        self._names: set[str] = set()
        self._emails: set[str] = set()
        self._reasons: set[str] = set()
        self.caller_allowed = False        # caller words are stored only after the recording/transcription notice was heard

    # -- internals --
    def _safe(self, fn: Callable[[], None]) -> None:
        if not self.enabled:
            return
        try:
            fn()
        except Exception as e:                                          # noqa: BLE001  (never interrupt the call)
            if not self._warned:
                self._warned = True
                logger.warning("transcript writing failed (%s); the call continues without it", type(e).__name__)

    def _line(self, text: str) -> None:
        if self._fh is None or self._closed:
            return
        self._fh.write(text + "\n")
        self._fh.flush()

    def _stamp(self) -> str:
        return self._now().strftime("%H:%M:%S")

    def _remember(self) -> None:
        """Keep every name / email this call has captured, even after the flow that held it has been cleared."""
        names, emails, reasons = known_terms(self.state)
        self._names.update(names)
        self._emails.update(emails)
        self._reasons.update(reasons)

    def _clean(self, text: str) -> str:
        self._remember()
        return " ".join(redact(text, self._names, self._emails, self._reasons).split())

    # -- public API --
    def start(self, notice: str = "spoken in the greeting: the call may be recorded and transcribed for quality purposes") -> None:
        def go() -> None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._started_at = self._now()
            self.path = self.directory / f"{self._started_at:%Y%m%d-%H%M%S}_{self.state.call_id}.txt"
            self._fh = open(self.path, "a", encoding="utf-8")
            clock = self.state.clock
            clinic = f"{clock.now():%Y-%m-%d %H:%M} ET" + (" (SIMULATED clinic clock)" if clock.simulated else "")
            for h in ("SUNRISE FAMILY CLINIC - CALL TRANSCRIPT (synthetic test call; sensitive details redacted)",
                      f"Call ID:      {self.state.call_id}",
                      f"Started:      {self._started_at:%Y-%m-%d %H:%M:%S}",
                      f"Clinic time:  {clinic}",
                      f"Notice:       {notice}",
                      "Redaction:    answers to name / date of birth / last four digits / phone / email / insurance / reason questions are",
                      "              not stored; other text has SSNs, phones, emails, dates with a year, amounts, IDs, medications and",
                      "              known patient names masked. See README, 'Transcripts'.",
                      "-" * 100):
                self._line(h)
        self._safe(go)

    def notice_given(self, delivered: bool) -> None:
        """Called once the greeting finished. Only then are the caller's words stored; if the notice could not be spoken
        (for example the TTS quota is used up) the caller is never transcribed on this call."""
        self.caller_allowed = bool(delivered) and self.enabled
        self.event("recording notice delivered; caller speech is transcribed (sensitive answers withheld)" if delivered else
                   "recording notice NOT delivered; caller speech is NOT transcribed on this call")

    def caller(self, text: str, tag: str | None = None) -> None:
        def go() -> None:
            self._remember()
            if not self.caller_allowed:
                self._line(f"[{self._stamp()}] CALLER: [speech not transcribed - recording notice not yet delivered]")
                self._counts["caller"] += 1
                return
            body = f"[{tag} provided - not stored]" if tag else (self._clean(text) or "[unintelligible]")
            self._line(f"[{self._stamp()}] CALLER: {body}")
            self._counts["caller"] += 1
        self._safe(go)

    def agent(self, text: str, interrupted: bool = False) -> None:
        def go() -> None:
            if not (text or "").strip():
                return
            label = "AGENT (interrupted by caller)" if interrupted else "AGENT"
            self._line(f"[{self._stamp()}] {label}: {self._clean(text)}")
            self._counts["agent"] += 1
        self._safe(go)

    def event(self, text: str) -> None:
        def go() -> None:
            self._line(f"[{self._stamp()}] EVENT: {self._clean(text)}")
            self._counts["events"] += 1
        self._safe(go)

    def close(self, reason: str = "call ended") -> None:
        """Write the end marker and close. Safe to call any number of times, from any shutdown path."""
        def go() -> None:
            if self._closed or self._fh is None:
                return
            ended = self._now()
            secs = int((ended - self._started_at).total_seconds()) if self._started_at else 0
            self._line("-" * 100)
            self._line(f"Call ended:   {ended:%Y-%m-%d %H:%M:%S} | duration {secs // 60}m {secs % 60:02d}s | reason: {reason} | "
                       f"caller turns {self._counts['caller']}, agent turns {self._counts['agent']}, events {self._counts['events']}")
            self._fh.close()
            self._closed = True
        self._safe(go)


# ---- LiveKit wiring (duck-typed, so it is testable without a live session) ---------------------------------------------------------
def on_item_added(writer: TranscriptWriter, ev: Any) -> None:
    """conversation_item_added: store what the AGENT said (caller turns are stored by the agent hook, where the step is known)."""
    item = getattr(ev, "item", None)
    if item is None or getattr(item, "type", "message") != "message" or getattr(item, "role", "") != "assistant":
        return
    writer.agent(getattr(item, "text_content", "") or "", interrupted=bool(getattr(item, "interrupted", False)))


def on_error(writer: TranscriptWriter, ev: Any) -> None:
    """error: record only the kind of failure (never the message, which could echo caller words)."""
    inner = getattr(ev, "error", None)
    writer.event(f"{type(getattr(ev, 'source', None)).__name__} error: {type(inner).__name__}"
                 f"{' (recoverable)' if getattr(inner, 'recoverable', False) else ''}")


def on_close(writer: TranscriptWriter, ev: Any) -> None:
    reason = getattr(getattr(ev, "reason", None), "value", None) or str(getattr(ev, "reason", "") or "") or "session closed"
    writer.close(f"session closed ({reason})")


def attach(session: Any, writer: TranscriptWriter) -> None:
    """Hook the writer to a session's events. Handlers are synchronous and swallow every exception."""
    def guarded(fn: Callable[[TranscriptWriter, Any], None]) -> Callable[[Any], None]:
        def run(ev: Any) -> None:
            try:
                fn(writer, ev)
            except Exception:                                   # noqa: BLE001  (never interrupt the call)
                pass
        return run
    session.on("conversation_item_added", guarded(on_item_added))
    session.on("error", guarded(on_error))
    session.on("close", guarded(on_close))


# ---- command line: python -m clinic.transcript list | show [latest|<file>] --------------------------------------------------------
def _files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.txt"), key=lambda p: p.stat().st_mtime) if directory.exists() else []


def main(argv: list[str]) -> int:
    config.load_env()
    directory = config.transcript_dir()
    cmd = argv[0] if argv else "list"
    files = _files(directory)
    if cmd == "list":
        if not files:
            print(f"No transcripts in {directory}. Make a test call with TRANSCRIPTS=on (the default).")
        for p in files:
            done = "complete" if "Call ended:" in p.read_text(encoding="utf-8", errors="ignore") else "INCOMPLETE (no end marker)"
            print(f"{p.name}  {p.stat().st_size:>6} bytes  {done}")
        return 0
    if cmd == "show":
        target = argv[1] if len(argv) > 1 else "latest"
        pick = files[-1] if target == "latest" and files else next((p for p in files if target in p.name), None)
        if pick is None:
            print("No matching transcript.")
            return 1
        sys.stdout.reconfigure(encoding="utf-8")
        print(pick.read_text(encoding="utf-8"))
        return 0
    print("usage: python -m clinic.transcript [list | show [latest|<part of the file name>]]")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
