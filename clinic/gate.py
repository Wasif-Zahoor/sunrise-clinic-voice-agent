"""Transcript gate: runs in code, before the LLM sees anything.

1. SSN capture/scrub - while verification is waiting for the last four SSN digits (or when the
   caller volunteers them after words like "social"), spoken digits are removed from the
   transcript and kept only in backend state. The LLM never receives them.
   LIMIT: the audio itself still goes to the STT vendor (Groq Whisper); see README.
2. Emergency detection (regex, before the LLM).
3. Confirmation detection - "did the caller really say yes?" is decided from the transcript,
   not from the LLM's claim.
4. Prompt-injection and third-party hints (logged and passed to the LLM as a system note).
"""

from __future__ import annotations

import re

from .state import CallState

WITHHELD = "[digits withheld]"

_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_W = "|".join(_WORDS)
_TOKEN = rf"(?:\d+|(?:double|triple)\s+(?:{_W})|(?:{_W}))"
_RUN = re.compile(rf"(?<![\w])(?:{_TOKEN})(?:[\s,.\-]+(?:{_TOKEN}))*(?![\w])", re.I)
_CUE = re.compile(r"social|\bssn\b|security number|last (four|4)", re.I)


def _run_to_digits(run: str) -> str:
    out = []
    for tok in re.findall(r"(?:double|triple)\s+[a-z]+|\d+|[a-z]+", run.lower()):
        if tok.startswith(("double", "triple")):
            mult = 2 if tok.startswith("double") else 3
            out.append(_WORDS.get(tok.split()[1], "") * mult)
        elif tok.isdigit():
            out.append(tok)
        else:
            out.append(_WORDS.get(tok, ""))
    return "".join(out)


def scrub_transcript(state: CallState, text: str, capture: bool = True) -> str:
    """Remove SSN digits from `text`; keep them in `state.ssn_buffer` (backend only)."""
    if state.awaiting_ssn:
        head, tail = "", text
    else:
        m = _CUE.search(text)
        if not m:
            return text
        head, tail = text[: m.end()], text[m.end():]

    def repl(m: re.Match) -> str:
        digits = _run_to_digits(m.group(0))
        # a lone "one" / "3" is ordinary speech, not an SSN fragment - unless it is the whole answer to the digits question
        if len(digits) < 2 and not (state.awaiting_ssn and m.group(0).strip(" .,") == text.strip(" .,")):
            return m.group(0)
        if capture:
            state.capture_ssn_digits(digits)
        return WITHHELD

    return head + _RUN.sub(repl, tail)


def ssn_input_status(state: CallState) -> str:
    """'none' | 'short' | 'ready' (exactly 4) | 'long' (more than 4: never used or stored)."""
    n = len(state.ssn_buffer)
    return "none" if n == 0 else "ready" if n == 4 else "short" if n < 4 else "long"


# ---- emergencies ---------------------------------------------------------------------------
_EMERGENCY = [re.compile(p, re.I) for p in (
    r"chest (pain|pressure|tightness|hurts?)",
    r"(can'?t|cannot|can not|unable to|trouble|difficulty|hard to|struggling to|problems?) (breathe|breathing|catch (my|a) breath)",
    r"(shortness of breath|short of breath|not breathing|stopped breathing|gasping)",
    r"suicid", r"kill (myself|my self)", r"end (my|my own) life", r"(want|going|plan(ning)?) to die",
    r"heart attack", r"\bstroke\b", r"overdos", r"unconscious|passed out|not responding|unresponsive",
    r"(severe|heavy) bleeding|bleeding (a lot|heavily|badly)",
    r"(this is|it'?s|it is|have|having|is) an? (medical )?emergency", r"\bcall(ing)? 911\b",
    r"anaphyla", r"seizure", r"choking", r"(he|she|they)('?s| is| are)? dying",
    # stroke / heart / allergy / bleeding signs described in plain words
    r"\b(arm|face|leg|side|hand)\b.{0,25}\b(numb|went numb|gone numb|weak|drooping)\b",
    r"\bnumb(ness)?\b.{0,25}\b(arm|face|leg|side|chest)\b",
    r"\b(faint(ing)?|pass(ing)? out|blacking out|collapsed)\b", r"slurred speech|can'?t speak|trouble speaking",
    r"can'?t feel (my|his|her)\b", r"allergic reaction", r"(throat|tongue|face) (is )?(swelling|swollen|closing)",
    r"(coughing|vomiting|throwing up) (up )?blood", r"(bleeding|blood) (won'?t|will not|can'?t) (stop|be stopped)",
)]
_SUICIDE = re.compile(r"suicid|kill (myself|my self)|end (my|my own) life|want to die", re.I)

EMERGENCY_MESSAGE = (
    "If this is an emergency, please hang up and call 911 now, or go to the nearest emergency room. "
    "I can't continue with other requests on this call."
)
EMERGENCY_988 = " You can also call or text 988 for immediate support."
EMERGENCY_REPEAT = (
    "Please hang up and call 911 now, or go to the nearest emergency room. "
    "I can't help with other requests until you are safe."
)


def detect_emergency(text: str) -> bool:
    return any(p.search(text) for p in _EMERGENCY)


def emergency_message(text: str) -> str:
    return EMERGENCY_MESSAGE + (EMERGENCY_988 if _SUICIDE.search(text) else "")


# ---- confirmations --------------------------------------------------------------------------
_YES = re.compile(r"\b(yes|yeah|yep|yup|correct|confirm(ed)?|that'?s (right|correct)|that is (right|correct)|"
                  r"sounds? good|go ahead|please do|sure|absolutely|perfect|exactly|okay|ok|right)\b", re.I)
_NO = re.compile(r"\b(no|nope|nah|not|wrong|incorrect|change|wait|don'?t|cancel|stop|actually|instead)\b", re.I)


def is_negative(text: str) -> bool:
    return bool(_NO.search(text))


def is_affirmative(text: str) -> bool:
    """Conservative: any negation/correction word makes it NOT a confirmation."""
    return bool(_YES.search(text)) and not is_negative(text)


# ---- hints ------------------------------------------------------------------------------------
_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore (all |any |your |the |previous |prior |above |these )*(instructions|rules|prompt|guidelines|policies)",
    r"disregard", r"(reveal|show|print|repeat|read|tell me) (me )?(your |the )?(system |hidden |initial )?(prompt|instructions)",
    r"(developer|admin|debug|god|dan) mode", r"\byou are now\b", r"pretend (to be|you)", r"\bact as\b",
    r"\b(bypass|override|jailbreak)\b", r"\b(all|every|other) (the )?patients?\b", r"\bdatabase\b",
    r"\bi('m| am) (already )?verified\b", r"(already|been) verified",
)]
_THIRD_PARTY = re.compile(
    r"\b(my|his|her|their) (wife|husband|spouse|mother|father|mom|dad|son|daughter|friend|neighbor|"
    r"brother|sister|parent|partner|girlfriend|boyfriend|grandmother|grandfather|boss|patient)\b", re.I)


def detect_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION)


def detect_third_party(text: str) -> bool:
    return bool(_THIRD_PARTY.search(text))
