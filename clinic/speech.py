"""Parsing of what a caller SAYS (speech-to-text output) into clean values.

Shared by the verification and booking conversations. Everything here is deterministic and
testable; the LLM is never asked to interpret these answers.
"""

from __future__ import annotations

import re
from datetime import datetime

from . import dates, gate

# ---- small intent regexes ---------------------------------------------------------------------------------
WAIT = re.compile(r"\b(wait|hold on|hang on|one (moment|sec|second|minute)|just (a )?(moment|sec|second|minute)|"
                  r"give me a (moment|sec|second|minute)|let me (find|get|check|look|grab|see|think)|sorry|"
                  r"stop|um+|uh+|hmm+|ah+|oh+)\b", re.I)
REFUSE = re.compile(r"\b(rather not|don'?t want|do not want|not comfortable|won'?t|refuse|no thanks|"
                    r"don'?t have it|can'?t remember|don'?t remember|don'?t know|not sure i should)\b", re.I)
WANT_STAFF = re.compile(r"\b(speak|talk|transfer|connect)\b.{0,25}\b(person|human|someone|staff|representative|"
                        r"agent|receptionist|nurse)\b|\b(real person|a human|staff member)\b", re.I)
NEW_PATIENT = re.compile(r"\b(new patient|first visit|first time|never been|haven'?t been (there|seen)|"
                         r"not (a|an) (existing|current) patient|not a patient)\b", re.I)
EXISTING_PATIENT = re.compile(r"\b(old patient|existing patient|current patient|returning|established|regular patient|"
                              r"been (there|here|seen) before|already a patient|been coming|i'?m a patient)\b", re.I)
CANCEL = re.compile(r"\b(never ?mind|forget it|cancel|skip (it|this)|not now)\b", re.I)
ALREADY = re.compile(r"\b(already (told|gave|said|answered|provided)|i (just )?(told|said|gave)( you)?)\b", re.I)


def is_wait(text: str) -> bool:
    return bool(WAIT.search(text)) and len(text.split()) <= 10


# ---- names --------------------------------------------------------------------------------------------------
_NAME_MARKER = re.compile(r"\bname\s*(?:is|'s|was)\b", re.I)          # "my (full) name is ..."
_NAME_LEAD = re.compile(r"^(?:(?:yes|yeah|yep|okay|ok|sure|um+|uh+|hi|hello|hey|sorry|well|so|alright|no)(?:[\s,.]+|$))*"
                        r"(?:(?:my name's|name's|this is|it's|it is|its|i'm|i am|call me|they call me|the name is|"
                        r"that's|that is|it should be|should be)\s+)?", re.I)
_DATE_WORDS = {"born", "birthday", "birth", "date", "dob", "of", "on", "and", "is", "was", "th", "st", "nd", "rd",
               "the", "my", "in"}
_NAME_TAIL = re.compile(r"(?:[\s,.]+(?:please|thanks|thank you|speaking|here|and|okay|ok))+\s*$", re.I)
_NOT_NAME = {"what", "why", "who", "how", "when", "where", "do", "does", "can", "could", "would", "is", "are",
             "don't", "dont", "not", "no", "refill", "appointment", "balance", "need", "want", "ignore",
             "instructions", "patient", "records", "the", "your", "my", "have", "has", "wait", "moment",
             "second", "minute", "hold", "sorry", "already", "gave", "told", "said", "yes", "yeah", "correct", "right",
             "thank", "thanks", "you", "please", "hello", "hi", "hey", "okay", "ok", "bye", "goodbye", "nothing",
             "none", "nope", "welcome"}


def despell(text: str) -> str:
    """'W-A-S-I-F' -> 'WASIF', 'j o h n' -> 'john'. Words separated by spaces stay separate."""
    text = re.sub(r"\b[A-Za-z](?:[-.][A-Za-z]){2,}\b\.?", lambda m: re.sub(r"[-.]", "", m.group(0)), text)
    return re.sub(r"\b(?:[A-Za-z]\s){2,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)


def extract_name(text: str, partial_first: str | None = None) -> tuple[str, str | None]:
    """-> (status, value). status: ok (value = full name) | partial (value = first name) | unusable."""
    raw = text or ""
    t = despell(raw)
    if (m := _NAME_MARKER.search(t)):                       # "my full name is X" -> "X"
        t = t[m.end():]
    t = re.sub(r"[^A-Za-z'\- ]+", " ", t)
    t = _NAME_LEAD.sub("", t.strip())
    if re.search(r"\d", raw) or any(w in dates._WORDNUM for w in re.findall(r"[a-z]+", raw.lower())):
        # the answer also contains a date ("Sarah Johnson, born March 14th 1985"): keep only the name part
        t = " ".join(w for w in t.split() if w.lower() not in _DATE_WORDS and w.lower() not in dates._MONTHS
                     and w.lower() not in dates._WORDNUM and w.lower() not in ("oh", "zero"))
    t = _NAME_TAIL.sub("", t).strip()
    toks = [w.strip("-'") for w in t.split()]
    toks = [w for w in toks if w]
    if not toks or len(toks) > 4 or any(w.lower() in _NOT_NAME for w in toks):
        return "unusable", None
    if any(w.lower() in dates._WORDNUM or w.lower() in ("oh", "zero") for w in toks):
        return "unusable", None                      # number words: that was a date, not a name
    if any(w.lower() in dates._MONTHS for w in toks) and dates.parse_spoken_date(raw).status in ("ok", "ambiguous"):
        return "unusable", None
    if partial_first and len(toks) == 1:
        return "ok", f"{partial_first} {toks[0].title()}".title()
    if len(toks) == 1:
        return "partial", toks[0].title()
    return "ok", " ".join(w.title() for w in toks)


# ---- yes / no ------------------------------------------------------------------------------------------------
_UNSURE = re.compile(r"\b(don'?t know|do not know|not sure|no idea|unsure|dunno|i guess|maybe)\b", re.I)


def yes_no(text: str) -> str | None:
    """'yes' | 'no' | None (neither, both, or unsure: \"I don't know\" is not a no)."""
    if _UNSURE.search(text):
        return None
    yes, no = gate.is_affirmative(text), gate.is_negative(text)
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"
    return None


# ---- phone ------------------------------------------------------------------------------------------------------
def digits_from_speech(text: str) -> str:
    return "".join(gate._run_to_digits(m.group(0)) for m in gate._RUN.finditer(text or ""))


def phone_from_speech(text: str) -> tuple[str, str | None]:
    """-> ('ok', '555-123-4567') | ('short', None) | ('long', None) | ('none', None)"""
    d = digits_from_speech(text)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if not d:
        return "none", None
    if len(d) < 10:
        return "short", None
    if len(d) > 10:
        return "long", None
    return "ok", f"{d[:3]}-{d[3:6]}-{d[6:]}"


# ---- email -------------------------------------------------------------------------------------------------------
_EMAIL_LITERAL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_EMAIL_FILLER = {"my", "email", "e", "mail", "address", "is", "its", "it", "s", "yes", "yeah", "yep", "okay", "ok",
                 "sure", "um", "uh", "so", "well", "the", "that", "thats", "it's"}
_EMAIL_SPECIAL = {"at": "@", "dot": ".", "period": ".", "underscore": "_", "dash": "-", "hyphen": "-", "plus": "+"}
_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9"}


_PROVIDERS = ("gmail", "yahoo", "outlook", "hotmail", "icloud", "aol")
_AT_PHRASES = re.compile(r"\bat (?:the )?rate(?: of)?\b|\bat[- ]sign\b|\bat symbol\b|\bat the red\b", re.I)
_DOMAIN_OK = re.compile(r"[a-z0-9\-]+(?:\.[a-z0-9\-]+)*\.[a-z]{2,}")
_LOCAL_OK = re.compile(r"[a-z0-9._%+\-]{2,}")


def email_parts(text: str) -> tuple[str, str]:
    """Split what was said into (local part, domain). Either may be '' when only one half was heard.

    Handles "vasif at gmail.com" (STT writes real dots), "vasif at the rate gmail dot com" (Indian English),
    spelled letters ("w a s i f"), spoken digits, and halves said in separate turns."""
    t = (text or "").strip()
    if (m := _EMAIL_LITERAL.search(t)):                      # STT already wrote a complete address
        local, _, domain = m.group(0).lower().partition("@")
        return local, domain
    low = _AT_PHRASES.sub(" @ ", t.lower()).replace("@", " @ ")
    toks = re.findall(r"[a-z]+|\d+|@|\.", low)
    while toks and toks[0] in _EMAIL_FILLER:
        toks.pop(0)
    while toks and toks[-1] in ("please", "thanks", "thank", "you"):
        toks.pop()
    out = []
    for w in toks:
        out.append("@" if w in ("@", "at") else _EMAIL_SPECIAL.get(w, _DIGIT_WORDS.get(w, w)))
    joined = "".join(out)
    local, at, domain = joined.partition("@")
    domain = domain.strip(".")
    for prov in _PROVIDERS:                                  # "gmail com" -> "gmail.com"
        if domain == prov + "com":
            domain = prov + ".com"
        elif domain == prov:
            domain = prov + ".com"
    local = local.strip(".")
    if not at:                                               # no "at" word: a bare local part, or a bare domain
        bare = joined.strip(".")
        if _DOMAIN_OK.fullmatch(bare) and bare.split(".")[0] in _PROVIDERS:
            return "", bare
        words = [w for w in toks if w not in (".",)]
        # a spelled-out or single-word name part; two ordinary words ("no idea") are not an email half
        spelled = all(len(w) == 1 or w.isdigit() or w in _DIGIT_WORDS or w in _EMAIL_SPECIAL for w in words)
        return (bare, "") if _LOCAL_OK.fullmatch(bare) and (spelled or len(words) == 1) else ("", "")
    return (local if _LOCAL_OK.fullmatch(local) else ""), (domain if _DOMAIN_OK.fullmatch(domain) else "")


def email_from_speech(text: str) -> str | None:
    """A complete address from one utterance, or None."""
    local, domain = email_parts(text)
    return f"{local}@{domain}" if local and domain else None


# ---- insurance -----------------------------------------------------------------------------------------------------
SELF_PAY = re.compile(r"\b(self[- ]?pay|no insurance|don'?t have (any )?insurance|do not have (any )?insurance|"
                      r"without insurance|cash|out of pocket|pay (for it )?myself|paying myself|uninsured)\b", re.I)
_PROVIDER_LEAD = re.compile(r"^(?:(?:yes|yeah|yep|okay|ok|sure|um+|uh+|so|well)(?:[\s,.]+|$))*"
                            r"(?:(?:my |our )?(?:insurance|provider|insurer|plan)?\s*(?:is|with|through|from)?\s*)?"
                            r"(?:it'?s |it is |i have |i'm with |i am with |i use )?", re.I)


def provider_from_speech(text: str) -> str | None:
    t = _PROVIDER_LEAD.sub("", re.sub(r"[^A-Za-z0-9&' \-]+", " ", text or "").strip())
    t = re.sub(r"\s+(please|thanks|thank you)$", "", t, flags=re.I).strip()
    words = t.split()
    return " ".join(w if w.isupper() else w.title() for w in words) if 1 <= len(words) <= 6 else None


_KNOWN_INSURERS = re.compile(
    r"\b(aetna|cigna|humana|anthem|kaiser|molina|oscar|ambetter|wellcare|amerigroup|tricare|medicare|medicaid|blue ?cross|"
    r"blue ?shield|bcbs|united ?health ?care|united|health ?net|oxford|empire|horizon|highmark|emblem|independence|geisinger|"
    r"premera|regence|centene|carefirst|florida blue|avmed|upmc|harvard pilgrim|tufts|fidelis|metlife|optum|evernorth)\b", re.I)


def is_known_insurer(name: str) -> bool:
    """Speech-to-text turns "self-pay" into things like "Sophie". A provider the code does not recognise is read back for a yes."""
    return bool(_KNOWN_INSURERS.search(name or ""))


def member_id_from_speech(text: str) -> str | None:
    """'B S one two three four five' -> 'BS12345'; 'I don't have it with me' -> 'to be provided at visit'."""
    low = (text or "").lower()
    if re.search(r"don'?t have|do not have|not with me|forgot|left it|at home|can'?t find|don'?t know", low):
        return "to be provided at visit"
    toks = re.findall(r"[a-z]+|\d+", low)
    while toks and toks[0] in {"my", "member", "id", "number", "is", "its", "it", "s", "the", "yes", "yeah", "okay", "ok",
                               "um", "uh", "that", "thats"}:
        toks.pop(0)
    out = "".join(_DIGIT_WORDS.get(w, w) for w in toks if w not in ("please", "thanks", "thank", "you"))
    return out.upper() if re.fullmatch(r"[A-Z0-9]{4,24}", out.upper()) else None


# ---- reason for visit ---------------------------------------------------------------------------------------------
_REASON_LEAD = re.compile(r"^(?:(?:yes|yeah|yep|okay|ok|sure|um+|uh+|so|well)(?:[\s,.]+|$))*"
                          r"(?:(?:it'?s|it is|that'?s|its)\s+)?"
                          r"(?:(?:i|we)\s*(?:need|want|would like|'d like|d like|am coming|'m coming)\s+(?:to\s+)?"
                          r"(?:come in |be seen |see (?:the )?doctor )?)?"
                          r"(?:(?:for|about|regarding|because of|to)\s+)?(?:an?\s+|my\s+)?", re.I)


def reason_from_speech(text: str) -> str | None:
    t = _REASON_LEAD.sub("", re.sub(r"\s+", " ", (text or "").strip()).rstrip(".!?"))
    return t if 2 <= len(t) <= 80 and not is_wait(t) else None


# ---- doctor / day preferences ------------------------------------------------------------------------------------------
_NO_PREF = re.compile(r"\b(no preference|doesn'?t matter|does not matter|any(one|body|thing|time| time| doctor| day)?|"
                      r"either|whichever|whatever|nope|no|nothing|don'?t care|not really|no thanks|up to you)\b", re.I)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")


def preferences_from_speech(text: str) -> dict | None:
    """-> {'doctor','weekday','time_of_day'} (values may be '') or None when nothing usable was said."""
    low = (text or "").lower()
    doctor = "Patel" if "patel" in low else "Lee" if re.search(r"\blee\b", low) else ""
    weekday = next((d for d in _WEEKDAYS if d in low), "")
    tod = "morning" if re.search(r"\bmorning\b", low) else "afternoon" if re.search(r"\bafternoon\b", low) else ""
    if doctor or weekday or tod:
        return {"doctor": doctor, "weekday": weekday, "time_of_day": tod}
    return {"doctor": "", "weekday": "", "time_of_day": ""} if _NO_PREF.search(low) else None


# ---- picking one of the offered appointment slots ------------------------------------------------------------------------
_OTHER_TIME = re.compile(r"\b(none of (those|them)|something else|different|another (day|time|doctor)|other (day|time)|"
                         r"later|earlier|neither)\b", re.I)


_CLOCK_MARK = re.compile(r"\b(am|a m|pm|p m|o'?clock)\b|\d\s*:\s*\d")
_BOOK_NUMBER = re.compile(r"\b(?:book|take|pick|choose|select|want|go with|do)\s+(?:me\s+)?(?:the\s+)?(?:number\s+|option\s+)?"
                          r"(one|two|three|1|2|3)\b(?!\s*(?:thirty|fifteen|forty|o'?clock|:|\d))")


def wants_other_slots(text: str) -> bool:
    return bool(_OTHER_TIME.search(text or ""))


def choose_slot(text: str, slots: list[dict]) -> tuple[str, int | None]:
    """slots: [{'slot_id', 'doctor', 'starts_at'}]. -> ('ok', slot_id) | ('ambiguous', None) | ('none', None)."""
    low = re.sub(r"(\d)\s*(am|pm)\b", r"\1 \2", (text or "").lower())    # "10am" -> "10 am" (there is no word boundary in "10am")
    cands = list(slots)
    ordinals = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    if (m := re.search(r"\b(first|1st|second|2nd|third|3rd|last)\b(?:\s+(?:one|option|slot|choice))?", low)):
        idx = -1 if m.group(1) == "last" else ordinals[m.group(1)]
        if -len(cands) <= idx < len(cands):
            return "ok", cands[idx]["slot_id"]
    if (m := re.search(r"\b(?:option|number|choice)\s+(one|two|three|1|2|3)\b", low)):
        idx = {"one": 0, "1": 0, "two": 1, "2": 1, "three": 2, "3": 2}[m.group(1)]
        if idx < len(cands):
            return "ok", cands[idx]["slot_id"]
    # "book 1" / "take two" is the Nth option, unless a clock time is also said ("book 1 for 10am": the time wins)
    if not _CLOCK_MARK.search(low) and (m := _BOOK_NUMBER.search(low)):
        idx = {"one": 0, "1": 0, "two": 1, "2": 1, "three": 2, "3": 2}[m.group(1)]
        if idx < len(cands):
            return "ok", cands[idx]["slot_id"]

    if "patel" in low:
        cands = [s for s in cands if "patel" in s["doctor"].lower()] or cands
    elif re.search(r"\blee\b", low):
        cands = [s for s in cands if "lee" in s["doctor"].lower()] or cands
    for day in _WEEKDAYS:
        if day in low:
            cands = [s for s in cands if datetime.fromisoformat(s["starts_at"]).strftime("%A").lower() == day] or cands
    timey = re.sub(r"\b(?:the|that|this)\s+(?:\w+\s+)?one\b", " ", low)   # "the Wednesday one" is not 1 o'clock
    toks = dates._tokens(timey)
    nums = [v for v in dates._scan(toks)[1] if 0 <= v <= 59]
    has_time = bool(nums) and (re.search(r"\b(am|pm|a m|p m|o'?clock)\b", timey) or re.search(r"\d", timey)
                               or any(w in dates._WORDNUM for w in toks))
    if has_time:
        hour = nums[0]
        minute = nums[1] if len(nums) > 1 and nums[1] in range(0, 60) and nums[1] not in range(1, 13) else None
        if (mer := re.search(r"\b(am|a m|pm|p m)\b", timey)) and (before := [
                v for v in dates._scan(dates._tokens(timey[:mer.start()]))[1] if 0 <= v <= 59]):
            # the time is the number(s) right before am/pm, not an option number said earlier ("book 1 for 10 am")
            hour, minute = (before[-2], before[-1]) if len(before) > 1 and before[-1] not in range(1, 13) else (before[-1], None)
        pm = bool(re.search(r"\b(pm|p m|afternoon|evening)\b", low))
        am = bool(re.search(r"\b(am|a m|morning)\b", low))

        def time_ok(s: dict) -> bool:
            dt = datetime.fromisoformat(s["starts_at"])
            if (dt.hour % 12 or 12) != hour:
                return False
            if minute is not None and dt.minute != minute:
                return False
            if pm and dt.hour < 12 or am and dt.hour >= 12:
                return False
            return True

        timed = [s for s in cands if time_ok(s)]
        cands = timed if timed else []
    if len(cands) == 1:
        return "ok", cands[0]["slot_id"]
    if not cands or len(cands) == len(slots) and not (has_time or "patel" in low or "lee" in low):
        return "none", None
    return "ambiguous", None
