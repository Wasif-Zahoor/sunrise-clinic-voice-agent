"""Natural spoken-date parsing for date of birth.

Accepts "February 17, 1973", "17 February 1973", "the seventeenth of February nineteen seventy-three",
"02/17/1973", "1973-02-17"... and normalizes to ISO (YYYY-MM-DD) internally. The ISO form is never
shown to the caller.

Statuses: ok | ambiguous (numeric dates readable two ways) | invalid (not a real / plausible date)
          | incomplete (month, day or year missing) | none (no date found at all)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
_MONTHS = {n.lower(): i for i, n in enumerate(MONTH_NAMES, 1)}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
                "oct": 10, "nov": 11, "dec": 12})

_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_ORD_ONES = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
             "eighth": 8, "ninth": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_ORD_TEENS = {"tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
              "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_ORD_TENS = {"twentieth": 20, "thirtieth": 30}
_WORDNUM = {**_ONES, **_ORD_ONES, **_TEENS, **_ORD_TEENS, **_TENS, **_ORD_TENS}


@dataclass
class DateResult:
    status: str
    iso: str | None = None
    candidates: list[str] = field(default_factory=list)   # for status == "ambiguous"
    missing: str = ""                                      # for status == "incomplete"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def speak_date(iso: str) -> str:
    y, m, d = (int(p) for p in iso.split("-"))
    return f"{MONTH_NAMES[m - 1]} {ordinal(d)}, {y}"


def _tokens(text: str) -> list[str]:
    t = text.lower()
    t = re.sub(r"(?<=\d)[/\-.](?=\d)", " ", t)       # 02/17/1973 and 1973-02-17
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)      # seventy-three
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)     # 17th
    return re.findall(r"[a-z]+|\d+", t)


def _group_start(tok: str) -> bool:
    return tok in _WORDNUM or tok in ("oh", "zero")


def _group(toks: list[str], i: int) -> tuple[int, int, bool]:
    """One spoken two-digit group -> (value, next index, is_ordinal)."""
    t = toks[i]
    if t in ("oh", "zero") and i + 1 < len(toks) and toks[i + 1] in _ONES:
        return _ONES[toks[i + 1]], i + 2, False
    if t in _ONES:
        return _ONES[t], i + 1, False
    if t in _ORD_ONES:
        return _ORD_ONES[t], i + 1, True
    if t in _TEENS:
        return _TEENS[t], i + 1, False
    if t in _ORD_TEENS:
        return _ORD_TEENS[t], i + 1, True
    if t in _ORD_TENS:
        return _ORD_TENS[t], i + 1, True
    if t in _TENS:
        v = _TENS[t]
        if i + 1 < len(toks):
            nxt = toks[i + 1]
            if nxt in _ONES:
                return v + _ONES[nxt], i + 2, False
            if nxt in _ORD_ONES:
                return v + _ORD_ONES[nxt], i + 2, True
        return v, i + 1, False
    return 0, i + 1, False


def _read_words(toks: list[str], i: int) -> tuple[int, int]:
    """Spoken number (day or year) starting at toks[i] -> (value, next index)."""
    if toks[i] == "two" and i + 1 < len(toks) and toks[i + 1] == "thousand":
        j = i + 2
        if j < len(toks) and toks[j] == "and":
            j += 1
        if j < len(toks) and _group_start(toks[j]):
            g, j, _ = _group(toks, j)
            return 2000 + g, j
        return 2000, j
    g, j, was_ordinal = _group(toks, i)
    if not was_ordinal and g in (18, 19, 20) and j < len(toks):
        if toks[j] == "hundred":
            return g * 100, j + 1
        if _group_start(toks[j]):
            g2, j2, _ = _group(toks, j)
            if not (g == 20 and g2 in (18, 19)):      # "twenty nineteen eighty" = day 20 + year 1980
                return g * 100 + g2, j2
    return g, j


def _scan(toks: list[str]) -> tuple[int | None, list[int]]:
    month, items, i = None, [], 0
    while i < len(toks):
        t = toks[i]
        if t in _MONTHS:
            month = month or _MONTHS[t]
            i += 1
        elif t.isdigit():
            items.append(int(t))
            i += 1
        elif t in _WORDNUM or (t in ("oh", "zero") and i + 1 < len(toks) and toks[i + 1] in _ONES):
            v, i = _read_words(toks, i)
            items.append(v)
        else:
            i += 1
    return month, items


def _valid(y: int, m: int, d: int, today: date) -> str | None:
    try:
        dt = date(y, m, d)
    except ValueError:
        return None
    return dt.isoformat() if date(1900, 1, 1) <= dt <= today else None


def parse_spoken_date(text: str | None, today: date | None = None) -> DateResult:
    today = today or date.today()
    month, items = _scan(_tokens(text or ""))
    years = [v for v in items if v >= 1000]
    smalls = [v for v in items if 0 < v < 1000 and v < 100]

    if month is not None:
        days = [v for v in smalls if v <= 31]
        if not days:
            return DateResult("incomplete", missing="day" if years or not items else "day")
        if not years:
            return DateResult("incomplete", missing="year")
        iso = _valid(years[0], month, days[0], today)
        return DateResult("ok", iso) if iso else DateResult("invalid")

    if not items:
        return DateResult("none")
    if items[0] >= 1000 and len(items) >= 3:                      # ISO order: 1973-02-17
        iso = _valid(items[0], items[1], items[2], today)
        return DateResult("ok", iso) if iso else DateResult("invalid")
    if len(items) < 3 or not years:
        return DateResult("incomplete", missing="month, day and year")
    a, b, y = items[0], items[1], years[-1]
    if y != items[2] or a >= 1000 or b >= 1000:
        return DateResult("incomplete", missing="month, day and year")
    us, eu = (_valid(y, a, b, today), _valid(y, b, a, today)) if a <= 12 and b <= 12 else (None, None)
    if a <= 12 and b > 12:
        iso = _valid(y, a, b, today)
        return DateResult("ok", iso) if iso else DateResult("invalid")
    if a > 12 and b <= 12:
        iso = _valid(y, b, a, today)
        return DateResult("ok", iso) if iso else DateResult("invalid")
    if a > 12 and b > 12:
        return DateResult("invalid")
    if us and eu and us != eu:
        return DateResult("ambiguous", candidates=[us, eu])
    iso = us or eu
    return DateResult("ok", iso) if iso else DateResult("invalid")


def resolve_ambiguity(text: str, candidates: list[str]) -> str | None:
    """Pick one of two candidate dates from the caller's clarification, or None if unclear."""
    toks = _tokens(text or "")
    named = [_MONTHS[t] for t in toks if t in _MONTHS]
    if named:
        hits = [c for c in candidates if int(c.split("-")[1]) in named]
        if len(hits) == 1:
            return hits[0]
    low = (text or "").lower()
    if re.search(r"\b(first one|the first|former)\b", low):
        return candidates[0]
    if re.search(r"\b(second one|the second|latter)\b", low):
        return candidates[1]
    return None
