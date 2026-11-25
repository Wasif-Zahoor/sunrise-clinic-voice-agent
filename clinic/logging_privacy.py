"""Keep caller speech and personal data out of application logs.

LiveKit tags personal fields in log records with names starting `lk.pii.`
(transcripts, participant identity, room name). At DEBUG level these include the full
conversation text. This filter replaces their values before any handler writes them.
Technical fields (timings, model names, error codes, tracebacks) are left untouched.
"""

from __future__ import annotations

import logging

PII_PREFIX = "lk.pii."
REDACTED = "[redacted]"


class RedactPII(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key in [k for k in record.__dict__ if k.startswith(PII_PREFIX)]:
            record.__dict__[key] = REDACTED
        return True


def install_pii_redaction() -> int:
    """Attach the filter to every root handler (idempotent). Returns handlers covered."""
    count = 0
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RedactPII) for f in handler.filters):
            handler.addFilter(RedactPII())
        count += 1
    return count
