"""Redaction helper (V2 §43), applied at every report boundary: audit report, the
approval summary, and the provenance record. The architecture already avoids
carrying raw secrets on the Canonical IR or workflow state (V2 §7 — only `secret_ref:
env://...` references travel through the system), so this is defense-in-depth: it catches
an accidental leak (e.g. an adapter dumping request headers into an error message), not
the primary control.
"""

from __future__ import annotations

import re

_REDACTED = "***REDACTED***"

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|token|password|passwd|credential|authorization|api[_-]?key|bearer)",
    re.IGNORECASE,
)

_INLINE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9]{10,}"), _REDACTED),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{10,}", re.IGNORECASE), rf"\1{_REDACTED}"),
    (
        re.compile(r"((?:api[_-]?key)\s*[:=]\s*)[\"']?[A-Za-z0-9._-]{6,}[\"']?", re.IGNORECASE),
        rf"\1{_REDACTED}",
    ),
]


def redact(value):
    """Recursively redacts dict values whose key looks sensitive; lists/scalars pass
    through unchanged (scalars have no key to judge by — redact_text handles those)."""
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _SENSITIVE_KEY_PATTERN.search(str(k)) else redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def redact_text(text: str) -> str:
    """Best-effort redaction of secret-shaped substrings inside free text (error messages,
    stack traces) — not a substitute for `redact()` on structured data."""
    redacted = text
    for pattern, repl in _INLINE_RULES:
        redacted = pattern.sub(repl, redacted)
    return redacted
