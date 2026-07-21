"""Shared consent helper: parse an opt-in list of `contacts.id` values from a
raw env-var string. Pure stdlib, deterministic, config-free -- this module
imports nothing from the app (no `config`, no sqlite, no Ollama). Each feature
owns its own env var and passes the raw string in; this helper is agnostic
about which list it is parsing.

Consent is additive, explicit, and default-closed: an id present in the list
is opted-in; absence means not consented. An empty list consents to nothing.
Membership (`is_consented`) only ever ADDS a contact -- callers apply their own
exclusions first, and consent never overrides an exclusion.

Parsing is deliberately tolerant: a malformed `.env` entry must never crash a
pipeline or a Wrapped build, so non-integer tokens are skipped rather than
raised. Zero and negative ids are skipped too (no `contacts.id` is <= 0), and
duplicates collapse.
"""

import re

# Split on commas and/or any run of whitespace, so "12, 44 88" and "12,44,88"
# and "12 44 88" all parse the same.
_SEP = re.compile(r"[,\s]+")


def parse_consent_ids(raw: str | None) -> frozenset[int]:
    """Parse a comma- and/or whitespace-separated list of `contacts.id` ints.

    Tolerant by design: `None`/empty/whitespace-only -> empty set; tokens that
    are not positive integers (non-numeric, zero, negative) are skipped, never
    raised; duplicates collapse.
    """
    if not raw:
        return frozenset()
    ids = set()
    for token in _SEP.split(raw.strip()):
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            ids.add(value)
    return frozenset(ids)


def is_consented(contact_id: int | None, raw: str | None) -> bool:
    """Return whether `contact_id` appears in the parsed consent list.

    `None` contact_id -> False. This is the single membership check every
    consumer calls; consent adds and never overrides a caller's exclusion.
    """
    if contact_id is None:
        return False
    return contact_id in parse_consent_ids(raw)
