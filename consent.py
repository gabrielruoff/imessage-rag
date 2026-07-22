"""Shared consent helper: parse an opt-in list of `contacts.id` values from a
raw env-var string. Pure stdlib, deterministic, config-free -- this module
imports nothing from the app (no `config`, no sqlite, no Ollama). Each feature
owns its own env var and passes the raw string in; this helper is agnostic
about which list it is parsing.

Two membership semantics share one parser. `is_consented` is the ALLOWLIST
check (default-closed): an id present is opted-in, absence means not consented,
an empty list consents to nothing -- it only ever ADDS a contact, and callers
apply their own exclusions first (consent never overrides an exclusion).
`is_denied` is the DENYLIST counterpart (default-open): an id present is
suppressed, absence means allowed, an empty list denies nothing. Which one a
feature wants is the caller's choice; both are pure membership over the same
tolerant `parse_consent_ids`.

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
    """ALLOWLIST membership (default-closed): whether `contact_id` appears in the
    parsed list, i.e. is opted in.

    `None` contact_id -> False. Consent only ADDS a contact and never overrides
    a caller's exclusion.
    """
    if contact_id is None:
        return False
    return contact_id in parse_consent_ids(raw)


def is_denied(contact_id: int | None, raw: str | None) -> bool:
    """DENYLIST membership (default-open): whether `contact_id` appears in the
    parsed list, i.e. is explicitly suppressed. The complement of `is_consented`
    for features that act by default and let the owner block specific ids -- an
    empty list denies nothing, so absence means allowed.

    `None` contact_id -> False (a self / no-identity read is never denied).
    """
    if contact_id is None:
        return False
    return contact_id in parse_consent_ids(raw)


def parse_member_block_ids(raw: str | None) -> frozenset[str]:
    """Parse a denylist keyed by `_shared.member_key` strings, not bare ints.

    Unlike `parse_consent_ids` (bare `contacts.id` ints only), this accepts:
      - a bare positive int -> `"contact:N"` (back-compat: the historical
        PERSONA_SIM_BLOCK_IDS format was contact ids), and
      - an explicit `"contact:N"` / `"handle:N"` key (case-insensitive kind),
    so a person who exists ONLY as an unlinked handle (no `contacts` row) can be
    denylisted too. Tolerant like the int parser: junk tokens are skipped, never
    raised; non-positive ids are dropped; duplicates collapse.
    """
    if not raw:
        return frozenset()
    keys = set()
    for token in _SEP.split(raw.strip()):
        if not token:
            continue
        kind, sep, num = token.lower().partition(":")
        if sep:  # "contact:12" / "handle:70"
            if kind not in ("contact", "handle"):
                continue
        else:    # bare int -> a contact id, for back-compat
            kind, num = "contact", token
        try:
            value = int(num)
        except ValueError:
            continue
        if value > 0:
            keys.add(f"{kind}:{value}")
    return frozenset(keys)


def is_blocked_member(person_key: str | None, raw: str | None) -> bool:
    """DENYLIST membership over a `member_key` string ("contact:N"/"handle:N").

    True iff `person_key` is present in `parse_member_block_ids(raw)`. `None`/
    empty person_key -> False. Bare ints in `raw` are treated as contact ids, so
    an existing `"12"` still blocks `"contact:12"`.
    """
    if not person_key:
        return False
    return person_key in parse_member_block_ids(raw)
