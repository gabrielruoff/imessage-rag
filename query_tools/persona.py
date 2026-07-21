"""Persona query tools.

V1 exposes get_voiceprint ONLY: read a person's precomputed deterministic
voiceprint from person_voiceprint (built by voiceprint.py), or recompute it
over a date window on the fly (capped). Later issues add draft_reply (V2),
style-match (V3) and simulate (V5) here -- this module stays a thin read layer
over the voiceprint the batch CLI computes; it never runs an LLM.

The voiceprint answers "how does WHOSE voice sound, WITH WHOM". The subject
defaults to "me": name a person to get YOUR voice with them (a dyad), or omit
to get your overall voice (scope='all'). subject='them' returns the named
person's voice instead. Group chats surface only when scope='group' is asked
for explicitly, with a group chat_id.
"""

import contextlib
import json

import config
from function_words import FUNCTION_WORDS

from ._shared import (
    _connect,
    _normalize_date_bound,
    member_key,
)

# Feature key sets, kept in step with voiceprint.py / person_voiceprint.
_STYLE_FEATURES = ("avg_msg_len", "lowercase_ratio", "exclaim_rate",
                   "ellipsis_rate", "burst_size")
_FW_COLUMNS = tuple(f"fw_{cat}_rate" for cat in FUNCTION_WORDS)


def _table_exists(con):
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='person_voiceprint'"
    ).fetchone() is not None


def _confidence(n_messages, n_reply_pairs, n_months):
    """The card confidence gate: 'high' only when volume, two-sided engagement,
    and breadth-over-time ALL clear their thresholds (read at call time so
    tests/.env can move them), else 'low'."""
    if (n_messages >= config.VOICEPRINT_MIN_MESSAGES
            and n_reply_pairs >= config.VOICEPRINT_MIN_REPLY_PAIRS
            and n_months >= config.VOICEPRINT_MIN_MONTHS):
        return "high"
    return "low"


def _contact_label(con, contact_id):
    row = con.execute(
        "SELECT COALESCE("
        "NULLIF(nickname, ''), "
        "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), '')"
        ") FROM contacts WHERE id = ?", (contact_id,)
    ).fetchone()
    if row and row[0]:
        return row[0]
    handle = con.execute(
        "SELECT handle_identifier FROM handles WHERE contact_id = ? "
        "ORDER BY id LIMIT 1", (contact_id,)
    ).fetchone()
    return handle[0] if handle else f"contact {contact_id}"


def _label_for_key(con, key):
    """Human-readable label for a member_key / counterpart_key string."""
    if key == "me":
        return "Me"
    if key == "*":
        return "everyone"
    kind, _, raw = key.partition(":")
    if kind == "contact":
        return _contact_label(con, int(raw))
    if kind == "handle":
        row = con.execute(
            "SELECT handle_identifier FROM handles WHERE id = ?", (int(raw),)
        ).fetchone()
        return row[0] if row else key
    if kind == "chat":
        row = con.execute(
            "SELECT COALESCE(NULLIF(display_name, ''), chat_identifier) "
            "FROM chats WHERE id = ?", (int(raw),)
        ).fetchone()
        return (row[0] if row and row[0] else f"group chat {raw}")
    return key


def _resolve_person(con, contact_id, handle_id, name):
    """Resolve a person param into (member_key, label, error_dict). All three
    None means no person was named. error_dict (non-None) is returned to the
    caller verbatim -- an unknown handle or an ambiguous name never guesses."""
    if contact_id is not None:
        return f"contact:{contact_id}", _contact_label(con, contact_id), None
    if handle_id is not None:
        row = con.execute(
            "SELECT contact_id, handle_identifier FROM handles WHERE id = ?",
            (handle_id,)
        ).fetchone()
        if row is None:
            return None, None, {"error": f"no handle with id {handle_id}"}
        cid, ident = row[0], row[1]
        key = member_key(False, cid, handle_id)
        label = _contact_label(con, cid) if cid is not None else ident
        return key, label, None
    if name:
        like = f"%{name}%"
        rows = con.execute(
            "SELECT id, COALESCE("
            "NULLIF(nickname, ''), "
            "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), ''), "
            "'') AS label FROM contacts "
            "WHERE first_name LIKE ? OR last_name LIKE ? OR nickname LIKE ? "
            "OR TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')) LIKE ? "
            "ORDER BY last_name, first_name LIMIT 10",
            (like, like, like, like)
        ).fetchall()
        if not rows:
            return None, None, {"error": f"no contact matches {name!r}"}
        if len(rows) > 1:
            return None, None, {
                "error": f"{len(rows)} contacts match {name!r}; pass contact_id",
                "matches": [{"contact_id": r[0], "label": r[1] or f"contact {r[0]}"}
                            for r in rows],
            }
        return f"contact:{rows[0][0]}", (rows[0][1] or f"contact {rows[0][0]}"), None
    return None, None, None


def _tone_block(valence, emotions_json):
    """The low-trust tone block, or None when the scope had no source='llm'
    annotations (both fields NULL) -- tone is then omitted from the card, never
    reported as a spurious neutral."""
    if valence is None and not emotions_json:
        return None
    try:
        emotions = json.loads(emotions_json) if emotions_json else []
    except (TypeError, ValueError):
        emotions = []
    return {"valence": valence, "emotions": emotions, "low_trust": True}


def _base_card(con, member_key_str, counterpart_key, scope, n_messages,
               n_reply_pairs, n_months, features, openers, closers,
               feature_std, tone):
    card = {
        "member_key": member_key_str,
        "counterpart_key": counterpart_key,
        "scope": scope,
        "member": _label_for_key(con, member_key_str),
        "counterpart": _label_for_key(con, counterpart_key),
        "confidence": _confidence(n_messages, n_reply_pairs, n_months),
        "n_messages": n_messages,
        "n_reply_pairs": n_reply_pairs,
        "n_months": n_months,
        "features": features,
        "openers": openers,
        "closers": closers,
        "feature_std": feature_std,
    }
    if tone is not None:
        card["tone"] = tone
    return card


def _card_from_row(con, row):
    features = {f: row[f] for f in _STYLE_FEATURES}
    features.update({c: row[c] for c in _FW_COLUMNS})
    return _base_card(
        con, row["member_key"], row["counterpart_key"], row["scope"],
        row["n_messages"], row["n_reply_pairs"], row["n_months"],
        features,
        json.loads(row["openers_json"] or "[]"),
        json.loads(row["closers_json"] or "[]"),
        json.loads(row["feature_std_json"] or "{}"),
        _tone_block(row["tone_valence"], row["tone_emotions_json"]),
    )


def _lookup(con, member_key_str, counterpart_key, scope):
    if not _table_exists(con):
        return {"note": "person_voiceprint not built yet -- run `make voiceprint`.",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "found": False}
    row = con.execute(
        "SELECT * FROM person_voiceprint "
        "WHERE member_key = ? AND counterpart_key = ? AND scope = ?",
        (member_key_str, counterpart_key, scope)
    ).fetchone()
    if row is None:
        return {"note": f"No voiceprint on file for member={member_key_str}, "
                        f"counterpart={counterpart_key}, scope={scope} "
                        "(too little interaction, or run `make voiceprint`).",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "found": False}
    return _card_from_row(con, row)


def _precomputed_std(con, member_key_str, counterpart_key, scope):
    """The stored per-scope feature std for this key -- a corpus property, so a
    date-scoped recompute reuses it rather than rescanning the whole corpus."""
    if not _table_exists(con):
        return {}
    row = con.execute(
        "SELECT feature_std_json FROM person_voiceprint "
        "WHERE member_key = ? AND counterpart_key = ? AND scope = ?",
        (member_key_str, counterpart_key, scope)
    ).fetchone()
    return json.loads(row[0] or "{}") if row else {}


def _recompute(con, member_key_str, counterpart_key, scope, chat_id, person_key,
               date_from, date_to):
    """Date-scoped path: recompute the features over a capped raw scan rather
    than reading the (whole-corpus) precomputed row."""
    import voiceprint  # lazy: avoids a query_tools <-> voiceprint import cycle

    df = _normalize_date_bound(date_from) if date_from is not None else None
    dt = _normalize_date_bound(date_to, upper=True) if date_to is not None else None

    if scope == "dyad":
        chats = voiceprint.member_chats(con, person_key, direct_only=True)
    elif scope == "group":
        chats = [chat_id]
    else:  # all
        chats = voiceprint.member_chats(con, member_key_str)

    cap = config.VOICEPRINT_SCAN_MAX_MESSAGES
    scan_n = voiceprint.scan_size(con, chats, df, dt)
    if scan_n > cap:
        return {"note": f"Date range too wide: {scan_n} messages exceed the "
                        f"{cap}-message scan cap -- narrow date_from/date_to.",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "messages_in_range": scan_n, "recomputed": False}

    result = voiceprint.compute_voiceprint(
        con, member_key_str, chats, date_from=df, date_to=dt, budget=cap)
    if result is None:
        return {"note": "No messages in the requested date range.",
                "member_key": member_key_str, "counterpart_key": counterpart_key,
                "scope": scope, "recomputed": True, "n_messages": 0}

    features = {f: result[f] for f in _STYLE_FEATURES}
    features.update(result["fw"])
    card = _base_card(
        con, member_key_str, counterpart_key, scope,
        result["n_messages"], result["n_reply_pairs"], result["n_months"],
        features, result["openers"], result["closers"],
        _precomputed_std(con, member_key_str, counterpart_key, scope),
        _tone_block(result["tone_valence"],
                    json.dumps(result["tone_emotions"])
                    if result["tone_emotions"] is not None else None),
    )
    card["recomputed"] = True
    card["date_from"] = date_from
    card["date_to"] = date_to
    if result["n_messages"] < config.VOICEPRINT_MIN_RANGE_MESSAGES:
        card["note"] = (f"Only {result['n_messages']} messages in range "
                        f"(< {config.VOICEPRINT_MIN_RANGE_MESSAGES}); this "
                        "voiceprint may be unreliable.")
    return card


def get_voiceprint(contact_id=None, handle_id=None, name=None, subject="me",
                   scope=None, chat_id=None, date_from=None, date_to=None):
    """A person's deterministic voiceprint -- how they text.

    Defaults to YOUR voice (subject='me'): name a person
    (contact_id/handle_id/name) to get your voice WITH them (scope='dyad'), or
    omit the person for your overall voice (scope='all'). subject='them'
    returns the named person's voice instead. scope='group' needs a group
    chat_id (group rows are never returned unless asked for). Passing
    date_from/date_to recomputes the card over that window from a capped raw
    scan; otherwise the precomputed person_voiceprint row is returned.

    The card carries the style/function-word features, signature
    openers/closers, a low-trust tone block (only when LLM chunk annotations
    exist for the scope), coverage counts, a confidence flag (high/low), and
    the per-feature corpus std (feature_std) for downstream normalization."""
    subject = (subject or "me").lower()
    if subject not in ("me", "them"):
        return {"error": "subject must be 'me' or 'them'"}

    with contextlib.closing(_connect()) as con:
        person_key, _label, err = _resolve_person(con, contact_id, handle_id, name)
        if err:
            return err

        if scope is None:
            scope = "dyad" if person_key is not None else "all"
        scope = str(scope).lower()
        if scope not in ("dyad", "group", "all"):
            return {"error": f"scope must be dyad, group, or all (got {scope!r})"}

        if scope == "dyad":
            if person_key is None:
                return {"error": "scope='dyad' needs a person "
                                 "(contact_id/handle_id/name)"}
            member_key_str, counterpart_key = (
                ("me", person_key) if subject == "me" else (person_key, "me"))
        elif scope == "all":
            counterpart_key = "*"
            if subject == "me":
                member_key_str = "me"
            elif person_key is None:
                return {"error": "scope='all' with subject='them' needs a person"}
            else:
                member_key_str = person_key
        else:  # group
            if chat_id is None:
                return {"error": "scope='group' needs a group chat_id"}
            counterpart_key = f"chat:{chat_id}"
            if subject == "me":
                member_key_str = "me"
            elif person_key is None:
                return {"error": "scope='group' with subject='them' needs a person"}
            else:
                member_key_str = person_key

        if date_from is not None or date_to is not None:
            return _recompute(con, member_key_str, counterpart_key, scope,
                              chat_id, person_key, date_from, date_to)
        return _lookup(con, member_key_str, counterpart_key, scope)
