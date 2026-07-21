"""Relationship-drift read tools (RES-59 / D2): get_relationship_drift (one
resolved counterparty's drift read) and get_fadeout_watchlist (a coverage-
partitioned leaderboard of fading ties).

These READ the deterministic drift pipeline's output (the `dyad_drift` table,
written by the root drift.py) -- no re-scoring, no Ollama. There is a
deliberate twin: this `query_tools/drift.py` is the read layer; the root
`drift.py` is the compute pipeline. They share a name but NEVER import each
other (this module keeps its own local copy of the signal-name tuple rather
than importing drift.SIGNALS).

Identity contract: `dyad_drift` is keyed by member_key's stable string
("contact:<id>" when a handle resolves to a contact, else "handle:<id>"), so a
person param resolves the SAME way it was stored -- contact_id -> "contact:<id>";
a bare handle_id -> its contact if it has one, else "handle:<id>"; a 1:1
chat_id -> that chat's single counterparty handle, then the same rule (a group
chat_id is an error -- drift is one relationship at a time).

Two load-bearing rules live HERE, in the read layer, so narration can't drop
them:
  * The "still talk elsewhere?" channel caveat is CODE-EMITTED on every
    cooling/fading/dormant payload (part of the data, not a prompt hope).
  * The watchlist NEVER ranks across coverage: mandatory-complete dyads form
    the rankable tier; everything else is a separate `insufficient_signal`
    bucket, sorted only within itself.

Evidence trails are read verbatim from D1's `monthly_series_json` (never
recomputed) and trimmed to the most recent _MAX_TRAIL_MONTHS months so the
payload is compact by construction -- the chat.run_turn TOOL_RESULT_MAX_CHARS
backstop should never have to lossily truncate one of these.
"""

import contextlib
import json

import config

from ._shared import _as_id_list, _connect, member_key

# The seven drift signals, in the pipeline's order. Duplicated locally on
# purpose: the read layer must not import the root drift.py twin.
_SIGNALS = ("volume", "reciprocity", "latency", "silence",
            "valence", "depth", "initiation")

# Phases whose read carries the iMessage-only channel caveat (a decline label).
_COOLING_PHASES = ("cooling", "fading", "dormant")

# Compactness caps (this tool's own, like analytics_social's eligibility gates).
# A monthly trail is trimmed to its most recent months; the watchlist buckets
# are row-capped. All comfortably under TOOL_RESULT_MAX_CHARS.
_MAX_TRAIL_MONTHS = 12
MAX_WATCHLIST_ROWS = 25

# Narration templates keyed by the PRE-COMPUTED confidence_tier: the model
# narrates at the tier the pipeline assigned, never phrasing more confidently
# than coverage warrants (D1 computes the tier; the read layer only relays it).
_TIER_TEMPLATE = {
    "high": "Well-covered signals -- you may state the trend directly.",
    "medium": "Partial coverage -- present the trend as a tentative read, "
              "not a conclusion.",
    "low": "Thin coverage -- describe this as a weak, low-confidence signal "
           "only; do not phrase it as certain.",
}


def _has_drift_table(con):
    """True only when dyad_drift exists -- a DB built before this feature (or
    with DRIFT_ENABLED off) has none, and the read tools return a note, not a
    traceback."""
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dyad_drift'"
    ).fetchone() is not None


def _label_for_key(con, key, stored):
    """Human-readable label for a member_key. Prefer the label D1 stored (it is
    CONTACT_LABEL_SQL, the same resolution every live aggregate uses); fall back
    to a fresh lookup, then to the key string itself."""
    if stored:
        return stored
    kind, _, raw = key.partition(":")
    if kind == "contact" and raw.isdigit():
        row = con.execute(
            "SELECT COALESCE("
            "NULLIF(nickname, ''), "
            "NULLIF(TRIM(COALESCE(first_name, '') || ' ' || COALESCE(last_name, '')), '')"
            ") FROM contacts WHERE id = ?", (int(raw),)).fetchone()
        if row and row[0]:
            return row[0]
    if kind == "handle" and raw.isdigit():
        row = con.execute(
            "SELECT handle_identifier FROM handles WHERE id = ?", (int(raw),)
        ).fetchone()
        if row and row[0]:
            return row[0]
    return key


def _resolve_contact_key(con, contact_id, handle_id, chat_id):
    """Resolve exactly one of (contact_id | handle_id | chat_id) into the
    member_key string dyad_drift stores. Returns (key, error_dict); error_dict
    (non-None) is returned to the caller verbatim -- an unknown/ambiguous input
    never guesses. Id params accept a scalar or a list; drift is one
    relationship at a time, so the first id is used."""
    cids = _as_id_list(contact_id)
    hids = _as_id_list(handle_id)
    chids = _as_id_list(chat_id)
    provided = [p for p in (cids, hids, chids) if p]
    if not provided:
        return None, {"error": "pass one of contact_id, handle_id, or chat_id "
                               "to identify the relationship"}
    if len(provided) > 1:
        return None, {"error": "pass only ONE of contact_id, handle_id, or "
                               "chat_id -- drift is one relationship at a time"}
    if cids:
        return f"contact:{cids[0]}", None
    if hids:
        return _key_from_handle(con, hids[0])
    return _key_from_chat(con, chids[0])


def _key_from_handle(con, handle_id):
    """A bare handle -> its contact key ('contact:<id>') when it resolves to a
    contact, else 'handle:<id>' (member_key does this merge, the same one D1
    keyed on)."""
    row = con.execute(
        "SELECT contact_id FROM handles WHERE id = ?", (handle_id,)).fetchone()
    if row is None:
        return None, {"error": f"no handle with id {handle_id}"}
    return member_key(False, row[0], handle_id), None


def _key_from_chat(con, chat_id):
    """A 1:1 chat -> its single counterparty's key. A group chat (or an oddball
    direct chat with more than one participant handle) has no single
    counterparty, so it is an explanatory error rather than a silent guess --
    matching drift.py's _chat_to_key, which skips exactly those chats."""
    row = con.execute(
        "SELECT is_group FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if row is None:
        return None, {"error": f"no chat with id {chat_id}"}
    if row[0]:
        return None, {"error": f"chat {chat_id} is a group chat; relationship "
                               "drift is per 1:1 relationship -- pass a "
                               "contact_id or handle_id instead"}
    parts = con.execute(
        "SELECT cp.handle_id, h.contact_id FROM chat_participants cp "
        "JOIN handles h ON h.id = cp.handle_id WHERE cp.chat_id = ?",
        (chat_id,)).fetchall()
    if len(parts) != 1:
        return None, {"error": f"chat {chat_id} has no single counterparty "
                               f"(found {len(parts)} participant handles) -- "
                               "pass a contact_id or handle_id instead"}
    handle_id, contact_id = parts[0]
    return member_key(False, contact_id, handle_id), None


def _contributing_signals(row, monthly_series):
    """The present signals as ranked contributors, each with its monthly
    evidence trail READ VERBATIM from D1's monthly_series_json (never
    recomputed) and its gate verdict. A signal with no stored series was absent
    from scoring, so it is skipped. Ranked by drift contribution, ties broken by
    name for determinism; the trail is trimmed to the most recent months."""
    out = []
    for s in _SIGNALS:
        trail = monthly_series.get(s)
        if not trail:
            continue
        out.append({
            "signal": s,
            "drift_value": row[f"{s}_value"],
            "gate": bool(row[f"{s}_gate"]),  # a significant trend was detected
            "monthly": [[m, v] for m, v in trail[-_MAX_TRAIL_MONTHS:]],
        })
    out.sort(key=lambda e: (-(e["drift_value"] or 0.0), e["signal"]))
    return out


def _top_driver(row):
    """The single signal contributing most drift (max *_value > 0), or None when
    nothing contributes. Used by the watchlist rows for an at-a-glance 'why'."""
    best = None
    for s in _SIGNALS:
        val = row[f"{s}_value"] or 0.0
        if val > 0 and (best is None or val > best["value"]):
            best = {"signal": s, "value": val, "gate": bool(row[f"{s}_gate"])}
    return best


def _gated_signal_count(row):
    """How many signals detected a significant trend (the *_gate flags)."""
    return sum(1 for s in _SIGNALS if row[f"{s}_gate"])


def _channel_caveat(phase, name):
    """The code-emitted 'do you still talk elsewhere?' caveat -- present on every
    cooling/fading/dormant payload so narration can't omit it. iMessage is one
    channel; a quiet iMessage tie may be loud on calls/IG/in person."""
    if phase not in _COOLING_PHASES:
        return None
    return {
        "confirmed_imessage_only": False,
        "ask": (f"Do you still talk with {name} somewhere else (calls, IG, "
                "in person)? This read is from iMessage only."),
    }


def _reconnection_window(last_message_days, phase):
    """A grounded reconnection hint from the current gap vs the dormant
    threshold (read at call time from config, the repo convention) -- not a
    recomputation, just framing of the stored gap."""
    thresholds = config.DRIFT_PHASE_THRESHOLDS
    dormant_days = float(thresholds.get("dormant_days", 90))
    if last_message_days is None:
        return {"current_gap_days": None, "dormant_threshold_days": dormant_days,
                "days_until_dormant": None, "status": "unknown"}
    days_until_dormant = round(dormant_days - last_message_days)
    if phase == "dormant" or days_until_dormant <= 0:
        status = "overdue"
    elif days_until_dormant <= 14:
        status = "closing"
    else:
        status = "open"
    return {
        "current_gap_days": last_message_days,
        "dormant_threshold_days": dormant_days,
        "days_until_dormant": days_until_dormant,
        "status": status,
    }


def _narration(tier):
    return {"confidence_tier": tier,
            "template": _TIER_TEMPLATE.get(tier, _TIER_TEMPLATE["low"])}


def get_relationship_drift(contact_id=None, handle_id=None, chat_id=None):
    """How one 1:1 relationship is trending over time, read from the drift
    pipeline (dyad_drift) -- no re-scoring. Identify the relationship with
    exactly ONE of contact_id / handle_id / chat_id (each a scalar or list; a
    group chat_id is an error). Returns a compact dict: the resolved `contact`,
    `phase` and `phase_corroborated`, `drift_score`, the ranked contributing
    `signals` (each with its monthly evidence trail and gate verdict, straight
    from the stored series), `gap` (current gap days vs its ECDF percentile),
    `reconnection_window`, `confidence_tier` with a `narration` template, and
    `coverage`. When the phase is cooling/fading/dormant the payload carries a
    `channel_caveat`: relay its ask (do you still talk elsewhere?) BEFORE
    characterizing the tie, and narrate at the confidence_tier template."""
    with contextlib.closing(_connect()) as con:
        if not _has_drift_table(con):
            return {"error": "no dyad_drift table -- build it with drift.py "
                             "(set DRIFT_ENABLED=true)"}
        key, err = _resolve_contact_key(con, contact_id, handle_id, chat_id)
        if err:
            return err
        row = con.execute(
            "SELECT * FROM dyad_drift WHERE contact_key = ?", (key,)).fetchone()
        if row is None:
            return {
                "contact_key": key,
                "found": False,
                "note": "no drift row for this relationship -- it is below the "
                        "inclusion gates (too few messages/months), or drift "
                        "has not been built for it yet",
            }

        coverage = json.loads(row["coverage_json"] or "{}")
        monthly_series = json.loads(row["monthly_series_json"] or "{}")
        label = _label_for_key(con, key, row["label"])
        phase = row["phase"]
        signals = _contributing_signals(row, monthly_series)

        silence_trail = monthly_series.get("silence") or []
        gap = {
            "current_gap_days": coverage.get("last_message_days"),
            # The most recent month's longest-gap ECDF placement (0..1): how
            # extreme the current silence is against this dyad's own history.
            "gap_ecdf_pct": (silence_trail[-1][1] if silence_trail else None),
        }

        payload = {
            "contact_key": key,
            "contact": label,
            "found": True,
            "phase": phase,
            "phase_corroborated": bool(row["phase_corroborated"]),
            "drift_score": row["drift_score"],
            "confidence_tier": row["confidence_tier"],
            "narration": _narration(row["confidence_tier"]),
            "n_messages": row["n_messages"],
            "n_months": row["n_months"],
            "first_month": row["first_month"],
            "last_month": row["last_month"],
            "signals": signals,
            "gap": gap,
            "reconnection_window": _reconnection_window(
                coverage.get("last_message_days"), phase),
            "coverage": coverage,
        }
        caveat = _channel_caveat(phase, label)
        if caveat is not None:
            payload["channel_caveat"] = caveat
        return payload


def _watchlist_row(con, row):
    """One compact leaderboard row: the resolved contact, phase, score, tier,
    its gated signal count, and its top driver -- enough to rank and explain
    without the full evidence trail."""
    coverage = json.loads(row["coverage_json"] or "{}")
    label = _label_for_key(con, row["contact_key"], row["label"])
    phase = row["phase"]
    entry = {
        "contact_key": row["contact_key"],
        "contact": label,
        "phase": phase,
        "phase_corroborated": bool(row["phase_corroborated"]),
        "drift_score": row["drift_score"],
        "confidence_tier": row["confidence_tier"],
        "signal_count": _gated_signal_count(row),
        "top_driver": _top_driver(row),
        "n_messages": row["n_messages"],
        "last_month": row["last_month"],
        "current_gap_days": coverage.get("last_message_days"),
    }
    caveat = _channel_caveat(phase, label)
    if caveat is not None:
        entry["channel_caveat"] = caveat
    return entry


def _normalize_phases(phase_filter):
    """Normalize phase_filter (None, a scalar, or a list) into a lowercased set
    of phase names, or None for no filter."""
    if phase_filter is None:
        return None
    vals = (phase_filter if isinstance(phase_filter, (list, tuple))
            else [phase_filter])
    normalized = {str(p).strip().lower() for p in vals if str(p).strip()}
    return normalized or None


def get_fadeout_watchlist(limit=10, min_messages=0, phase_filter=None):
    """A leaderboard of relationships fading out, read from dyad_drift and
    PARTITIONED BY COVERAGE -- it never ranks across coverage. `watchlist` holds
    the rankable tier (mandatory signals all present), ordered by drift_score;
    `insufficient_signal` is a SEPARATE bucket (a mandatory signal missing) that
    is only ordered within itself and must never be compared to the ranked tier.
    Each row carries its gated `signal_count` and `top_driver`, plus the channel
    caveat when its phase is cooling/fading/dormant. min_messages floors the
    dyad size; phase_filter (scalar or list) narrows to given phases; limit caps
    each bucket."""
    limit = max(1, min(int(limit), MAX_WATCHLIST_ROWS))
    min_messages = max(0, int(min_messages))
    phases = _normalize_phases(phase_filter)
    order = "drift_score-desc WITHIN the mandatory-complete tier only"

    with contextlib.closing(_connect()) as con:
        if not _has_drift_table(con):
            return {"watchlist": [], "insufficient_signal": [], "order": order,
                    "min_messages": min_messages,
                    "phase_filter": sorted(phases) if phases else None,
                    "note": "no dyad_drift table -- build it with drift.py "
                            "(set DRIFT_ENABLED=true)"}
        rows = con.execute(
            "SELECT * FROM dyad_drift WHERE n_messages >= ?",
            (min_messages,)).fetchall()

        ranked, insufficient = [], []
        for row in rows:
            if phases is not None and (row["phase"] or "").lower() not in phases:
                continue
            entry = _watchlist_row(con, row)
            (ranked if row["mandatory_complete"] else insufficient).append(entry)

    # Rank each bucket ONLY within itself -- the two are never merged, so a
    # low-coverage dyad's high score can never outrank a mandatory-complete one.
    sort_key = lambda e: (-(e["drift_score"] or 0.0), -e["signal_count"],  # noqa: E731
                          e["contact_key"])
    ranked.sort(key=sort_key)
    insufficient.sort(key=sort_key)
    return {
        "watchlist": ranked[:limit],
        "insufficient_signal": insufficient[:limit],
        "order": order,
        "min_messages": min_messages,
        "phase_filter": sorted(phases) if phases else None,
        "watchlist_total": len(ranked),
        "insufficient_signal_total": len(insufficient),
        "note": "the two buckets are never ranked against each other: "
                "insufficient_signal rows are missing a mandatory signal and "
                "are not comparable to the ranked watchlist",
    }
