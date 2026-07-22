"""Loose Ends (nudges) query tools: the three LLM-callable functions over the
loose_threads / nudge_exclusions tables, plus the small exclusion-store helpers
shared with the loose_ends.py detector CLI.

  - get_loose_threads       read the current nudge digest (capped, urgency-ranked)
  - resolve_loose_thread    dismiss one thread (sets dismissed_at; never deletes)
  - mute_contact_from_nudges permanently exclude a contact AND dismiss their open threads

The detectors live in loose_ends.py; this module never generates threads, it
only reads them and records dismissals/exclusions. Exclusion durability: rows in
nudge_exclusions store contacts.id / handles.id (source PKs, stable for the same
source DB). reset_db preserves them; they dangle harmlessly until re-ingest, and
a stale exclusion fails OPEN (the contact simply reappears) -- acceptable for v1.
"""

import contextlib
import datetime
import json

from config import NUDGE_MAX_RESULTS
from db import CONTACT_LABEL_SQL, get_state

from ._shared import _connect

# get_loose_threads is capped at NUDGE_MAX_RESULTS regardless of what the caller
# asks for -- a nudge digest is a handful, not a feed.


def _utcnow_iso():
    """Event timestamp (real wall-clock UTC) for created_at/detected_at/
    dismissed_at columns. Distinct from message `date` values, which are the
    corpus's own unlabeled-UTC strings."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _load_exclusions(con):
    """Return (excluded_handle_ids, excluded_contact_ids) sets from
    nudge_exclusions. Either side of a row may be NULL; NULLs are simply not
    added to the sets."""
    handles, contacts = set(), set()
    for handle_id, contact_id in con.execute(
        "SELECT handle_id, contact_id FROM nudge_exclusions"
    ):
        if handle_id is not None:
            handles.add(handle_id)
        if contact_id is not None:
            contacts.add(contact_id)
    return handles, contacts


def _is_excluded(handle_id, contact_id, excluded_handles, excluded_contacts):
    """True when a candidate's counterparty is muted -- by its handle or by the
    contact that handle belongs to."""
    return (handle_id is not None and handle_id in excluded_handles) or (
        contact_id is not None and contact_id in excluded_contacts
    )


def _exclusion_exists(con, handle_id, contact_id, reason):
    """NULL-safe existence check (uses IS, not =) so a duplicate mute is caught
    even when contact_id/handle_id is NULL -- a UNIQUE constraint alone would
    not, since NULLs compare distinct there."""
    return con.execute(
        "SELECT 1 FROM nudge_exclusions "
        "WHERE handle_id IS ? AND contact_id IS ? AND reason IS ?",
        (handle_id, contact_id, reason),
    ).fetchone() is not None


def _add_exclusion(con, handle_id=None, contact_id=None, reason="muted", now=None,
                   persist=True):
    """Insert one nudge_exclusions row unless an identical (handle_id,
    contact_id, reason) already exists. Returns True if a row was (or, with
    persist=False, would have been) added. With persist=False the existence
    check still runs but the INSERT is skipped -- so a --dry-run caller gets the
    same would-add count without mutating the table. Does NOT commit -- the
    caller owns the transaction."""
    if _exclusion_exists(con, handle_id, contact_id, reason):
        return False
    if persist:
        con.execute(
            "INSERT INTO nudge_exclusions(handle_id, contact_id, reason, created_at) "
            "VALUES (?, ?, ?, ?)",
            (handle_id, contact_id, reason, now or _utcnow_iso()),
        )
    return True


def _pct(value):
    try:
        return f"{round(float(value) * 100)}%"
    except (TypeError, ValueError):
        return None


def _thread_summary(kind, contact, evidence):
    """A deterministic, plain-language description of ONE loose thread, keyed on
    its kind. Grounds the model's narration so it cannot invert who went quiet
    (faded_initiator is about the COUNTERPARTY's initiation, not the user's) or
    re-label the kind / guess an occasion type it wasn't told. Returns None for
    an unknown kind (the model falls back to the raw evidence)."""
    who = contact or "this contact"
    ev = evidence if isinstance(evidence, dict) else {}
    if kind == "faded_initiator":
        allt, trail = _pct(ev.get("all_time_initiation_share")), _pct(
            ev.get("trailing_initiation_share"))
        s = f"{who} used to start conversations with you"
        s += f" about {allt} of the time" if allt else " more than they do now"
        s += f", but only about {trail} recently" if trail else ", but has gone quiet lately"
        return s + " -- they have gone quiet, so reaching out could reconnect."
    if kind == "unanswered_question":
        q = (ev.get("question") or "").strip()
        return (f"{who} asked you a question you have not answered"
                + (f': "{q}"' if q else "") + ".")
    if kind == "inferred_occasion":
        md, yrs = ev.get("month_day"), ev.get("observed_years")
        sample = (str(ev.get("sample_message") or "")).strip()
        s = "A recurring date"
        s += f" around {md}" if md else ""
        s += f" you have marked with {who} in past years"
        s += f" ({yrs})" if yrs else ""
        # Include the actual past message verbatim so the type is not guessed
        # (a 'Congrats women' note is not a birthday).
        s += f'; a past message then was "{sample}"' if sample else ""
        return s + "."
    if kind == "unreciprocated_goodnews":
        return f"{who} shared good news that may not have been fully acknowledged."
    if kind == "dormant_tie":
        return f"You and {who} used to talk regularly but have not in a while."
    return None


def get_loose_threads(status="open", kind=None, limit=5):
    """The current loose-thread digest: nudges the detector found, ranked by
    urgency (highest first). status defaults to 'open' (pass 'dismissed' to
    review resolved ones); kind optionally narrows to one detector
    (e.g. 'faded_initiator'). limit is hard-capped at NUDGE_MAX_RESULTS -- the
    digest is meant to be a short, actionable list. Each row carries the
    resolved `contact` label, the anchor message/date, urgency, the parsed
    `evidence`, and the one-line `nudge_text` (NULL until narrated)."""
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = NUDGE_MAX_RESULTS
    limit = max(1, min(limit, NUDGE_MAX_RESULTS))

    clauses = ["lt.status = ?"]
    params = [status]
    if kind is not None:
        clauses.append("lt.kind = ?")
        params.append(kind)

    with contextlib.closing(_connect()) as con:
        # --- feature: nudges ---
        # First-run batch-hide (N2). Dormant-tie ("reconnect") nudges are a
        # sensitive class: they are DETECTED and written as normal open rows by
        # loose_ends.py, but stay hidden from this digest until the user opts in
        # ONCE via `loose_ends.py --acknowledge-reconnect` (which sets the
        # durable ingest_state key nudges_reconnect_ack). Absent key => filter
        # dormant_tie out of the open digest and surface a one-line notice
        # instead. Keeps the batch pipeline non-interactive; the opt-in is
        # explicit and durable.
        hide_dormant = status == "open" and get_state(con, "nudges_reconnect_ack") is None
        if hide_dormant:
            clauses.append("lt.kind != 'dormant_tie'")
        # --- end feature: nudges ---
        rows = con.execute(
            f"""
            SELECT lt.id, lt.kind, lt.chat_id, lt.handle_id, lt.anchor_message_id,
                   lt.anchor_date, lt.due_by, lt.urgency, lt.status, lt.evidence_json,
                   lt.nudge_text, lt.llm_verdict, lt.llm_reason, lt.detected_at,
                   lt.dismissed_at,
                   CASE WHEN lt.handle_id IS NULL THEN NULL ELSE {CONTACT_LABEL_SQL} END
                       AS contact
            FROM loose_threads lt
            LEFT JOIN handles h ON h.id = lt.handle_id
            LEFT JOIN contacts c ON c.id = h.contact_id
            WHERE {' AND '.join(clauses)}
            ORDER BY lt.urgency DESC, lt.anchor_date DESC, lt.id DESC
            LIMIT ?
            """,
            params + [limit],
        ).fetchall()
        # --- feature: nudges ---
        hidden_dormant = 0
        if hide_dormant and kind in (None, "dormant_tie"):
            hidden_dormant = con.execute(
                "SELECT COUNT(*) FROM loose_threads "
                "WHERE status = 'open' AND kind = 'dormant_tie'"
            ).fetchone()[0]
        # Population total for the SAME filter (no LIMIT), so a capped digest is
        # never mistaken for the whole list (mirrors _shared._envelope's rationale).
        n_total = con.execute(
            f"SELECT COUNT(*) FROM loose_threads lt WHERE {' AND '.join(clauses)}",
            params,
        ).fetchone()[0]
        # --- end feature: nudges ---

    out = []
    for r in rows:
        row = dict(r)
        row["evidence"] = json.loads(row.pop("evidence_json") or "null")
        # Deterministic, unambiguous per-kind description (grounds narration:
        # who went quiet, and the real kind -- not a guessed one).
        row["summary"] = _thread_summary(row["kind"], row.get("contact"),
                                         row["evidence"])
        out.append(row)
    # --- feature: nudges ---
    if hidden_dormant:
        out.append({
            "notice": f"{hidden_dormant} dormant-tie reconnect nudge(s) hidden. "
                      "Run `loose_ends.py --acknowledge-reconnect` to include them "
                      "in the digest.",
            "kind": "dormant_tie",
            "hidden": hidden_dormant,
        })
    # Completeness signal: this digest is urgency-ranked and hard-capped, so
    # expose the population total when it exceeds what was returned -- the model
    # must not claim it has seen every loose thread.
    n_returned = len(rows)
    if n_total > n_returned:
        out.append({
            "digest_meta": True,
            "n_total": n_total,
            "returned": n_returned,
            "truncated": True,
        })
    # --- end feature: nudges ---
    return out


def resolve_loose_thread(thread_id):
    """Dismiss one loose thread by its id: sets status='dismissed' and stamps
    dismissed_at. The row is never deleted -- a dismissed thread stays as a
    record and is what stops the detector from re-opening the same nudge."""
    now = _utcnow_iso()
    with contextlib.closing(_connect()) as con:
        cur = con.execute(
            "UPDATE loose_threads SET status = 'dismissed', dismissed_at = ? "
            "WHERE id = ? AND status != 'dismissed'",
            (now, thread_id),
        )
        changed = cur.rowcount
        exists = con.execute(
            "SELECT 1 FROM loose_threads WHERE id = ?", (thread_id,)
        ).fetchone()
        con.commit()
    if not exists:
        return {"thread_id": thread_id, "dismissed": False,
                "error": "no such loose thread"}
    return {"thread_id": thread_id, "dismissed": True, "status": "dismissed",
            "changed": changed > 0, "dismissed_at": now}


def mute_contact_from_nudges(contact_id=None, handle_id=None, reason="muted"):
    """Permanently exclude a contact (or a bare handle) from future nudges AND
    dismiss any of their currently-open loose threads. Writes a durable
    nudge_exclusions row (deduped -- muting twice never stacks rows) so no
    detector ever generates a candidate for them again. Provide contact_id
    (preferred -- covers every handle they own) and/or handle_id."""
    if contact_id is None and handle_id is None:
        raise ValueError("mute_contact_from_nudges needs contact_id or handle_id")
    now = _utcnow_iso()
    with contextlib.closing(_connect()) as con:
        added = _add_exclusion(con, handle_id=handle_id, contact_id=contact_id,
                               reason=reason, now=now)
        # Dismiss open threads for every handle this mute covers: the explicit
        # handle, plus all handles the contact owns.
        handle_ids = set()
        if handle_id is not None:
            handle_ids.add(handle_id)
        if contact_id is not None:
            handle_ids.update(r[0] for r in con.execute(
                "SELECT id FROM handles WHERE contact_id = ?", (contact_id,)))
        dismissed = 0
        if handle_ids:
            placeholders = ",".join("?" * len(handle_ids))
            dismissed = con.execute(
                f"UPDATE loose_threads SET status = 'dismissed', dismissed_at = ? "
                f"WHERE status = 'open' AND handle_id IN ({placeholders})",
                [now, *handle_ids],
            ).rowcount
        con.commit()
    return {"muted": True, "exclusion_added": added, "dismissed_threads": dismissed,
            "contact_id": contact_id, "handle_id": handle_id, "reason": reason}
