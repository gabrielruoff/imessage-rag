"""Shared dyad primitives + the detector registry for the Loose Ends pipeline.

Extracted from loose_ends.py to break the loose_ends <-> nudge_detectors import
cycle: running `python3 loose_ends.py` loads loose_ends as __main__, so a
nudge_detectors `from loose_ends import ...` re-executed the file under a second
name mid-import and failed (the second copy reached the re-export block before
nudge_detectors had defined its detectors). Both modules now import these from
here, giving a one-directional graph (loose_ends -> nudge_detectors ->
nudge_shared).
"""

import datetime
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable

from query_tools._shared import _REAL_MESSAGE_CLAUSES

# The shared real-message predicate, spelled once for the string joins below.
_REAL = " AND ".join(_REAL_MESSAGE_CLAUSES)


# ---------------------------------------------------------------------------
# Candidate rows + the detector registry
# ---------------------------------------------------------------------------

@dataclass
class CandidateRow:
    """A detector's output: the loose_threads column fields it is responsible
    for (minus id/urgency/status, which the driver owns). A detector fills in
    the detection-relevant fields; the driver computes urgency, sets
    status='open'/detected_at, applies the Stage-0 exclusion gate, and later
    narrates nudge_text. A detector never writes a row itself."""

    kind: str
    chat_id: int
    handle_id: int | None = None
    anchor_message_id: int | None = None
    anchor_date: str | None = None
    evidence: dict = field(default_factory=dict)
    due_by: str | None = None
    source: str = "sql"


# Detector registry: (kind, detect_fn) pairs. N1 registers faded_initiator;
# follow-ups append their own (kind, detect_fn) in a per-detector fence. The
# driver loop in loose_ends is the single place exclusion + urgency run.
DETECTORS: list[tuple[str, Callable[[object, dict], Iterable[CandidateRow]]]] = []


# ---------------------------------------------------------------------------
# Shared dyad primitives (consumed by detectors here and in N2/N3)
# ---------------------------------------------------------------------------

def _dyad_gap_p90(con, chat_id):
    """p90 of inter-message gaps (seconds) between consecutive real messages of
    a dyad, or None when the chat has fewer than two such messages. Single home
    for this primitive: N2/N3 detectors consume it (e.g. to decide a current
    silence is anomalously long for how this pair normally paces)."""
    dates = [
        datetime.datetime.fromisoformat(r[0])
        for r in con.execute(
            f"SELECT m.date FROM messages m "
            f"WHERE m.chat_id = ? AND m.date IS NOT NULL AND {_REAL} "
            f"ORDER BY m.date, m.id",
            (chat_id,),
        ).fetchall()
    ]
    if len(dates) < 2:
        return None
    gaps = [(b - a).total_seconds() for a, b in zip(dates, dates[1:])]
    if len(gaps) < 2:
        return gaps[0]
    return statistics.quantiles(gaps, n=10)[8]


def _counterparty_handle(con, chat_id):
    """The 1:1 chat's counterparty handle_id (the participant that is not me)."""
    row = con.execute(
        "SELECT handle_id FROM chat_participants WHERE chat_id = ? "
        "ORDER BY handle_id LIMIT 1",
        (chat_id,),
    ).fetchone()
    return row[0] if row else None


def _last_message(con, chat_id):
    """(id, date) of the most recent real message in a chat -- the nudge anchor."""
    row = con.execute(
        f"SELECT m.id, m.date FROM messages m "
        f"WHERE m.chat_id = ? AND m.date IS NOT NULL AND {_REAL} "
        f"ORDER BY m.date DESC, m.id DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return (row[0], row[1]) if row else (None, None)
