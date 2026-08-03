"""The LLM-free deterministic half of conflict detection: the raw reads, the
per-signal computations, and the valence corroboration that ranks (never gates)
a detected window.

Split out of conflict_detect.py, which owns the CLI, the per-chat detection
sweep and the labeling pass, and re-exports everything here so
`conflict_detect._window_qualifies` (etc.) keeps resolving for the tests.

The knobs and the SQL fragments are read back through `conflict_detect` rather
than imported by value, so a test that patches CONFLICT_MIN_SIGNALS (or any
sibling) on that module still moves this code. Nothing here calls an LLM.
"""

import bisect
import datetime
import statistics

import stats

import conflict_detect


# ---------------------------------------------------------------------------
# Deterministic signal computation
# ---------------------------------------------------------------------------

# Signals that indicate active tension rather than mere quiet. one_sided_session
# and volume_collapse both just describe a low-activity stretch (and co-fire on
# any lull), so a window needs at least one of these to qualify -- see
# _window_qualifies.
_ACTIVE_TENSION_SIGNALS = frozenset({"latency_withdrawal", "dislike_run"})


def _window_qualifies(fired):
    """True iff a fired-signal set is a real conflict window: it must trip
    >= conflict_detect.CONFLICT_MIN_SIGNALS distinct signals AND include at least one
    active-tension signal. one_sided_session and volume_collapse are not
    independent -- a single quiet, one-directional lull trips both -- so their
    pair is one phenomenon double-counted, not corroboration."""
    return (len(fired) >= conflict_detect.CONFLICT_MIN_SIGNALS
            and bool(fired & _ACTIVE_TENSION_SIGNALS))


def _dt(iso):
    return datetime.datetime.fromisoformat(iso)


def _hours(a, b):
    return (b - a).total_seconds() / 3600.0


def _corpus_end(con):
    row = con.execute("SELECT MAX(date) FROM messages WHERE date IS NOT NULL").fetchone()
    return _dt(row[0]) if row and row[0] else datetime.datetime.utcnow()


# _load_messages / _load_sessions share their names with drift.py's and are NOT
# the same queries. These take ONE chat_id and return that chat's raw rows --
# messages with id and TEXT (the signal scan reads bodies), sessions with the
# full window shape -- because detection walks a single chat at a time. drift's
# take a chat->key map, fold the WHOLE corpus into per-counterparty streams, and
# select strictly fewer columns (no text, no ids). Column tuples verified
# 2026-08-03: they differ, so the two stay separate rather than one side
# fetching message bodies it never reads.

def _load_messages(con, chat_id):
    return con.execute(
        f"SELECT m.id, m.is_from_me, m.date, m.text FROM messages m "
        f"WHERE m.chat_id = ? AND {conflict_detect._REAL_WHERE} ORDER BY m.date, m.id",
        (chat_id,),
    ).fetchall()


def _load_sessions(con, chat_id):
    return con.execute(
        "SELECT id, first_date, last_date, n_messages, n_from_me, "
        "initiator_is_me, ender_is_me FROM sessions "
        "WHERE chat_id = ? ORDER BY first_date, id",
        (chat_id,),
    ).fetchall()


def _load_dislike_dts(con, chat_id):
    return [
        _dt(r[0]) for r in con.execute(
            "SELECT date FROM reactions WHERE chat_id = ? "
            "AND associated_message_type = ? AND date IS NOT NULL ORDER BY date",
            (chat_id, conflict_detect._DISLIKE_TAPBACK),
        ).fetchall()
    ]


def _reply_gaps(messages, corpus_end):
    """RAW reply gaps for the survival estimator. A reply is a sender flip;
    the gap (hours) is attributed to the message being answered (so a long
    inter-session withdrawal lands on the message left hanging). Adds one
    right-CENSORED observation for the final message, whose reply is still
    open at corpus end. Returns (dts, hours, censored, answered_ids), all
    date-sorted and parallel."""
    dts, hrs, cen, mids = [], [], [], []

    def _add(dt, hours, censored, mid):
        dts.append(dt)
        hrs.append(hours)
        cen.append(censored)
        mids.append(mid)

    for k in range(1, len(messages)):
        if messages[k]["is_from_me"] != messages[k - 1]["is_from_me"]:
            a = _dt(messages[k - 1]["date"])
            g = _hours(a, _dt(messages[k]["date"]))
            if g >= 0:
                _add(a, g, False, messages[k - 1]["id"])
    if messages:
        a = _dt(messages[-1]["date"])
        g = _hours(a, corpus_end)
        if g > 0:
            _add(a, g, True, messages[-1]["id"])
    return dts, hrs, cen, mids


def _session_latency(gap, first_dt, last_dt):
    """(fired, frac, baseline_frac, peak_gap_hours, peak_gap_message_id) for one
    session's attributed reply gaps vs its trailing-90-day baseline."""
    dts, hrs, cen, mids = gap
    q, frac_min = conflict_detect.CONFLICT_LATENCY_Q_HOURS, conflict_detect.CONFLICT_LATENCY_FRAC
    lo = bisect.bisect_left(dts, first_dt)
    hi = bisect.bisect_right(dts, last_dt)
    dur, c = hrs[lo:hi], cen[lo:hi]
    frac = stats.frac_over_threshold(dur, c, q) if dur else None
    peak_h = peak_id = None
    for i in range(lo, hi):
        if not cen[i] and (peak_h is None or hrs[i] > peak_h):
            peak_h, peak_id = hrs[i], mids[i]
    b0 = first_dt - datetime.timedelta(days=90)
    blo = bisect.bisect_left(dts, b0)
    bdur, bc = hrs[blo:lo], cen[blo:lo]
    bfrac = stats.frac_over_threshold(bdur, bc, q) if bdur else None
    baseline = bfrac or 0.0
    fired = frac is not None and frac >= frac_min and frac >= 3.0 * baseline
    return fired, frac, bfrac, peak_h, peak_id


def _session_signals(session, gap, dislike_dts):
    """The three per-session deterministic signals, as (fired_set, meta)."""
    first_dt, last_dt = _dt(session["first_date"]), _dt(session["last_date"])
    fired = set()
    l_fired, frac, bfrac, peak_h, peak_id = _session_latency(gap, first_dt, last_dt)
    if l_fired:
        fired.add("latency_withdrawal")

    n = bisect.bisect_right(dislike_dts, last_dt) - bisect.bisect_left(dislike_dts, first_dt)
    if n >= 2:
        fired.add("dislike_run")

    nm, nfm = session["n_messages"], session["n_from_me"]
    if nm >= 2 and (nfm == nm or nfm == 0):
        fired.add("one_sided_session")

    meta = {"latency_frac": frac, "latency_baseline_frac": bfrac,
            "peak_gap_hours": peak_h, "peak_gap_message_id": peak_id,
            "dislike_count": n}
    return fired, meta


def _daily_counts(messages):
    counts = {}
    for m in messages:
        day = m["date"][:10]
        counts[day] = counts.get(day, 0) + 1
    return counts


def _volume_collapse(daily, first_dt, last_dt, window_msgs):
    """(fired, ratio) -- window daily rate vs the trailing-8-week (56d) median
    active-day rate before the window start."""
    base_lo = (first_dt - datetime.timedelta(days=56)).date()
    base_hi = first_dt.date()
    base = [n for day, n in daily.items()
            if base_lo <= datetime.date.fromisoformat(day) < base_hi and n > 0]
    if not base:
        return False, None
    median = statistics.median(base)
    span_days = max(1.0, _hours(first_dt, last_dt) / 24.0)
    rate = window_msgs / span_days
    ratio = rate / median if median else None
    fired = median > 0 and rate <= (1.0 - conflict_detect.CONFLICT_VOLUME_COLLAPSE) * median
    return fired, ratio


def _off_text(messages, first_dt, last_dt):
    """(flag, max_gap_hours) over the window's messages -- a gap this large
    means the exchange likely moved off-channel (forbids a resolution
    verdict downstream)."""
    seg = [m for m in messages if first_dt <= _dt(m["date"]) <= last_dt]
    max_gap = 0.0
    for k in range(1, len(seg)):
        max_gap = max(max_gap, _hours(_dt(seg[k - 1]["date"]), _dt(seg[k]["date"])))
    return max_gap > conflict_detect.CONFLICT_OFF_TEXT_GAP_HOURS, round(max_gap, 1)


# ---------------------------------------------------------------------------
# Valence corroboration (source='llm' rows ONLY)
# ---------------------------------------------------------------------------

def _llm_valence(con, chat_id):
    """(chat_mean, rows) where rows = [(first_date, last_date, valence)] over
    chunk_annotations.source='llm' only. Distilled rows are ignored -- they
    can't corroborate (their emotions are rougher and their events empty)."""
    rows = con.execute(
        "SELECT ch.first_date, ch.last_date, a.valence "
        "FROM chunk_annotations a JOIN chunks ch ON ch.id = a.chunk_id "
        "WHERE ch.chat_id = ? AND a.source = 'llm' AND a.valence IS NOT NULL",
        (chat_id,),
    ).fetchall()
    vals = [r[2] for r in rows]
    return (statistics.fmean(vals) if vals else None), rows


def _corroboration(chat_mean, llm_rows, first_date, last_date):
    """'none' (no llm chunks overlap -> no valence rank boost), 'weak' (llm
    chunks overlap but no meaningful drop) or 'llm' (drop >= threshold), plus a
    boost flag. Overlap is a plain-string ISO compare."""
    overlap = [v for (cf, cl, v) in llm_rows if cf <= last_date and cl >= first_date]
    if not overlap:
        return "none", False, None
    drop = (chat_mean - statistics.fmean(overlap)) if chat_mean is not None else 0.0
    if drop >= conflict_detect.CONFLICT_MIN_VALENCE_DROP:
        return "llm", True, round(drop, 3)
    return "weak", False, round(drop, 3)
