#!/usr/bin/env python3
"""Relationship-drift pipeline: score how each 1:1 relationship is trending
over time and materialize the verdict into the `dyad_drift` table.

Deterministic and LLM-free -- it consumes what earlier pipelines already
computed (messages, the gap-split `sessions` table, and annotate.py's
`chunk_annotations`) plus the shared robust-statistics primitives in
`stats.py`. There is a deliberate twin: this root `drift.py` is the compute
pipeline; a separate `query_tools/drift.py` (a different issue) will hold the
read tools. They share a name but never import each other.

Design (load-bearing)
---------------------
Aggregate by RESOLVED COUNTERPARTY, not by chat or handle. Every 1:1 chat is
attributed to its counterparty's `member_key` string ("contact:<id>" when the
handle resolves to a contact, "handle:<id>" otherwise -- "me"/unknown never get
a row), so a contact's several handles/chats fold into ONE relationship before
anything is scored. Rebuilds are full, so a handle that later gains a contact_id
(key flips handle:N -> contact:M) self-heals.

Seven monthly signals, each gated (< DRIFT_MIN_VALID_MONTHS distinct months =>
absent) and each turned into a 0..1 drift contribution by a Theil-Sen trend with
a bootstrap CI (`stats.robust_slope`): a trend whose CI still spans 0 is flat or
uncertain and contributes 0. The mandatory-plus-available survivors are combined
into a weight-renormalized composite `drift_score`.

  volume       total messages/month           (mandatory)  decline = drift
  reciprocity  min/max sent balance/month      (mandatory)  decline = drift
  latency      P(reply > 1h) via survival      (mandatory)  rise    = drift
  silence      month's longest gap, ECDF-placed              rise    = drift
  valence      mean chunk valence (llm-first)                decline = drift
  depth        mean chunk depth                              decline = drift
  initiation   their share of session starts                 decline = drift

Latency is measured with the survival estimators (`frac_over_threshold`, with
`kaplan_meier_median` kept for the evidence trail) rather than
get_response_dynamics' censored path: reply pairs are derived here from the raw
messages (per chat, in (date, id) order), and the trailing gap after a dyad's
last message is a RIGHT-CENSORED observation. Dropping those long/absent
replies -- exactly what a fading tie produces -- would bias latency downward and
make a dying relationship look responsive; the survival curve keeps them.

Silence never decides on its own. The phase label
(thriving/steady/cooling/fading/dormant) caps at `cooling` unless an
intra-iMessage corroborator -- reciprocity erosion, or latency inflation while
both sides still actively send -- backs the decline (`phase_corroborated`).

Guard: `dyad_drift` is a full rebuild, skip-if-unchanged. The `ingest_state`
high-water mark `drift_max_message_id` plus a hash of DRIFT_WEIGHTS /
DRIFT_PHASE_THRESHOLDS (`drift_params_hash`) decide whether a rerun rebuilds or
skips. DRIFT_ENABLED (default false) short-circuits the whole thing before any
write, so a disabled pipeline never touches the table.

Usage: drift.py [--force] [--dry-run]
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
from collections import defaultdict

import config
import progress
import schema
import stats
from db import CONTACT_LABEL_SQL, connect_default, get_state, run_cli, set_state
from query_tools._shared import _REAL_MESSAGE_CLAUSES, member_key

# Module attribute so tests can patch it; _connect() reads it at call time
# (same seam as annotate.py / topics.py).
DB_PATH = config.TARGET_DB

# The seven signals and the mandatory subset (the always-message-derived trio
# that mandatory_complete requires and that the corroboration rule leans on).
SIGNALS = ("volume", "reciprocity", "latency", "silence",
           "valence", "depth", "initiation")
MANDATORY = ("volume", "reciprocity", "latency")

# Orientation: map each signal's raw monthly value so that HIGHER == MORE DRIFT.
# (min-max normalization inside _signal_trend is affine, so orienting here fixes
# the sign of the slope; robust_slope's significance is scale/shift invariant.)
_ORIENT = {
    "volume": lambda v: -v,       # raw = messages/month; fewer = drift
    "reciprocity": lambda v: -v,  # raw = min/max balance (1 = even); decline = drift
    "latency": lambda v: v,       # raw = P(reply > threshold); rise = drift
    "silence": lambda v: v,       # raw = gap ECDF percentile; rise = drift
    "valence": lambda v: -v,      # raw = mean valence; cooling = drift
    "depth": lambda v: -v,        # raw = mean depth; shallowing = drift
    "initiation": lambda v: -v,   # raw = their initiation share; decline = drift
}

# Bootstrap resamples per trend. Lower than stats.robust_slope's default 2000
# because drift fits seven trends for every dyad; 400 keeps the CI stable
# (deterministic under the fixed seed) while bounding the O(months^2) Theil-Sen
# cost. Series are capped to the last _MAX_SERIES_MONTHS before fitting.
_N_BOOT = 400
_MAX_SERIES_MONTHS = 36
# A reply slower than this counts as a "slow reply" for the survival latency
# signal (P(reply > threshold)). We care about the trend in that probability,
# not the absolute, so a fixed, dyad-agnostic threshold is fine.
_LATENCY_THRESHOLD_SECONDS = 3600.0
# Distilled annotation rows inherit the seed model's bias and carry no events;
# a month with only distilled valence is down-weighted (dropped when enough
# llm-backed months exist) rather than trusted equally.
_RECENT_ACTIVE_DAYS = 60
_RECENT_ACTIVE_MIN = 3   # "both sides still actively send" => each >= this many


def _connect():
    return connect_default(DB_PATH, rows=False)


# ---------------------------------------------------------------------------
# Month helpers
# ---------------------------------------------------------------------------

def _month(dt: datetime.datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


def _month_index(month: str) -> int:
    """'YYYY-MM' -> integer month count, so the trend x-axis reflects real time
    distance (a skipped month leaves a real gap)."""
    year, mon = month.split("-")
    return int(year) * 12 + (int(mon) - 1)


def _cap_series(monthly: dict) -> dict:
    """Keep only the most recent _MAX_SERIES_MONTHS months of a {month: value}
    series (the slope input and the stored evidence both cap here)."""
    months = sorted(monthly)
    if len(months) <= _MAX_SERIES_MONTHS:
        return dict(monthly)
    keep = set(months[-_MAX_SERIES_MONTHS:])
    return {m: v for m, v in monthly.items() if m in keep}


# ---------------------------------------------------------------------------
# Reply-latency survival (derived from raw messages, per chat)
# ---------------------------------------------------------------------------

def reply_latency_observations(chat_messages, observation_end):
    """Reply observations for ONE chat's (is_from_me, datetime) stream, already
    in (date, id) order. Yields (month, duration_seconds, censored):

      * a sender switch is an observed reply -- gap from the immediately prior
        message, tagged by that prior message's month, censored=False;
      * the chat's final message contributes a RIGHT-CENSORED observation
        (observation_end - last), tagged by its month: the reply that has not
        (yet) come. Keeping it is the whole point -- a fading tie's longest
        waits must not be silently dropped.
    """
    obs = []
    for i in range(1, len(chat_messages)):
        prev_me, prev_dt = chat_messages[i - 1]
        cur_me, cur_dt = chat_messages[i]
        if cur_me != prev_me:
            gap = (cur_dt - prev_dt).total_seconds()
            if gap >= 0:
                obs.append((_month(prev_dt), gap, False))
    if chat_messages:
        _, last_dt = chat_messages[-1]
        censored = (observation_end - last_dt).total_seconds()
        if censored > 0:
            obs.append((_month(last_dt), censored, True))
    return obs


def monthly_latency_signal(observations):
    """(frac_series, km_series) from pooled reply observations.

    frac_series[month] = frac_over_threshold(durations, censored, 1h) -- the
    KM survival estimate P(reply > threshold), which RISES as replies inflate
    and, because censored waits stay in the risk set, never drops just because
    the slow replies were the ones that never came. km_series[month] is the
    Kaplan-Meier median (seconds, or None when unreachable) kept only for the
    evidence trail. A month with no estimable events is omitted from frac_series
    (survival estimators return None on all-censored input -- guard, don't
    fabricate)."""
    by_month = defaultdict(lambda: ([], []))
    for month, dur, censored in observations:
        durs, cens = by_month[month]
        durs.append(dur)
        cens.append(censored)
    frac_series, km_series = {}, {}
    for month, (durs, cens) in by_month.items():
        frac = stats.frac_over_threshold(durs, cens, _LATENCY_THRESHOLD_SECONDS)
        if frac is not None:
            frac_series[month] = frac
        km = stats.kaplan_meier_median(durs, cens)
        km_series[month] = None if km is None else round(km, 1)
    return frac_series, km_series


# ---------------------------------------------------------------------------
# Per-signal trend
# ---------------------------------------------------------------------------

def _signal_trend(monthly, orient, *, min_months, alpha, n_boot=_N_BOOT):
    """Turn a {month: raw_value} series into a gated 0..1 drift contribution.

    Orient so higher == more drift, min-max normalize (affine -- keeps the
    slope's sign and significance), then fit a Theil-Sen slope with a bootstrap
    CI. The contribution is the modeled normalized rise across the window
    (slope * span, clamped to [0, 1]) ONLY when the CI excludes 0 in the drift
    direction; a flat-or-uncertain trend contributes 0. Fewer than min_months
    distinct months (or a degenerate fit) => absent."""
    months = sorted(monthly)
    n = len(months)
    base = {"present": False, "significant": False, "drift_val": 0.0,
            "n_months": n, "slope": None, "ci_low": None, "ci_high": None}
    if n < min_months:
        return base
    xs = [_month_index(m) for m in months]
    oriented = [orient(monthly[m]) for m in months]
    lo, hi = min(oriented), max(oriented)
    norm = [(v - lo) / (hi - lo) for v in oriented] if hi > lo else [0.0] * n
    fit = stats.robust_slope(list(zip(xs, norm)), n_boot=n_boot, alpha=alpha)
    if fit is None:
        return base
    span = xs[-1] - xs[0]
    significant = fit["significant"]
    drift_val = (min(1.0, max(0.0, fit["slope"] * span))
                 if significant and fit["slope"] > 0 else 0.0)
    return {
        "present": True,
        "significant": significant,
        "drift_val": round(drift_val, 4),
        "n_months": n,
        "slope": round(fit["slope"], 6),
        "ci_low": round(fit["ci_low"], 6),
        "ci_high": round(fit["ci_high"], 6),
    }


def score_signals(monthly_raw, *, min_months, alpha):
    """Run every signal's trend over its {month: raw_value} series."""
    return {
        sig: _signal_trend(monthly_raw.get(sig, {}), _ORIENT[sig],
                            min_months=min_months, alpha=alpha)
        for sig in SIGNALS
    }


def composite(signals, weights):
    """Weight-renormalized drift_score over the PRESENT signals (a non-present
    signal drops out of both numerator and denominator), plus mandatory_complete
    and the list of present signal names."""
    present = [s for s in SIGNALS if signals[s]["present"]]
    total_w = sum(float(weights.get(s, 0.0)) for s in present)
    if total_w > 0:
        score = sum(float(weights.get(s, 0.0)) * signals[s]["drift_val"]
                    for s in present) / total_w
    else:
        score = 0.0
    mandatory_complete = all(signals[s]["present"] for s in MANDATORY)
    return round(score, 4), mandatory_complete, present


def _drifting(signal):
    """A signal with a significant trend in the DRIFT direction (oriented slope
    > 0)."""
    return bool(signal["present"] and signal["significant"]
                and (signal["slope"] or 0.0) > 0)


def corroborated(signals, both_active):
    """Whether a decline label is backed by an intra-iMessage corroborator:
    reciprocity erosion, OR latency inflation while both sides still actively
    send. Silence and the annotation signals never corroborate on their own."""
    return _drifting(signals["reciprocity"]) or (
        _drifting(signals["latency"]) and both_active)


def phase(drift_score, is_corroborated, last_message_days, thresholds):
    """Map drift_score + recency to a phase, then apply the corroboration floor:
    a decline label (fading/dormant) that lacks an intra-iMessage corroborator
    is capped at `cooling`, so silence alone can never produce a decline."""
    thriving = float(thresholds.get("thriving", 0.10))
    steady = float(thresholds.get("steady", 0.35))
    cooling = float(thresholds.get("cooling", 0.60))
    dormant_days = float(thresholds.get("dormant_days", 90))

    if last_message_days is not None and last_message_days >= dormant_days:
        candidate = "dormant"
    elif drift_score >= cooling:
        candidate = "fading"
    elif drift_score >= steady:
        candidate = "cooling"
    elif drift_score >= thriving:
        candidate = "steady"
    else:
        candidate = "thriving"

    if candidate in ("fading", "dormant") and not is_corroborated:
        candidate = "cooling"
    return candidate


def confidence_tier(mandatory_complete, n_present, n_months, valence_llm_fraction):
    """high/medium/low, computed in Python from coverage. An all-distilled
    valence corpus degrades the tier -- it never refuses."""
    if not mandatory_complete:
        return "low"
    valence_ok = valence_llm_fraction is None or valence_llm_fraction >= 0.5
    if n_present >= 5 and n_months >= 8 and valence_ok:
        return "high"
    if n_present >= 3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# DB aggregation
# ---------------------------------------------------------------------------

def _real_clause():
    return " AND ".join(_REAL_MESSAGE_CLAUSES)


def _chat_to_key(con):
    """{chat_id: (contact_key, label)} for every 1:1 chat with exactly one
    counterparty handle. Group chats (no single counterparty) and any oddball
    direct chat carrying multiple participant handles are skipped -- drift is
    about one relationship at a time."""
    rows = con.execute(f"""
        SELECT cp.chat_id, h.id AS handle_id, h.contact_id,
               {CONTACT_LABEL_SQL} AS label
        FROM chat_participants cp
        JOIN chats ch ON ch.id = cp.chat_id AND ch.is_group = 0
        JOIN handles h ON h.id = cp.handle_id
        LEFT JOIN contacts c ON c.id = h.contact_id
    """).fetchall()
    by_chat = defaultdict(list)
    for chat_id, handle_id, contact_id, label in rows:
        by_chat[chat_id].append((handle_id, contact_id, label))
    out = {}
    for chat_id, parts in by_chat.items():
        if len(parts) != 1:
            continue
        handle_id, contact_id, label = parts[0]
        key = member_key(False, contact_id, handle_id)
        out[chat_id] = (key, label)
    return out


def _observation_end(con):
    """Corpus-wide max message date (the right-censoring / dormancy reference).
    Plain-string MAX -- messages.date is unlabeled-UTC ISO, ordered lexically."""
    row = con.execute(
        f"SELECT MAX(m.date) FROM messages m "
        f"WHERE m.date IS NOT NULL AND {_real_clause()}"
    ).fetchone()
    return None if not row or row[0] is None else datetime.datetime.fromisoformat(row[0])


def _load_messages(con, chat_to_key):
    """{contact_key: {chat_id: [(is_from_me, datetime), ...]}}, each chat's
    stream in (date, id) order. Only 1:1 chats that resolved to a key."""
    per_key = defaultdict(lambda: defaultdict(list))
    for chat_id, is_from_me, date in con.execute(
        f"SELECT m.chat_id, m.is_from_me, m.date FROM messages m "
        f"JOIN chats ch ON ch.id = m.chat_id AND ch.is_group = 0 "
        f"WHERE m.date IS NOT NULL AND {_real_clause()} "
        f"ORDER BY m.chat_id, m.date, m.id"
    ):
        keyed = chat_to_key.get(chat_id)
        if keyed is None:
            continue
        per_key[keyed[0]][chat_id].append(
            (bool(is_from_me), datetime.datetime.fromisoformat(date)))
    return per_key


def _load_sessions(con, chat_to_key):
    """{contact_key: [(month, initiator_is_me), ...]} from the sessions table,
    for the resolved 1:1 chats."""
    per_key = defaultdict(list)
    for chat_id, first_date, initiator_is_me in con.execute(
        "SELECT s.chat_id, s.first_date, s.initiator_is_me FROM sessions s "
        "JOIN chats ch ON ch.id = s.chat_id AND ch.is_group = 0 "
        "WHERE s.first_date IS NOT NULL"
    ):
        keyed = chat_to_key.get(chat_id)
        if keyed is None:
            continue
        per_key[keyed[0]].append((first_date[:7], bool(initiator_is_me)))
    return per_key


def _load_annotations(con, chat_to_key):
    """{contact_key: [(month, valence, depth, source), ...]} joining chunks to
    their chunk_annotations for the resolved 1:1 chats."""
    per_key = defaultdict(list)
    for chat_id, first_date, valence, depth, source in con.execute(
        "SELECT ck.chat_id, ck.first_date, a.valence, a.depth, a.source "
        "FROM chunks ck JOIN chunk_annotations a ON a.chunk_id = ck.id "
        "JOIN chats ch ON ch.id = ck.chat_id AND ch.is_group = 0 "
        "WHERE ck.first_date IS NOT NULL"
    ):
        keyed = chat_to_key.get(chat_id)
        if keyed is None:
            continue
        per_key[keyed[0]].append((first_date[:7], valence, depth, source))
    return per_key


# ---------------------------------------------------------------------------
# Per-dyad monthly series
# ---------------------------------------------------------------------------

def _behavioral_monthly(chats, observation_end):
    """Build the message-derived monthly series for one dyad from {chat_id:
    [(is_from_me, dt)]}: volume, reciprocity (balance), the pooled reply-latency
    observations, this dyad's per-month longest gap, the total message count,
    the last message datetime, and the trailing-60-day per-side send counts."""
    volume = defaultdict(int)
    sent = defaultdict(lambda: [0, 0])          # month -> [me, them]
    max_gap = defaultdict(float)                # month -> longest inter-msg gap
    latency_obs = []
    all_gaps = []                               # feeds the corpus silence sample
    n_messages = 0
    last_dt = None
    recent_cut = (observation_end - datetime.timedelta(days=_RECENT_ACTIVE_DAYS)
                  if observation_end is not None else None)
    recent = [0, 0]                             # [me, them] in the trailing window

    for msgs in chats.values():
        latency_obs.extend(reply_latency_observations(msgs, observation_end))
        prev_dt = None
        for is_from_me, dt in msgs:
            n_messages += 1
            mon = _month(dt)
            volume[mon] += 1
            sent[mon][0 if is_from_me else 1] += 1
            if recent_cut is not None and dt >= recent_cut:
                recent[0 if is_from_me else 1] += 1
            if last_dt is None or dt > last_dt:
                last_dt = dt
            if prev_dt is not None:
                gap = (dt - prev_dt).total_seconds()
                if gap > 0:
                    all_gaps.append(gap)
                    if gap > max_gap[mon]:
                        max_gap[mon] = gap
            prev_dt = dt

    reciprocity = {}
    for mon, (me, them) in sent.items():
        hi = max(me, them)
        reciprocity[mon] = (min(me, them) / hi) if hi else 0.0
    both_active = recent[0] >= _RECENT_ACTIVE_MIN and recent[1] >= _RECENT_ACTIVE_MIN
    return {
        "volume": dict(volume),
        "reciprocity": reciprocity,
        "max_gap": dict(max_gap),
        "latency_obs": latency_obs,
        "all_gaps": all_gaps,
        "n_messages": n_messages,
        "last_dt": last_dt,
        "both_active": both_active,
    }


def _silence_monthly(max_gap, winsor_logs):
    """Place each month's longest gap on the corpus-wide winsorized log-gap
    ECDF (stats primitives): higher percentile == a more extreme silence."""
    if not winsor_logs:
        return {}
    out = {}
    for mon, gap in max_gap.items():
        if gap > 0:
            pct = stats.ecdf_percentile(math.log(gap), winsor_logs)
            if pct is not None:
                out[mon] = round(pct, 4)
    return out


def _initiation_monthly(sessions, min_sessions):
    """Their share of session initiations per month, but only when the dyad has
    >= min_sessions sessions overall (too few makes the share noise)."""
    if len(sessions) < min_sessions:
        return {}
    total = defaultdict(int)
    theirs = defaultdict(int)
    for mon, initiator_is_me in sessions:
        total[mon] += 1
        if not initiator_is_me:
            theirs[mon] += 1
    return {mon: theirs[mon] / total[mon] for mon in total}


def _valence_depth_monthly(annotations, min_months):
    """(valence_series, depth_series, valence_llm_fraction).

    Valence prefers source='llm' rows: a month with any llm row uses only its
    llm valences; a month with only distilled rows is down-weighted -- dropped
    when >= min_months llm-backed months already exist, kept (degraded) when
    that is all the corpus has. Depth distills cleanly, so it averages every
    row. valence_llm_fraction feeds the confidence tier."""
    llm_by_month = defaultdict(list)
    dist_by_month = defaultdict(list)
    depth_by_month = defaultdict(list)
    for mon, valence, depth, source in annotations:
        if valence is not None:
            (llm_by_month if source == "llm" else dist_by_month)[mon].append(valence)
        if depth is not None:
            depth_by_month[mon].append(depth)

    llm_months = {m: sum(v) / len(v) for m, v in llm_by_month.items()}
    dist_only = {m: sum(v) / len(v) for m, v in dist_by_month.items()
                 if m not in llm_months}
    if len(llm_months) >= min_months:
        valence = dict(llm_months)                 # enough llm months: drop distilled
    else:
        valence = {**dist_only, **llm_months}      # degrade: use what we have

    total_valence_months = len(set(llm_months) | set(dist_only))
    llm_fraction = (len(llm_months) / total_valence_months
                    if total_valence_months else None)
    depth = {m: sum(v) / len(v) for m, v in depth_by_month.items()}
    return valence, depth, llm_fraction


# ---------------------------------------------------------------------------
# Scoring one dyad -> a dyad_drift row dict
# ---------------------------------------------------------------------------

def score_dyad(key, label, chats, sessions, annotations, *,
               observation_end, winsor_logs, cfg, behav=None):
    """Score one resolved counterparty into a dyad_drift row dict, or None when
    the dyad is below the message/month inclusion gates. `cfg` bundles the
    tunables so this stays a pure function testable without a DB. `behav` may be
    passed precomputed (build_drift already computes it to pool the corpus gap
    sample); otherwise it is derived from `chats`."""
    if behav is None:
        behav = _behavioral_monthly(chats, observation_end)
    if behav["n_messages"] < cfg["min_messages"]:
        return None
    if len(behav["volume"]) < cfg["min_months"]:
        return None

    frac_series, km_series = monthly_latency_signal(behav["latency_obs"])
    valence, depth, valence_llm_fraction = _valence_depth_monthly(
        annotations, cfg["min_valid_months"])

    monthly_raw = {
        "volume": _cap_series(behav["volume"]),
        "reciprocity": _cap_series(behav["reciprocity"]),
        "latency": _cap_series(frac_series),
        "silence": _cap_series(_silence_monthly(behav["max_gap"], winsor_logs)),
        "valence": _cap_series(valence),
        "depth": _cap_series(depth),
        "initiation": _cap_series(_initiation_monthly(sessions, cfg["min_sessions"])),
    }

    signals = score_signals(monthly_raw, min_months=cfg["min_valid_months"],
                            alpha=cfg["alpha"])
    drift_score, mandatory_complete, present = composite(signals, cfg["weights"])
    is_corroborated = corroborated(signals, behav["both_active"])

    last_dt = behav["last_dt"]
    last_message_days = None
    if last_dt is not None and observation_end is not None:
        last_message_days = (observation_end.date() - last_dt.date()).days
    ph = phase(drift_score, is_corroborated, last_message_days, cfg["thresholds"])
    tier = confidence_tier(mandatory_complete, len(present),
                           len(behav["volume"]), valence_llm_fraction)

    months_active = sorted(behav["volume"])
    n_llm = sum(1 for _m, _v, _d, s in annotations if s == "llm")
    n_dist = sum(1 for _m, _v, _d, s in annotations if s != "llm")
    coverage = {
        "n_messages": behav["n_messages"],
        "n_months": len(months_active),
        "first_month": months_active[0],
        "last_month": months_active[-1],
        "last_message_days": last_message_days,
        "both_active_recent": behav["both_active"],
        "signals_present": present,
        "signals_absent": [s for s in SIGNALS if not signals[s]["present"]],
        "annotation": {
            "n_llm": n_llm,
            "n_distilled": n_dist,
            "valence_llm_month_fraction": (round(valence_llm_fraction, 4)
                                           if valence_llm_fraction is not None else None),
        },
    }
    per_signal = {s: dict(signals[s]) for s in SIGNALS}
    per_signal["latency"]["km_median_seconds_by_month"] = _cap_series(km_series)
    monthly_series = {s: [[m, monthly_raw[s][m]] for m in sorted(monthly_raw[s])]
                      for s in SIGNALS if monthly_raw[s]}

    row = {
        "contact_key": key,
        "label": label,
        "n_messages": behav["n_messages"],
        "n_months": len(months_active),
        "first_month": months_active[0],
        "last_month": months_active[-1],
        "drift_score": drift_score,
        "phase": ph,
        "phase_corroborated": int(is_corroborated),
        "mandatory_complete": int(mandatory_complete),
        "confidence_tier": tier,
        "per_signal_gate_json": json.dumps(per_signal),
        "coverage_json": json.dumps(coverage),
        "monthly_series_json": json.dumps(monthly_series),
    }
    for s in SIGNALS:
        row[f"{s}_value"] = signals[s]["drift_val"]
        row[f"{s}_gate"] = int(signals[s]["present"] and signals[s]["significant"])
    return row


_INSERT_COLUMNS = (
    "contact_key", "label", "n_messages", "n_months", "first_month", "last_month",
    "drift_score", "phase", "phase_corroborated", "mandatory_complete",
    "confidence_tier",
    "volume_value", "volume_gate", "reciprocity_value", "reciprocity_gate",
    "latency_value", "latency_gate", "silence_value", "silence_gate",
    "valence_value", "valence_gate", "depth_value", "depth_gate",
    "initiation_value", "initiation_gate",
    "per_signal_gate_json", "coverage_json", "monthly_series_json",
)


# ---------------------------------------------------------------------------
# Build (full rebuild, skip-if-unchanged)
# ---------------------------------------------------------------------------

def _params_hash():
    """Stable hash of the two structured params. A change forces a rebuild even
    when no new messages arrived, so re-weighting takes effect immediately."""
    payload = json.dumps(
        {"weights": config.DRIFT_WEIGHTS, "thresholds": config.DRIFT_PHASE_THRESHOLDS},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _max_message_id(con):
    row = con.execute(
        f"SELECT MAX(m.id) FROM messages m "
        f"WHERE m.date IS NOT NULL AND {_real_clause()}"
    ).fetchone()
    return int(row[0] or 0)


def _current_config():
    return {
        "min_messages": config.DRIFT_MIN_MESSAGES,
        "min_months": config.DRIFT_MIN_MONTHS,
        "min_valid_months": config.DRIFT_MIN_VALID_MONTHS,
        "min_sessions": config.DRIFT_MIN_SESSIONS,
        "alpha": config.DRIFT_SLOPE_CI,
        "weights": config.DRIFT_WEIGHTS,
        "thresholds": config.DRIFT_PHASE_THRESHOLDS,
    }


def build_drift(con, *, force=False, dry_run=False):
    """Full rebuild of dyad_drift, skipping when neither the corpus high-water
    mark nor the params hash changed. Returns the number of rows written (0 on a
    skip or dry run). Idempotent: same inputs -> byte-identical rows."""
    schema.create_schema(con)  # self-heal the table if drift.py runs pre-ingest

    cur_max = _max_message_id(con)
    cur_hash = _params_hash()
    prev_max = get_state(con, "drift_max_message_id")
    prev_hash = get_state(con, "drift_params_hash")
    unchanged = (prev_max is not None and int(prev_max) == cur_max
                 and prev_hash == cur_hash)
    if unchanged and not force and not dry_run:
        print("dyad_drift up to date (skipped)")
        return 0

    cfg = _current_config()
    observation_end = _observation_end(con)
    chat_to_key = _chat_to_key(con)
    messages = _load_messages(con, chat_to_key)
    sessions = _load_sessions(con, chat_to_key)
    annotations = _load_annotations(con, chat_to_key)

    key_label = {}
    for _chat_id, (key, label) in chat_to_key.items():
        key_label.setdefault(key, label)

    # Behavioral series once per dyad, pooling every gap into the corpus-wide
    # winsorized log-gap sample the silence signal places against (winsorized
    # once here, not per dyad-month).
    behav_by_key = {}
    all_gaps = []
    for key, chats in messages.items():
        behav = _behavioral_monthly(chats, observation_end)
        behav_by_key[key] = behav
        all_gaps.extend(behav["all_gaps"])
    positives = [g for g in all_gaps if g > 0]
    winsor_logs = stats.winsorize([math.log(g) for g in positives]) if positives else []

    rows = []
    with progress.track("Scoring drift", total=len(messages)) as p:
        for key, chats in messages.items():
            row = score_dyad(
                key, key_label.get(key, key), chats,
                sessions.get(key, []), annotations.get(key, []),
                observation_end=observation_end, winsor_logs=winsor_logs,
                cfg=cfg, behav=behav_by_key[key])
            if row is not None:
                rows.append(row)
            p.advance()

    if dry_run:
        print(f"--dry-run: {len(rows)} dyads would be scored "
              f"(of {len(messages)} resolved 1:1 relationships); nothing written")
        return 0

    placeholders = ",".join("?" * len(_INSERT_COLUMNS))
    con.execute("DELETE FROM dyad_drift")
    con.executemany(
        f"INSERT INTO dyad_drift({','.join(_INSERT_COLUMNS)}) VALUES ({placeholders})",
        [tuple(row[c] for c in _INSERT_COLUMNS) for row in rows],
    )
    set_state(con, "drift_max_message_id", cur_max)
    set_state(con, "drift_params_hash", cur_hash)
    con.commit()
    print(f"Built dyad_drift: {len(rows)} dyads scored "
          f"(of {len(messages)} resolved 1:1 relationships)")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Score relationship drift into the dyad_drift table "
                    "(deterministic; no Ollama). Full rebuild, skip-if-unchanged.")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even when the corpus and params are unchanged")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many dyads would be scored; write nothing")
    args = parser.parse_args(argv)

    # DRIFT_ENABLED gates before any DB access, so a disabled pipeline leaves
    # dyad_drift (and a durable D5 purge) completely untouched.
    if not config.DRIFT_ENABLED:
        print("DRIFT_ENABLED is false -- drift scoring is disabled; dyad_drift "
              "left untouched. Set DRIFT_ENABLED=true to build it.")
        return 0

    with contextlib.closing(_connect()) as con:
        build_drift(con, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    run_cli(main)
