"""Conflict read tools (RES-46 / C1): find_conflicts (ranked conflict episodes
in scope) and get_conflict (one episode's details + its labeled signal turns).

These read the deterministic finder's output (conflicts / conflict_signals,
written by conflict_detect.py) -- no Ollama, no re-detection. Conflicts exist
only for 1:1 chats by construction; a person/chat scope resolves the same way
as the other analytics tools and is then restricted to direct chats
(_restrict_to_direct_chats -- F3). Escalation labels are eval-gated BANDS
(tentative/likely), never certainties -- surface them as such.

conflicts.first_date/last_date are unlabeled-UTC ISO strings: plain-string
compares only, never SQL date functions in WHERE (see CLAUDE.md gotchas)."""

import contextlib
import json

import config
from function_words import tokenize

from ._shared import (
    _connect,
    _date_clauses,
    _envelope,
    _resolve_chat_scope,
    _restrict_to_direct_chats,
)
# LSM primitives are imported, never reimplemented (F1/analytics_language is the
# single home): _weighted_rates -> word-count-weighted category rates over
# lexicon rows, _lsm_score -> the me-vs-them style-matching score.
from .analytics_language import _lsm_score, _weighted_rates

# Result caps (contract-fixed; this tool's own, like analytics_social's
# eligibility gates). Kept well under TOOL_RESULT_MAX_CHARS: <=15 signal rows,
# each excerpt <=200 chars, is a few KB.
MAX_CONFLICT_ROWS = 25       # find_conflicts page size
MAX_SIGNAL_ROWS = 15         # get_conflict signal rows per page
EXCERPT_CAP = 200            # defensive re-cap on stored excerpts
# find_conflicts rank tie-break: tone-corroborated episodes surface first.
_CORROBORATION_RANK = {"llm": 2, "weak": 1, "none": 0}

# Fixed, code-emitted humility disclaimer for the conflict READ surfaces
# (find_conflicts / get_conflict / diagnose_conflict). Delivered out-of-band via
# the generic `_epilogue` channel (chat.run_turn strips it from the model's view
# and prints it verbatim after the reply -- the wellbeing support_block
# precedent), so it always reaches the user regardless of model compliance.
# Never LLM-authored, never persisted. It replaces a hard who-started boundary:
# the tool surfaces signals, not a fault verdict, and says so.
_ANALYSIS_DISCLAIMER = (
    "Heads up: this is an automated read of text-only signals (reply timing, "
    "message volume, tapbacks). It has limited information -- it can't see calls, "
    "in-person context, or tone -- so it can be wrong and does not determine who "
    "was at fault or how things actually resolved. Treat it as a prompt to "
    "reflect, not a verdict."
)


def _has_conflict_tables(con):
    """True only when both conflict tables exist -- a DB built before this
    feature has neither, and the read tools must return empty, not raise."""
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('conflicts', 'conflict_signals')")}
    return {"conflicts", "conflict_signals"} <= names


def _signals_json(peak_signal_json):
    try:
        data = json.loads(peak_signal_json) if peak_signal_json else {}
    except (TypeError, ValueError):
        data = {}
    return data


def _conflict_summary(row):
    """Compact conflict row for find_conflicts (no per-signal excerpts).

    `labeled_signal_turns` (non-abstain conflict_signals rows) is folded into the
    find_conflicts SELECT as a GROUP BY aggregate, so this is a pure row->dict
    shaper with no per-row query (was an N+1 COUNT)."""
    peak = _signals_json(row["peak_signal_json"])
    n_signal_rows = row["labeled_signal_turns"]
    # A make-up window is one the LLM topic pass labeled 'repair'. Kept explicit
    # so the ranking can de-prioritize reconciliations (a make-up is not the
    # "tensest" stretch) and the model can narrate the arc faithfully. NOTE:
    # peak's `valence_boost` is NOT a make-up signal -- it flags a corroborated
    # valence DROP (tension), so it is surfaced for context but never drives
    # `reconciliation`.
    valence_boost = bool(peak.get("valence_boost"))
    reconciliation = row["topic"] == "repair"
    return {
        "conflict_id": row["id"],
        "chat_id": row["chat_id"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "topic": row["topic"],
        "signals": peak.get("signals", []),
        "signal_count": peak.get("signal_count", 0),
        "corroboration": row["corroboration"],
        "off_text_flag": bool(row["off_text_flag"]),
        "valence_boost": valence_boost,
        "reconciliation": reconciliation,
        "labeled_signal_turns": n_signal_rows,
        "peak_gap_hours": peak.get("peak_gap_hours"),
    }


def find_conflicts(chat_id=None, handle_id=None, contact_id=None,
                   date_from=None, date_to=None, limit=10):
    """Detected conflict episodes in scope, ranked so the best-substantiated
    surface first: tone-corroborated episodes (corroboration llm > weak > none),
    then deterministic signal strength, then recency. Each is a tense stretch
    found from deterministic
    signals WITH RECEIPTS (reply-latency withdrawal, one-sided sessions,
    dislike-tapback runs, volume collapse); `signals` names which fired and
    `signal_count` how many. 1:1 chats only. handle_id/contact_id scope to that
    person's direct chats; date bounds filter on the episode start. corroboration
    is 'none' unless source='llm' annotations backed the tone drop. Returns an
    envelope {rows, total_matching, n_corroborated, returned, order, truncated};
    total_matching counts EVERY detected stretch in scope (mostly uncorroborated
    lulls -- not a count of real fights), while n_corroborated is the
    substantiated subset (weak+llm) -- prefer it as a headline count. Expand one
    with get_conflict(conflict_id)."""
    limit = max(1, min(int(limit), MAX_CONFLICT_ROWS))
    order = ("corroboration-desc, non-reconciliation-first, signal_count-desc, "
             "most-recent-first")
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return _envelope([], 0, order)
        chat_ids, empty = _resolve_chat_scope(con, chat_id, handle_id, contact_id)
        if empty:
            return _envelope([], 0, order)
        if chat_ids is not None:
            chat_ids = _restrict_to_direct_chats(con, chat_ids)
            if not chat_ids:
                return _envelope([], 0, order)

        clauses, params = [], []
        if chat_ids is not None:
            clauses.append(f"c.chat_id IN ({','.join('?' * len(chat_ids))})")
            params.extend(chat_ids)
        _date_clauses(clauses, params, date_from, date_to, column="c.first_date")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        # Fold the per-conflict labeled-signal count into one query (was a per-row
        # COUNT -> N+1). LEFT JOIN keeps conflicts with zero signals; the CASE sums
        # only non-abstain rows (there is no index on label_class). No SQL LIMIT --
        # total_matching/n_corroborated below must count the full in-scope
        # population; the Python sort + [:limit] slice do the ranking.
        rows = con.execute(
            "SELECT c.id, c.chat_id, c.first_date, c.last_date, c.topic, "
            "c.peak_signal_json, c.corroboration, c.off_text_flag, "
            "COALESCE(SUM(CASE WHEN s.label_class != 'abstain' THEN 1 ELSE 0 END), 0) "
            "AS labeled_signal_turns "
            "FROM conflicts c "
            "LEFT JOIN conflict_signals s ON s.conflict_id = c.id"
            f"{where} GROUP BY c.id", params,
        ).fetchall()
        summaries = [_conflict_summary(r) for r in rows]

    # Best-substantiated first: tone-corroborated episodes (llm > weak > none)
    # outrank raw signal count, so the ~5% with a corroborated valence drop
    # surface ahead of the majority of uncorroborated lulls. WITHIN a
    # corroboration tier, genuine tension outranks a reconciliation/make-up
    # window (`not reconciliation` sorts first under reverse=True), so an apology
    # exchange is never surfaced as the "tensest" stretch. Then most distinct
    # deterministic signals, then most recent. (Detection + total_matching are
    # unchanged -- this only reorders which episodes fill the top `limit`.)
    summaries.sort(
        key=lambda s: (_CORROBORATION_RANK.get(s["corroboration"], 0),
                       not s["reconciliation"],
                       s["signal_count"], s["last_date"] or ""),
        reverse=True)
    total = len(summaries)
    n_corroborated = sum(
        1 for s in summaries if _CORROBORATION_RANK.get(s["corroboration"], 0) >= 1)
    env = _envelope(summaries[:limit], total, order)
    env["n_corroborated"] = n_corroborated
    if total:
        env["_epilogue"] = _ANALYSIS_DISCLAIMER
    return env


def _signal_row(row):
    try:
        top2 = json.loads(row["top2_labels_json"]) if row["top2_labels_json"] else []
    except (TypeError, ValueError):
        top2 = []
    excerpt = row["excerpt"] or ""
    return {
        "message_id": row["message_id"],
        "actor": "me" if row["actor_is_me"] else "them",
        "label_class": row["label_class"],
        "top2": top2,
        "band": row["band"],
        "excerpt": excerpt[:EXCERPT_CAP],
    }


def get_conflict(conflict_id, offset=0, limit=15):
    """One conflict episode: its deterministic evidence (peak_signal), tone
    corroboration, off_text_flag, and up to 15 labeled signal turns (paginated;
    excerpts capped at 200 chars). Labels are eval-gated BANDS (band
    'tentative'/'likely'), never raw scores -- present them as such, and never
    read a resolution/apology into an episode whose off_text_flag is set. Pass
    offset to page further signals."""
    conflict_id = int(conflict_id)
    limit = max(1, min(int(limit), MAX_SIGNAL_ROWS))
    offset = max(0, int(offset))
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT id, chat_id, first_date, last_date, start_session_id, "
            "end_session_id, topic, peak_signal_json, corroboration, "
            "off_text_flag, source FROM conflicts WHERE id = ?", (conflict_id,),
        ).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}
        total = con.execute(
            "SELECT COUNT(*) FROM conflict_signals WHERE conflict_id = ?",
            (conflict_id,)).fetchone()[0]
        sig_rows = con.execute(
            "SELECT message_id, actor_is_me, label_class, top2_labels_json, "
            "band, excerpt FROM conflict_signals WHERE conflict_id = ? "
            "ORDER BY id LIMIT ? OFFSET ?", (conflict_id, limit, offset),
        ).fetchall()

    signals = [_signal_row(r) for r in sig_rows]
    return {
        "conflict_id": row["id"],
        "chat_id": row["chat_id"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "start_session_id": row["start_session_id"],
        "end_session_id": row["end_session_id"],
        "topic": row["topic"],
        "corroboration": row["corroboration"],
        "off_text_flag": bool(row["off_text_flag"]),
        "peak_signal": _signals_json(row["peak_signal_json"]),
        "signals": signals,
        "signals_total": total,
        "returned": len(signals),
        "offset": offset,
        "limit": limit,
        "truncated": total > offset + len(signals),
        "note": "labels are eval-gated bands (tentative/likely), not certainties",
        # Code-printed humility disclaimer (stripped from the model's view by
        # chat.run_turn, appended verbatim after the reply).
        "_epilogue": _ANALYSIS_DISCLAIMER,
    }


# ===========================================================================
# C2 (RES-51): diagnose_conflict scorecard + softened DV-suppression floor.
#
# The scorecard composes DETERMINISTIC pieces ONLY -- lexicon I-vs-you /
# we-talk ratios, LSM, session initiator/ender roles, reply-latency + volume
# collapse read off C1's stored peak_signal survival stats -- and carries NO
# aggressor / who-won verdict. The one interpretive-ish section (which side
# withdrew vs pursued, a NON-BLAME demand-withdraw read) is the "role
# assignment" the DV floor withholds.
# ===========================================================================

# --- DV power-and-control screen (Master Spec A10, SOFTENED / code-enforced) ---
# Vocabulary seeded from the PUBLIC Duluth Model "Power and Control Wheel"
# (theduluthmodel.org) -- Coercion & Threats, Intimidation, Isolation, and the
# fear such tactics instill -- plus common texting variants. This is GENERIC
# English screening vocabulary, NOT anyone's personal data. Whole-token matches
# run over function_words.tokenize() output (lowercased); the multi-word
# phrases run as substring checks over the apostrophe-normalized lowercased
# text (tokenize() would split them apart).
_DV_THREAT_TERMS = frozenset("""
    kill kills killing killed strangle strangled strangling choke choked
    choking stab stabbed stabbing murder murdered gun guns knife weapon
    weapons punch punched kys
""".split())
_DV_ISOLATION_TERMS = frozenset("""
    isolate isolated isolating forbid forbidden
""".split())
_DV_FEAR_TERMS = frozenset("""
    afraid scared terrified terrify terrifies terrifying frightened petrified
""".split())

# Multi-word phrases (substring, apostrophe-normalized lowercased text), grouped
# by the same three categories so a phrase-only hit is still attributable.
_DV_THREAT_PHRASES = frozenset({
    "kill you", "hurt you", "make you pay", "you'll pay", "you'll regret",
    "you will regret", "watch your back", "or else", "i know where you live",
    "hunt you down", "beat you", "shut your mouth", "do as i say",
    "you made me do", "look what you made me", "if you ever leave",
    "you can't leave", "you cant leave",
})
_DV_ISOLATION_PHRASES = frozenset({
    "you can't see", "you cant see", "stop talking to", "not allowed to",
    "you don't need anyone", "you dont need anyone", "no one else will",
    "nobody else will", "who are you texting", "who were you with",
    "you belong to me",
})
_DV_FEAR_PHRASES = frozenset({
    "scared of you", "afraid of you", "you scare me", "walking on eggshells",
    "please don't hurt me", "please dont hurt me", "don't hurt me",
    "dont hurt me",
})

# Fixed, code-emitted resource footer. Constant on purpose (never LLM-authored,
# never persisted): a hit only ever appends THIS string.
_DV_RESOURCE_TEXT = (
    "If any part of this involves fear for your safety, you are not alone and "
    "it is not your fault. National Domestic Violence Hotline: 1-800-799-7233 "
    "(thehotline.org) -- free, confidential, 24/7."
)

# Transient per-run DV-screen bypass, set by chat.py's --no-dv-screen for one
# process. False = defer to config.CONFLICT_DV_SCREEN; True = force the screen
# off for this run only. NEVER persisted.
_DV_SCREEN_BYPASS = False


def set_dv_screen_bypass(bypass):
    """Set the transient per-run DV-screen bypass (chat.py --no-dv-screen). A
    True value overrides config.CONFLICT_DV_SCREEN to 'off' for THIS process
    only -- role analysis is restored, but a marker hit still appends the
    resource footer. Nothing is persisted."""
    global _DV_SCREEN_BYPASS
    _DV_SCREEN_BYPASS = bool(bypass)


def _dv_screen_enabled():
    """True when the DV role-suppression screen is active: config default
    (config.CONFLICT_DV_SCREEN, read at call time so tests/CLI can override),
    unless the transient per-run bypass is set."""
    if _DV_SCREEN_BYPASS:
        return False
    return bool(config.CONFLICT_DV_SCREEN)


def _dv_scan_text(text):
    """The DV power-and-control categories present in one message's FULL text:
    whole-token matches over tokenize() plus substring phrase matches. Returns
    a set, a subset of {'threat', 'isolation', 'fear'} (empty when clean)."""
    if not text:
        return set()
    toks = set(tokenize(text))
    lowered = text.lower().replace("’", "'")
    fired = set()
    for cat, terms, phrases in (
        ("threat", _DV_THREAT_TERMS, _DV_THREAT_PHRASES),
        ("isolation", _DV_ISOLATION_TERMS, _DV_ISOLATION_PHRASES),
        ("fear", _DV_FEAR_TERMS, _DV_FEAR_PHRASES),
    ):
        if (toks & terms) or any(p in lowered for p in phrases):
            fired.add(cat)
    return fired


def _window_message_texts(con, conflict_id, chat_id, first_date, last_date):
    """FULL (untruncated) text of every message in the conflict window -- the
    chat's messages in [first_date, last_date] UNION the conflict_signals
    message ids (A10: the full window, not the stored 200-char excerpts).
    Plain-string date compares only (never SQL date functions -- see the
    CLAUDE.md gotcha)."""
    seen = set()
    texts = []
    for mid, text in con.execute(
        "SELECT id, text FROM messages WHERE chat_id = ? AND date IS NOT NULL "
        "AND date >= ? AND date <= ? AND text IS NOT NULL",
        (chat_id, first_date, last_date),
    ):
        seen.add(mid)
        texts.append(text)
    # Signal turns whose id somehow falls outside the plain date window
    # (defensive -- they are the labeled receipts and must be scanned).
    extra = [r[0] for r in con.execute(
        "SELECT message_id FROM conflict_signals WHERE conflict_id = ? "
        "AND message_id IS NOT NULL", (conflict_id,)) if r[0] not in seen]
    if extra:
        ph = ",".join("?" * len(extra))
        for (text,) in con.execute(
            f"SELECT text FROM messages WHERE id IN ({ph}) AND text IS NOT NULL",
            extra,
        ):
            texts.append(text)
    return texts


def _dv_screen(con, conflict_id):
    """The softened DV floor for a conflict window (Master Spec A10). Scans the
    FULL text of every message in the window for high-risk threat / isolation /
    fear markers and returns a STATUS dict (NOT a bare bool) that both the
    scorecard and C4's draft_apology branch on:

      status 'clean'                -> no marker; full role analysis, no footer.
      status 'role-suppress'        -> marker + screen active; the caller MUST
                                       withhold the role/aggressor output but
                                       keep the mechanical scorecard, and append
                                       `footer`.
      status 'bypassed-with-footer' -> marker + screen bypassed
                                       (CONFLICT_DV_SCREEN=off or a per-run
                                       --no-dv-screen); role analysis is
                                       RESTORED but `footer` is STILL appended.

    `suppress_role` is the single bool the caller acts on; `footer` is None only
    for 'clean'; `markers` lists the generic categories that fired (never the
    raw matched text). The scan is read-only -- the hit is NEVER persisted.
    Requires a sqlite3.Row connection (like the rest of this module)."""
    conflict_id = int(conflict_id)
    categories = set()
    row = con.execute(
        "SELECT chat_id, first_date, last_date FROM conflicts WHERE id = ?",
        (conflict_id,)).fetchone()
    if row is not None:
        for text in _window_message_texts(
                con, conflict_id, row["chat_id"], row["first_date"], row["last_date"]):
            categories |= _dv_scan_text(text)
    if not categories:
        return {"status": "clean", "suppress_role": False, "footer": None,
                "markers": []}
    markers = sorted(categories)
    if not _dv_screen_enabled():
        return {"status": "bypassed-with-footer", "suppress_role": False,
                "footer": _DV_RESOURCE_TEXT, "markers": markers}
    return {"status": "role-suppress", "suppress_role": True,
            "footer": _DV_RESOURCE_TEXT, "markers": markers}


# --- Deterministic scorecard sections ---------------------------------------

def _side_ratios(rows):
    """I-vs-you & we-talk ratios for one side (F1 contract: SUM raw counts then
    form the rate against summed n_words; COALESCE the nullable pronoun-split
    columns). Returns None when the side has no words."""
    n_msgs = sum(r["n_messages"] for r in rows)
    n_words = sum(r["n_words"] for r in rows)
    if not n_words or not n_msgs:
        return None
    i_c = sum((r["i_count"] or 0) for r in rows)
    you_c = sum((r["you_count"] or 0) for r in rows)
    we_c = sum(r["we_count"] for r in rows)
    return {
        "n_messages": n_msgs,
        "n_words": n_words,
        "i_rate": round(i_c / n_words, 4),
        "you_rate": round(you_c / n_words, 4),
        "i_to_you_ratio": round(i_c / you_c, 3) if you_c else None,
        "we_per_100_msgs": round(100 * we_c / n_msgs, 2),
    }


def _lexicon_section(con, chat_id, first_date, last_date):
    """Per-side I-vs-you & we-talk ratios + the LSM style-matching score over
    the window's months (person_period_lexicon, chat-scoped). Deterministic;
    'available' False when the lexicon has not been built for this window."""
    lo, hi = first_date[:7], last_date[:7]
    rows = con.execute(
        "SELECT * FROM person_period_lexicon WHERE chat_id = ? "
        "AND month >= ? AND month <= ?", (chat_id, lo, hi)).fetchall()
    me_rows = [r for r in rows if r["handle_id"] is None]
    them_rows = [r for r in rows if r["handle_id"] is not None]
    me, them = _side_ratios(me_rows), _side_ratios(them_rows)
    section = {"me": me, "them": them}
    if me is None or them is None:
        section["available"] = False
        section["note"] = ("lexicon not built for this window "
                           "(person_period_lexicon is built by ingest)")
    else:
        section["available"] = True
    # LSM reuses the F1 primitives (never reimplemented).
    rates_me = _weighted_rates(me_rows) if me_rows else None
    rates_them = _weighted_rates(them_rows) if them_rows else None
    if rates_me is not None and rates_them is not None:
        section["lsm"] = round(_lsm_score(rates_me, rates_them)[0], 4)
    else:
        section["lsm"] = None
    return section


def _session_roles(con, start_session_id, end_session_id):
    """Deterministic session roles: who opened the episode's first session
    (initiator) and who had the last word in its final session (ender). Raw
    is-me facts; NOT a verdict. Always returned, even under a DV hit."""
    out = {"opener": None, "last_word": None}
    if start_session_id is not None:
        s = con.execute("SELECT initiator_is_me FROM sessions WHERE id = ?",
                        (start_session_id,)).fetchone()
        if s is not None:
            out["opener"] = "me" if s["initiator_is_me"] else "them"
    if end_session_id is not None:
        e = con.execute("SELECT ender_is_me FROM sessions WHERE id = ?",
                        (end_session_id,)).fetchone()
        if e is not None:
            out["last_word"] = "me" if e["ender_is_me"] else "them"
    return out


def _interaction_pattern(con, peak, session_roles):
    """The demand-withdraw READ -- who withdrew vs who kept reaching out --
    derived DETERMINISTICALLY from C1's peak reply-gap (the actor of the
    message left hanging pursued; the other withdrew), falling back to who had
    the last word. A NON-BLAME interactional description with NO aggressor /
    who-won verdict -- and the one section the DV floor withholds (A10)."""
    pursuer = None
    peak_mid = peak.get("peak_gap_message_id")
    if peak_mid is not None:
        m = con.execute("SELECT is_from_me FROM messages WHERE id = ?",
                        (peak_mid,)).fetchone()
        if m is not None:
            pursuer = "me" if m["is_from_me"] else "them"
    if pursuer is None:
        pursuer = session_roles.get("last_word")
    withdrawer = {"me": "them", "them": "me"}.get(pursuer)
    return {
        "pursuer": pursuer,
        "withdrawer": withdrawer,
        "basis": ("deterministic demand-withdraw read from the peak reply gap; "
                  "a mechanical pattern, NOT a judgment of fault"),
    }


def diagnose_conflict(conflict_id):
    """A deterministic scorecard for one conflict episode (by conflict_id from
    find_conflicts): lexicon I-vs-you & we-talk ratios and LSM style-matching
    per side, the session opener/last-word roles, reply-latency and
    volume-collapse read off the stored deterministic evidence, and a NON-BLAME
    demand-withdraw read (who withdrew vs pursued). It carries NO aggressor or
    who-won verdict. A code-enforced DV safety screen (Master Spec A10) scans
    the window's full text: on a high-risk marker it withholds ONLY the
    demand-withdraw read and appends a fixed hotline resource footer, while the
    mechanical scorecard still returns; CONFLICT_DV_SCREEN=off (or chat.py
    --no-dv-screen) restores the read but still appends the footer on a hit.
    The screen result is never persisted."""
    conflict_id = int(conflict_id)
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT id, chat_id, first_date, last_date, start_session_id, "
            "end_session_id, topic, peak_signal_json, corroboration, "
            "off_text_flag FROM conflicts WHERE id = ?", (conflict_id,),
        ).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}
        peak = _signals_json(row["peak_signal_json"])
        signals = peak.get("signals", [])
        result = {
            "conflict_id": row["id"],
            "chat_id": row["chat_id"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "topic": row["topic"],
            "signals": signals,
            "corroboration": row["corroboration"],
            "off_text_flag": bool(row["off_text_flag"]),
            "lexicon": _lexicon_section(
                con, row["chat_id"], row["first_date"], row["last_date"]),
            "session_roles": _session_roles(
                con, row["start_session_id"], row["end_session_id"]),
            # Latency + volume come from C1's stored survival stats
            # (stats.frac_over_threshold); get_response_dynamics is NOT called.
            "latency": {
                "peak_gap_hours": peak.get("peak_gap_hours"),
                "withdrawal_fired": "latency_withdrawal" in signals,
                "source": "peak_signal survival stats (frac_over_threshold), C1",
            },
            "volume": {
                "volume_ratio": peak.get("volume_ratio"),
                "collapse_fired": "volume_collapse" in signals,
            },
            "note": ("deterministic scorecard -- no aggressor/who-won verdict; "
                     "labels elsewhere are eval-gated bands, not certainties"),
        }
        pattern = _interaction_pattern(con, peak, result["session_roles"])
        screen = _dv_screen(con, conflict_id)

    result["dv_screen"] = screen["status"]
    if screen["suppress_role"]:
        result["interaction_pattern"] = {
            "withheld": True,
            "reason": ("high-risk safety markers present -- role/aggressor read "
                       "withheld (DV floor); the mechanical scorecard still "
                       "applies"),
        }
    else:
        result["interaction_pattern"] = pattern
    if screen["footer"]:
        result["resource_footer"] = screen["footer"]
    # Code-printed humility disclaimer (out-of-band via chat.run_turn); distinct
    # from the in-payload resource_footer, which the model narrates.
    result["_epilogue"] = _ANALYSIS_DISCLAIMER
    return result


# --- feature: conflict ---
# ===========================================================================
# C3 (RES-52): gated cross-chat conflict trace.
#
# trace_conflict_across_chats(conflict_id, k, consent) answers "did this one
# argument spill into my OTHER conversations?" -- chunks in a DIFFERENT chat,
# within a few days of the episode, that are BOTH semantically close to it AND
# literally name chat A's partner (their handle, first name, or one of the
# chat's distinctive dyad_terms). Two STRUCTURAL gates keep it safe:
#   1. Registration gate (the real boundary): the schema is registered in
#      chat.py's tool list ONLY when config.CONFLICT_TRACE_ENABLED. Off by
#      default -- withholding the schema is the repo's one reliable way to stop
#      the model calling a tool (see the CLAUDE.md gotcha), so a disabled
#      deployment never needs prompt-level "don't trace" text.
#   2. Per-episode consent (UX ONLY, NOT the boundary): without consent=True the
#      tool returns a fixed ask-the-user payload and reads no data.
# The name-reference gate filters topically-similar-but-unnamed chunks (a
# "weekend with my boss" chunk near a "weekend with Jordan" episode is dropped
# unless it actually names Jordan). Computed on demand and written to NO table
# (there is deliberately no conflict_links table): nothing here is persisted.
# ===========================================================================
import re  # noqa: E402 (local to the additive C3 block)

from db import iso_to_day  # noqa: E402

from ._shared import CHUNK_PREVIEW_CHARS, MAX_VEC_K  # noqa: E402

# Shortest token that counts as a literal partner reference -- a 1-2 char name
# or dyad_term ("i", "ok") would match half the corpus, defeating the gate.
_TRACE_MIN_REF_LEN = 3
# Query-text budget fed to the embedder (the episode's own window text).
_TRACE_QUERY_CHARS = 4000
# Over-fetch factor: the sim floor + name gate prune hard, so pull a wider KNN
# pool than k before filtering (still capped at MAX_VEC_K).
_TRACE_POOL_FACTOR = 5


def _trace_ask_payload(conflict_id):
    """The fixed no-data response returned when consent was NOT given: it
    carries zero retrieved content -- only the instruction to ask first. This
    is the UX layer, not the security boundary (registration is)."""
    return {
        "conflict_id": conflict_id,
        "status": "consent_required",
        "links": [],
        "ask": ("Tracing whether this argument spilled into your OTHER "
                "conversations reads across chats. Want me to check? I'll look "
                "only at messages that both match this episode AND name the "
                "person involved, and I won't save anything."),
        "note": ("no data was read -- call again with consent=true only after "
                 "the user agrees; nothing is ever persisted"),
    }


def _chat_partner_references(con, chat_id):
    """The literal strings a related chunk must contain to count as naming chat
    A's partner: each participant handle_identifier, the counterparty contact
    first_name(s), and chat A's distinctive dyad_terms. Returns an ordered,
    de-duplicated list of (kind, lowercased_value); blank/too-short values are
    dropped so the gate stays meaningful."""
    refs = []
    for hid, first_name in con.execute(
        "SELECT h.handle_identifier, c.first_name "
        "FROM chat_participants cp "
        "JOIN handles h ON h.id = cp.handle_id "
        "LEFT JOIN contacts c ON c.id = h.contact_id "
        "WHERE cp.chat_id = ?", (chat_id,),
    ):
        if hid and len(hid) >= _TRACE_MIN_REF_LEN:
            refs.append(("handle", hid.lower()))
        if first_name and len(first_name) >= _TRACE_MIN_REF_LEN:
            refs.append(("name", first_name.lower()))
    for (term,) in con.execute(
        "SELECT term FROM dyad_terms WHERE chat_id = ?", (chat_id,),
    ):
        if term and len(term) >= _TRACE_MIN_REF_LEN:
            refs.append(("dyad_term", term.lower()))
    seen, out = set(), []
    for kind, value in refs:
        if value not in seen:
            seen.add(value)
            out.append((kind, value))
    return out


def _mentions_reference(text, references):
    """The first partner reference `text` literally mentions ({kind, value}) or
    None. Handle identifiers (phones/emails) match as substrings; names and
    dyad_terms match on WORD BOUNDARIES, so 'Jordan' fires on 'weekend with
    Jordan' but not on 'Jordania', and a topically-similar chunk that names no
    one is filtered out."""
    if not text:
        return None
    lowered = text.lower()
    for kind, value in references:
        if kind == "handle":
            if value in lowered:
                return {"kind": kind, "value": value}
        elif re.search(rf"\b{re.escape(value)}\b", lowered):
            return {"kind": kind, "value": value}
    return None


def trace_conflict_across_chats(conflict_id, k=10, consent=False):
    """Trace whether ONE conflict episode (a conflict_id from find_conflicts)
    spilled into your OTHER conversations: conversation chunks in a DIFFERENT
    chat, within CONFLICT_TRACE_BUFFER_DAYS of the episode, that are BOTH
    semantically close to it AND literally name chat A's partner (their handle,
    first name, or one of the chat's distinctive terms). A link needs cosine
    similarity > CONFLICT_TRACE_SIM_THRESHOLD AND a name hit -- topically
    similar but unnamed chunks are dropped.

    Cross-chat by nature, so it is off by default and consent-gated. The real
    off-switch is REGISTRATION: the whole tool is withheld from the model unless
    config.CONFLICT_TRACE_ENABLED. The `consent` param is a UX prompt only, NOT
    the security boundary -- without consent=True this returns a fixed
    ask-the-user payload and reads no data. Computed on demand; NOTHING is
    persisted (there is deliberately no conflict_links table)."""
    conflict_id = int(conflict_id)
    if not consent:
        return _trace_ask_payload(conflict_id)

    # Local import keeps this additive block self-contained and cycle-safe (the
    # package is fully initialized by the time a tool is called).
    from .chunks import _chunk_knn, _embed_query

    k = max(1, min(int(k), MAX_VEC_K))
    buffer = max(0, int(config.CONFLICT_TRACE_BUFFER_DAYS))
    threshold = float(config.CONFLICT_TRACE_SIM_THRESHOLD)

    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"conflict_id": conflict_id, "links": [],
                    "error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT chat_id, first_date, last_date, topic FROM conflicts "
            "WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            return {"conflict_id": conflict_id, "links": [],
                    "error": f"no conflict with id {conflict_id}"}
        origin_chat = row["chat_id"]
        first_date, last_date = row["first_date"], row["last_date"]
        # Convert the episode's UTC-ISO message dates to vec0 day numbers with
        # db.iso_to_day -- the SAME converter the metadata writer uses; mixing
        # formats silently empties the scan (see the CLAUDE.md gotcha).
        from_day = iso_to_day(first_date) - buffer
        to_day = iso_to_day(last_date) + buffer

        references = _chat_partner_references(con, origin_chat)
        # Query text = the episode's own full window text (same source the DV
        # screen reads), falling back to its topic when the window has no text.
        window_texts = _window_message_texts(
            con, conflict_id, origin_chat, first_date, last_date)
        query_text = " ".join(t for t in window_texts if t)[:_TRACE_QUERY_CHARS]
        if not query_text.strip():
            query_text = (row["topic"] or "")
        if not query_text.strip():
            return {"conflict_id": conflict_id, "origin_chat_id": origin_chat,
                    "links": [], "returned": 0,
                    "note": "no episode text to trace; nothing persisted"}

        query_vec = _embed_query(query_text)
        pool = max(1, min(k * _TRACE_POOL_FACTOR, MAX_VEC_K))
        candidates = _chunk_knn(
            con, query_vec, pool, chat_ids=None, from_day=from_day, to_day=to_day,
            exclude_chat_ids=[origin_chat])

    links = []
    for r in candidates:
        distance = r.get("distance")
        if distance is None:
            continue
        similarity = 1.0 - float(distance)
        if similarity <= threshold:
            continue
        matched = _mentions_reference(r.get("text") or "", references)
        if matched is None:
            continue
        links.append({
            "chunk_id": r["chunk_id"],
            "chat_id": r["chat_id"],
            "participants": r.get("participants"),
            "first_date": r["first_date"],
            "last_date": r["last_date"],
            "similarity": round(similarity, 4),
            "matched_reference": matched,
            "text": (r.get("text") or "")[:CHUNK_PREVIEW_CHARS],
        })
        if len(links) >= k:
            break

    return {
        "conflict_id": conflict_id,
        "origin_chat_id": origin_chat,
        "window": {"first_date": first_date, "last_date": last_date,
                   "buffer_days": buffer},
        "sim_threshold": threshold,
        "n_references": len(references),
        "links": links,
        "returned": len(links),
        "note": ("cross-chat spillover -- each link BOTH exceeds the similarity "
                 "threshold AND literally names the partner; computed on demand "
                 "and NOT persisted (no conflict_links table exists)"),
    }
# --- end feature: conflict ---


# --- feature: conflict (C4 apology) ---
# ===========================================================================
# C4 (RES-53): draft_apology scaffold (Lewicki-6, NEVER ghostwrites).
#
# draft_apology(conflict_id) SCAFFOLDS a repair message; it does not write one.
# It returns the ranked Lewicki-6 apology checklist (Lewicki, Polin & Lount
# 2016, "An Exploration of the Structure of Effective Apologies" -- the six
# components ordered by MEASURED effectiveness: acknowledging responsibility
# first, offering repair second, requesting forgiveness last and flagged
# skippable), the user's REAL grievance excerpts pulled VERBATIM from the
# episode's labeled turns (never invented), and -- only where the `repair`
# escalation label has cleared its Cohen's-kappa eval gate
# (eval_gate_passed(con, 'conflict_labels', 'repair')) -- the ignored repair
# attempts the other person made. When `repair` is NOT eval-cleared that
# component is OMITTED and returned as {"status": "unavailable", ...} rather
# than guessed from raw text; the checklist + excerpts still return.
#
# It emits NO finished prose (emits_prose is always False). Any finished draft
# must sit behind a SECOND explicit confirmation and be labeled "draft template
# -- edit before sending" (guidance carried in the payload note).
#
# DV floor (Master Spec A10, reusing C2's _dv_screen / _DV_RESOURCE_TEXT):
# drafting an apology is ITSELF the risky output -- it can pressure a victim
# into apologizing to an abuser -- so UNLIKE diagnose_conflict there is NO
# partial scaffold. A 'role-suppress' screen (marker hit, screening on)
# WITHHOLDS the entire scaffold and returns ONLY the code-emitted resource
# pointer. A 'bypassed-with-footer' screen (screening off) returns the full
# scaffold with the resource footer appended. The pointer is never fully
# removed; the screen result is never persisted.
# ===========================================================================

# The Lewicki-6 checklist, ranked by measured effectiveness (rank 1 = highest).
# A module constant, returned as-is and never mutated; grounding excerpts (not
# generated prose) are attached separately per call.
_LEWICKI_CHECKLIST = (
    {"rank": 1, "component": "acknowledge_responsibility",
     "prompt": ("Name what you did and own it plainly -- no 'if', no 'but', no "
                "shifting it onto them."),
     "skippable": False,
     "note": "most effective component (Lewicki, Polin & Lount 2016)"},
    {"rank": 2, "component": "offer_of_repair",
     "prompt": ("Say concretely how you'll make it right or keep it from "
                "happening again."),
     "skippable": False,
     "note": "second most effective"},
    {"rank": 3, "component": "expression_of_regret",
     "prompt": "Say you're sorry and that you regret the hurt it caused.",
     "skippable": False},
    {"rank": 4, "component": "explanation",
     "prompt": ("Briefly explain what happened -- an honest account, NOT an "
                "excuse."),
     "skippable": False},
    {"rank": 5, "component": "declaration_of_repentance",
     "prompt": "State that you don't intend to repeat it.",
     "skippable": False},
    {"rank": 6, "component": "request_for_forgiveness",
     "prompt": ("Optionally ask for forgiveness -- never demand or expect it, "
                "and skip it if pressing for it would burden them."),
     "skippable": True,
     "note": "least effective; safe to omit"},
)

# Escalation labels that count as grievance turns (the horsemen); NOT 'repair'
# (its own eval-gated component) and NOT 'abstain'.
_GRIEVANCE_CLASSES = ("criticism", "contempt", "defensiveness", "stonewalling")
# Per-component excerpt cap (kept well under TOOL_RESULT_MAX_CHARS alongside the
# 200-char EXCERPT_CAP re-cap on each stored excerpt).
_APOLOGY_MAX_EXCERPTS = 15


def _grievance_excerpts(con, conflict_id):
    """The episode's REAL grievance turns: VERBATIM conflict_signals excerpts
    labeled with one of the horsemen classes (never invented, never guessed),
    each re-capped at EXCERPT_CAP and tagged with the actor. Ordered as stored;
    capped at _APOLOGY_MAX_EXCERPTS."""
    ph = ",".join("?" * len(_GRIEVANCE_CLASSES))
    rows = con.execute(
        f"SELECT message_id, actor_is_me, label_class, excerpt "
        f"FROM conflict_signals WHERE conflict_id = ? "
        f"AND label_class IN ({ph}) AND excerpt IS NOT NULL "
        f"ORDER BY id LIMIT ?",
        (conflict_id, *_GRIEVANCE_CLASSES, _APOLOGY_MAX_EXCERPTS)).fetchall()
    return [{
        "message_id": r["message_id"],
        "actor": "me" if r["actor_is_me"] else "them",
        "label_class": r["label_class"],
        "excerpt": (r["excerpt"] or "")[:EXCERPT_CAP],
    } for r in rows]


def _ignored_repair_attempts(con, conflict_id):
    """The repair BIDS the OTHER person made during the episode (label_class
    'repair', actor them): VERBATIM conflict_signals excerpts the user can
    acknowledge, never guessed. Called ONLY when the `repair` label class is
    eval-cleared -- its rows only exist at all when that class cleared the gate
    at label-write time (a suppressed class writes zero rows)."""
    rows = con.execute(
        "SELECT message_id, actor_is_me, excerpt FROM conflict_signals "
        "WHERE conflict_id = ? AND label_class = 'repair' AND actor_is_me = 0 "
        "AND excerpt IS NOT NULL ORDER BY id LIMIT ?",
        (conflict_id, _APOLOGY_MAX_EXCERPTS)).fetchall()
    return [{
        "message_id": r["message_id"],
        "actor": "them",
        "excerpt": (r["excerpt"] or "")[:EXCERPT_CAP],
    } for r in rows]


def draft_apology(conflict_id):
    """SCAFFOLD (never ghostwrite) a repair message for one conflict episode (a
    conflict_id from find_conflicts). Returns the ranked Lewicki-6 apology
    checklist (Lewicki, Polin & Lount 2016 -- components ordered by measured
    effectiveness: acknowledging responsibility first, offering repair second,
    requesting forgiveness last and flagged skippable), the user's REAL
    grievance excerpts pulled VERBATIM from the episode (never invented), and --
    only where the `repair` escalation label has cleared its eval gate -- the
    ignored repair attempts the other person made; when `repair` is not
    eval-cleared that component is omitted and marked unavailable rather than
    guessed from raw text. It emits NO finished apology prose (emits_prose is
    False) -- do not ghostwrite the message; walk the user through the checklist
    grounded in their real excerpts.

    DV safety floor (Master Spec A10): the same code-enforced screen
    diagnose_conflict uses runs over the full conflict window. Because drafting
    an apology is ITSELF the risky output (it can pressure a person into
    apologizing to someone who is hurting them), a role-suppress result WITHHOLDS
    the ENTIRE scaffold and returns ONLY the fixed hotline resource pointer --
    there is NO partial scaffold here (unlike diagnose_conflict). With screening
    off (CONFLICT_DV_SCREEN=off or chat.py --no-dv-screen) the scaffold returns
    but STILL carries the appended footer on a marker hit. The screen result is
    never persisted."""
    conflict_id = int(conflict_id)
    with contextlib.closing(_connect()) as con:
        if not _has_conflict_tables(con):
            return {"error": "no conflict tables (run conflict_detect.py first)"}
        row = con.execute(
            "SELECT id, chat_id, first_date, last_date, topic, off_text_flag "
            "FROM conflicts WHERE id = ?", (conflict_id,)).fetchone()
        if row is None:
            return {"error": f"no conflict with id {conflict_id}"}

        screen = _dv_screen(con, conflict_id)
        # DV role-suppress: drafting the apology is itself the risk. WITHHOLD the
        # whole scaffold and return ONLY the code-emitted resource pointer -- no
        # partial scaffold, unlike diagnose_conflict's mechanical section.
        if screen["suppress_role"]:
            return {
                "conflict_id": row["id"],
                "chat_id": row["chat_id"],
                "dv_screen": screen["status"],
                "withheld": True,
                "reason": ("high-risk safety markers present -- the apology "
                           "scaffold is withheld entirely; drafting an apology "
                           "can pressure a person into apologizing to someone "
                           "who is hurting them"),
                "resource_footer": screen["footer"],
            }

        # Scaffold path (screen is 'clean' or 'bypassed-with-footer'). Lazy,
        # Ollama-free import (app modules are pulled in lazily by gates).
        from eval.evalkit.gates import eval_gate_passed

        result = {
            "conflict_id": row["id"],
            "chat_id": row["chat_id"],
            "first_date": row["first_date"],
            "last_date": row["last_date"],
            "topic": row["topic"],
            "off_text_flag": bool(row["off_text_flag"]),
            "dv_screen": screen["status"],
            "checklist": list(_LEWICKI_CHECKLIST),
            "grievances": _grievance_excerpts(con, conflict_id),
            "emits_prose": False,
            "note": ("this is a SCAFFOLD, not an apology -- it emits no finished "
                     "prose. Do NOT ghostwrite the message; walk the user "
                     "through the checklist grounded in their real grievance "
                     "excerpts. If they explicitly ask for drafted text AFTER "
                     "seeing this, produce it only on a second explicit "
                     "confirmation and label it 'draft template -- edit before "
                     "sending'."),
        }
        # Repair-attempts component is eval-gated: present only when the `repair`
        # label class has cleared its kappa gate (absent-key => suppressed).
        if eval_gate_passed(con, "conflict_labels", "repair"):
            result["repair_attempts"] = {
                "status": "available",
                "attempts": _ignored_repair_attempts(con, conflict_id),
                "note": ("repair bids the other person made -- acknowledging "
                         "these is part of an effective repair"),
            }
        else:
            result["repair_attempts"] = {
                "status": "unavailable",
                "reason": "label class not eval-cleared",
            }

    # The resource pointer is never fully removed: a 'bypassed-with-footer' hit
    # appends the footer even though role analysis was restored.
    if screen["footer"]:
        result["resource_footer"] = screen["footer"]
    return result
# --- end feature: conflict (C4 apology) ---
