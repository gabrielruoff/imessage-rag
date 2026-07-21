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


def _conflict_summary(con, row):
    """Compact conflict row for find_conflicts (no per-signal excerpts)."""
    peak = _signals_json(row["peak_signal_json"])
    n_signal_rows = con.execute(
        "SELECT COUNT(*) FROM conflict_signals WHERE conflict_id = ? "
        "AND label_class != 'abstain'", (row["id"],)).fetchone()[0]
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
        "labeled_signal_turns": n_signal_rows,
        "peak_gap_hours": peak.get("peak_gap_hours"),
    }


def find_conflicts(chat_id=None, handle_id=None, contact_id=None,
                   date_from=None, date_to=None, limit=10):
    """Detected conflict episodes in scope, ranked by deterministic signal
    strength then recency. Each is a tense stretch found from deterministic
    signals WITH RECEIPTS (reply-latency withdrawal, one-sided sessions,
    dislike-tapback runs, volume collapse); `signals` names which fired and
    `signal_count` how many. 1:1 chats only. handle_id/contact_id scope to that
    person's direct chats; date bounds filter on the episode start. corroboration
    is 'none' unless source='llm' annotations backed the tone drop. Returns an
    envelope {rows, total_matching, returned, order, truncated}; expand one with
    get_conflict(conflict_id)."""
    limit = max(1, min(int(limit), MAX_CONFLICT_ROWS))
    order = "signal_count-desc, most-recent-first"
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
            clauses.append(f"chat_id IN ({','.join('?' * len(chat_ids))})")
            params.extend(chat_ids)
        _date_clauses(clauses, params, date_from, date_to, column="first_date")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = con.execute(
            f"SELECT id, chat_id, first_date, last_date, topic, peak_signal_json, "
            f"corroboration, off_text_flag FROM conflicts{where}", params,
        ).fetchall()
        summaries = [_conflict_summary(con, r) for r in rows]

    # Strongest (most distinct deterministic signals) first, then most recent.
    summaries.sort(key=lambda s: (s["signal_count"], s["last_date"] or ""),
                   reverse=True)
    total = len(summaries)
    return _envelope(summaries[:limit], total, order)


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
    return result
