"""Wrapped analytical-suite sections (drift / persona / conflict / wellbeing /
nudges).

Split out of wrapped_sections.py: the feature-fenced section builders that
build_sections appends when their feature is present. Re-exported from
wrapped_sections so callers/tests keep using wrapped_sections._sec_*. Reads
the shared section helpers from wrapped_sections and the visual builders from
wrapped_html.
"""

import config
import db
import wrapped
import wrapped_html
import wrapped_sections

from wrapped_sections import _person_chat_ids, _session_duration


# --- feature: drift ---
def _sec_drifting(con, excluded_names=frozenset(), limit=8):
    """Opt-in "Who's Drifting" section (RES-66 / D4) -- a reflective surface,
    OFF by default. Renders ONLY when config.DRIFT_ENABLED, and only for
    phase_corroborated fading/dormant ties: a decline label that an
    intra-iMessage corroborator (reciprocity erosion, or reply-time inflation)
    actually backs. Every number is read verbatim from the drift model's
    dyad_drift table -- no re-scoring, no Ollama -- so the narrator only
    narrates already-computed values. Returns None (section omitted) when
    drift is disabled, the table is absent (a pre-feature DB), or nothing
    corroborated survives the report's person exclusions.

    Split bars contrast the two corroborating signals' magnitude per tie;
    open reconnection-window chips flag ties still inside the window before the
    drift model would count them dormant -- the same framing as
    query_tools.drift._reconnection_window, reading config.DRIFT_PHASE_THRESHOLDS
    at call time (the repo convention). excluded_names keeps a person the
    report deliberately dropped (a named --exclude, or the silent-contact
    guard) out of this section too: a 'who's drifting' card for an excluded ex
    would defeat the exclusion."""
    import json  # function-local: keeps this additive block off the top imports
    if not config.DRIFT_ENABLED:
        return None
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dyad_drift'"
    ).fetchone():
        return None
    rows = con.execute(
        """
        SELECT contact_key, label, phase, drift_score,
               reciprocity_value, latency_value, coverage_json,
               last_month, confidence_tier, n_messages
        FROM dyad_drift
        WHERE phase_corroborated = 1 AND phase IN ('fading', 'dormant')
        ORDER BY drift_score DESC
        """
    ).fetchall()

    dormant_days = float(config.DRIFT_PHASE_THRESHOLDS.get("dormant_days", 90))
    per_tie, windows, seen = [], [], set()
    for row in rows:
        label = row["label"] or row["contact_key"]
        if str(label).lower() in excluded_names or row["contact_key"] in seen:
            continue
        seen.add(row["contact_key"])
        try:
            coverage = json.loads(row["coverage_json"] or "{}")
        except (TypeError, ValueError):
            coverage = {}
        gap_days = coverage.get("last_message_days")
        per_tie.append({
            "contact": label,
            "reciprocity_erosion": round(row["reciprocity_value"] or 0.0, 3),
            "latency_inflation": round(row["latency_value"] or 0.0, 3),
            "phase": row["phase"],
            "quiet_for_days": gap_days,
            "confidence": row["confidence_tier"],
        })
        # Days left before the dormant threshold, from the stored gap -- same
        # open/closing/overdue framing as query_tools.drift._reconnection_window.
        if isinstance(gap_days, (int, float)) and not isinstance(gap_days, bool):
            days_until = round(dormant_days - gap_days)
            if row["phase"] == "dormant" or days_until <= 0:
                status = "overdue"
            elif days_until <= 14:
                status = "closing"
            else:
                status = "open"
            if status in ("open", "closing"):
                windows.append({"contact": label,
                                "days_until_dormant": days_until,
                                "status": status})
        if len(per_tie) >= limit:
            break

    if not per_tie:
        return None

    split = wrapped_html.render_split_bars(
        per_tie, "contact", "reciprocity_erosion", "latency_inflation",
        left_label="Reciprocity erosion", right_label="Reply-time drift")
    if split is None:  # nothing chartable -- degrade to the table view
        split = wrapped_html._rows_table(per_tie, wrapped_html._columns(per_tie))
    visuals = [wrapped_html.block("Cooling ties", split)]
    chips = wrapped_html.render_window_chips(windows)
    if chips:
        visuals.append(wrapped_html.block("Still time to reach out", chips))

    return {
        "title": "Who's Drifting",
        "stats": {},
        "narrate_stats": {"cooling_ties": per_tie,
                          "reconnection_windows": windows},
        "extra_html": "".join(visuals),
        "caption": ("iMessage only -- a thread that has gone quiet here may be "
                    "loud on calls, in person, or elsewhere. Only ties whose "
                    "cooling is corroborated by an eroding back-and-forth or "
                    "stretching reply times appear; both magnitudes are read "
                    "from the drift model, never re-scored."),
        "hint": ("A gentle, reflective look at 1:1 ties that have cooled on "
                 "iMessage ONLY -- never diagnose a relationship or give advice. "
                 "Reciprocity erosion is how lopsided the sending became; "
                 "reply-time drift is how much slower replies got (both are 0-1 "
                 "magnitudes). A reconnection window is the days left before a "
                 "tie is counted dormant. Note softly that iMessage is one "
                 "channel, mention at most one or two ties, and quote the "
                 "numbers exactly as given."),
    }
# --- end feature: drift ---


# --- feature: persona ---
def _sec_voiceprint(roster, contact_entry=None):
    """"Your Voiceprint" (RES-64 / V4): the reader's OWN deterministic style
    voiceprint -- the ("me", "*", "all") card from get_voiceprint -- rendered in
    iMessage bubbles, plus a consent-gated "guess who texts like this" game.

    PRIVACY (third-party content is default-redacted): a contact's verbatim
    signature phrases and stylometric card enter the generated file ONLY when
    that contact's `contacts.id` is in WRAPPED_VOICEPRINT_CONSENT_IDS (parsed by
    consent.py -- never re-implemented here). With the default empty list the
    section is self-only and the game is ABSENT. Consent ADDS a contact; it never
    overrides a roster exclusion -- the game iterates the already-exclusion-
    filtered featured roster (the silent-contact guard, --contact, and --exclude
    have all run before build_sections), so a consented-but-excluded contact is
    simply not present to feature. Nothing is ever sent; this only renders."""
    import consent

    me_card, note = wrapped._tool("get_voiceprint")  # ("me", "*", "all")
    if not (isinstance(me_card, dict) and "features" in me_card):
        reason = note or (me_card.get("note") if isinstance(me_card, dict) else None)
        return {"title": "Your Voiceprint", "stats": None,
                "note": reason or "no voiceprint on file yet -- run `make voiceprint`"}

    feats = me_card.get("features") or {}
    stats = {}
    if feats.get("avg_msg_len") is not None:
        stats["typical_message_length"] = f"{round(feats['avg_msg_len']):,} chars"
    if feats.get("lowercase_ratio") is not None:
        stats["lowercase_share"] = feats["lowercase_ratio"]  # 'share' -> percent
    if feats.get("burst_size") is not None:
        stats["typical_burst"] = f"{round(feats['burst_size']):,} messages"
    if me_card.get("confidence"):
        stats["confidence"] = me_card["confidence"]

    narrate = {"you": {
        "openers": wrapped_html._voiceprint_phrases(me_card.get("openers")),
        "closers": wrapped_html._voiceprint_phrases(me_card.get("closers")),
        "typical_message_length_chars": (round(feats["avg_msg_len"])
                                         if feats.get("avg_msg_len") is not None else None),
    }}

    visuals = []
    you = wrapped_html.render_voiceprint_card(
        me_card.get("openers"), me_card.get("closers"), outgoing=True)
    if you:
        visuals.append(wrapped_html.block("This is you", you))

    # Consent-gated third-party game. Default-redacted: a contact's card is
    # rendered ONLY when opted in AND present in the featured roster (which the
    # exclusions already filtered), so consent can never resurface an excluded
    # contact. Empty consent list -> the loop bodies never run -> no game.
    consent_raw = getattr(config, "WRAPPED_VOICEPRINT_CONSENT_IDS", "")
    mystery = []
    for entry in roster:
        cid = entry.get("contact_id")
        if cid is None or not consent.is_consented(cid, consent_raw):
            continue
        card, _card_note = wrapped._tool("get_voiceprint", contact_id=cid,
                                         subject="them", scope="all")
        if not (isinstance(card, dict) and "features" in card):
            continue
        clue = wrapped_html.render_voiceprint_card(
            card.get("openers"), card.get("closers"),
            outgoing=False, reveal=entry.get("name"))
        if clue:
            mystery.append(wrapped_html.block(f"Mystery #{len(mystery) + 1}",
                                              clue, depth=1))
    if mystery:
        visuals.append(wrapped_html.block("Guess Who Texts Like This?",
                                          "".join(mystery)))
        narrate["guess_who"] = {"players": len(mystery)}

    if not (stats or visuals):
        return {"title": "Your Voiceprint", "stats": None,
                "note": "no voiceprint on file yet"}
    caption = ("Your voiceprint is a deterministic fingerprint of how you text -- "
               "message length, casing, and the phrases you open and close with.")
    if mystery:
        caption += (" The guess-who game features only friends who explicitly "
                    "opted in; open a card to reveal who it is.")
    return {
        "title": "Your Voiceprint",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": caption,
        "hint": ("This is the reader's own texting style -- mention a couple of "
                 "signature traits (a favorite opener, message length, a lowercase "
                 "habit) warmly. If there is a guess-who game, tease it WITHOUT "
                 "naming any of the players."),
    }
# --- end feature: persona ---


# --- feature: conflict ---
# C5 (RES-54): the Wrapped "Repair Report" section. Deterministic aggregates
# over the conflict finder's tables (conflicts / conflict_signals, RES-46 / C1);
# the LLM only narrates the numbers, it never computes them.
#
# EVERY repair-derived stat is eval-gated (RES-70 / F4): a label class appears
# only when eval_gate_passed(con, 'conflict_labels', <class>) clears it, so a
# below-kappa class is SUPPRESSED, not softened. Gated stats:
#   * personal most-common horseman -- computed over ONLY the gate-cleared
#     Gottman horsemen, so the class it surfaces always cleared kappa;
#   * most-repaired relationship AND the recovery-time trend -- BOTH ride the
#     one 'repair' gate; if 'repair' hasn't cleared, BOTH vanish entirely.
# Recovery time is a resolution claim, so off_text_flag=1 windows (a large
# intra-window gap forbids a resolution verdict) are excluded from it.
# The deterministic-finder facts (window counts, cool-off gap, volume collapse)
# are NOT label-derived and are never gated -- they are the fallback the section
# renders when no class clears; with no conflict windows at all the builder
# returns None (skipped), never empty scaffolding.
import json  # noqa: E402 (local to the additive C5 block)

_CONFLICT_GATE_TASK = "conflict_labels"
_HORSEMEN = ("criticism", "contempt", "defensiveness", "stonewalling")
_HORSEMAN_LABELS = {
    "criticism": "Criticism", "contempt": "Contempt",
    "defensiveness": "Defensiveness", "stonewalling": "Stonewalling",
}


def _horseman_label(cls):
    return _HORSEMAN_LABELS.get(cls, str(cls).replace("_", " ").capitalize())


def _conflict_class_cleared(con, cls):
    """Per-class eval clearance for the conflict_labels gate (RES-70 / F4). A
    thin module-level seam: the import stays lazy (only runs once the section
    has conflicts to report) and tests can patch this one point to drive the
    suppressed path. eval_gate_passed treats an absent/failing verdict as
    False, so an uncleared class is suppressed."""
    from eval.evalkit.gates import eval_gate_passed
    return eval_gate_passed(con, _CONFLICT_GATE_TASK, cls)


def _conflict_tables_present(con):
    names = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('conflicts', 'conflict_signals')")}
    return {"conflicts", "conflict_signals"} <= names


def _conflict_scope(con, contact_entry, year, exclude_chat_ids):
    """(where, params) restricting `conflicts` (aliased `cf`) to the report
    scope: the report year (on first_date), the deep-dive contact's chats, and
    the excluded chats. Columns are qualified `cf.*` so the clause drops
    unchanged into the join queries. Returns None when a contact scope resolves
    to no chats. Plain-string date compares only (see the CLAUDE.md gotcha)."""
    clauses, params = ["1 = 1"], []
    if year is not None:
        clauses += ["cf.first_date >= ?", "cf.first_date < ?"]
        params += [f"{year:04d}-01-01", f"{year + 1:04d}-01-01"]
    for cid in exclude_chat_ids:
        clauses.append("cf.chat_id != ?")
        params.append(cid)
    if contact_entry is not None:
        chat_ids = _person_chat_ids(con, contact_entry)
        if not chat_ids:
            return None
        clauses.append(f"cf.chat_id IN ({','.join('?' * len(chat_ids))})")
        params += chat_ids
    return " AND ".join(clauses), params


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return None
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _conflict_base(con, where, params, period_width):
    """Deterministic finder facts (NOT label-derived, never gated): the number
    of conflict windows in scope, the typical cool-off gap (median peak reply
    gap), how many windows saw a volume collapse, and the window count per
    period. None when scope has no windows."""
    rows = con.execute(
        f"SELECT cf.first_date, cf.peak_signal_json FROM conflicts cf "
        f"WHERE {where}", params).fetchall()
    if not rows:
        return None
    gaps, collapses, by_period = [], 0, {}
    for row in rows:
        try:
            peak = json.loads(row["peak_signal_json"]) if row["peak_signal_json"] else {}
        except (TypeError, ValueError):
            peak = {}
        gap = peak.get("peak_gap_hours")
        if isinstance(gap, (int, float)) and not isinstance(gap, bool):
            gaps.append(float(gap))
        if "volume_collapse" in (peak.get("signals") or []):
            collapses += 1
        period = (row["first_date"] or "")[:period_width]
        if period:
            by_period[period] = by_period.get(period, 0) + 1
    return {
        "conflict_windows": len(rows),
        "typical_gap_hours": _median(gaps) if gaps else None,
        "volume_collapses": collapses,
        "windows_by_period": [{"period": p, "windows": by_period[p]}
                              for p in sorted(by_period)],
    }


def _personal_horseman(con, where, params):
    """The reader's own most-common Gottman horseman (actor_is_me = 1),
    computed over ONLY the horseman classes that cleared the eval gate -- so
    the surfaced class always cleared kappa. None when no cleared horseman
    fired in scope."""
    # Looked up through the wrapped_sections module so tests that patch
    # wrapped_sections._conflict_class_cleared reach this call (the helper lives
    # here and is re-exported there); same object when unpatched.
    cleared = [h for h in _HORSEMEN
               if wrapped_sections._conflict_class_cleared(con, h)]
    if not cleared:
        return None
    placeholders = ",".join("?" * len(cleared))
    rows = con.execute(
        f"SELECT cs.label_class AS label_class, COUNT(*) AS n "
        f"FROM conflict_signals cs JOIN conflicts cf ON cf.id = cs.conflict_id "
        f"WHERE {where} AND cs.actor_is_me = 1 "
        f"AND cs.label_class IN ({placeholders}) "
        f"GROUP BY cs.label_class ORDER BY n DESC, cs.label_class",
        params + cleared).fetchall()
    if not rows:
        return None
    breakdown = [{"horseman": _horseman_label(r["label_class"]), "your_turns": r["n"]}
                 for r in rows]
    return {"horseman": breakdown[0]["horseman"],
            "your_turns": breakdown[0]["your_turns"], "breakdown": breakdown}


def _most_repaired(con, where, params):
    """The relationship with the most repair turns in scope (either side), by
    chat, resolved to the counterpart's contact label. Caller gates this on the
    'repair' class clearing. None when no repair turns in scope."""
    row = con.execute(
        f"SELECT cf.chat_id AS chat_id, COUNT(*) AS n, "
        f"NULLIF(ch.display_name, '') AS display_name, "
        f"(SELECT {db.CONTACT_LABEL_SQL} FROM chat_participants cp "
        f"JOIN handles h ON h.id = cp.handle_id "
        f"LEFT JOIN contacts c ON c.id = h.contact_id "
        f"WHERE cp.chat_id = cf.chat_id ORDER BY h.id LIMIT 1) AS partner "
        f"FROM conflict_signals cs JOIN conflicts cf ON cf.id = cs.conflict_id "
        f"JOIN chats ch ON ch.id = cf.chat_id "
        f"WHERE {where} AND cs.label_class = 'repair' "
        f"GROUP BY cf.chat_id ORDER BY n DESC, cf.chat_id LIMIT 1",
        params).fetchone()
    if row is None:
        return None
    label = row["display_name"] or row["partner"] or f"chat {row['chat_id']}"
    return {"with": label, "repair_turns": row["n"]}


def _recovery_trend(con, where, params, period_width):
    """Average recovery time per period for repaired, RESOLVABLE conflicts:
    windows that (a) have off_text_flag = 0 -- off-text forbids a resolution
    verdict, so those windows are EXCLUDED -- AND (b) contain at least one
    'repair' turn (a conflict with no repair has no recovery to time). Recovery
    time is the window's own span (first_date -> last_date). Caller gates this
    on the 'repair' class clearing."""
    rows = con.execute(
        f"SELECT cf.first_date, cf.last_date FROM conflicts cf "
        f"WHERE {where} AND cf.off_text_flag = 0 AND EXISTS ("
        f"SELECT 1 FROM conflict_signals cs "
        f"WHERE cs.conflict_id = cf.id AND cs.label_class = 'repair')",
        params).fetchall()
    buckets = {}
    for row in rows:
        dur = _session_duration(row["first_date"], row["last_date"])
        if dur is None:
            continue
        period = (row["first_date"] or "")[:period_width]
        if period:
            buckets.setdefault(period, []).append(dur)
    return [{"period": p, "recovery_seconds": round(sum(v) / len(v)),
             "conflicts": len(v)} for p, v in sorted(buckets.items())]


def _sec_repair_report(con, contact_entry=None, year=None, exclude_chat_ids=()):
    """The Repair Report: deterministic conflict aggregates, LLM-narrated. The
    deterministic-finder facts always render (the fallback); the label-derived
    stats (horseman, most-repaired relationship, recovery-time trend) appear
    only for the eval-cleared classes. None (skipped) when there is no conflict
    data in scope -- never empty scaffolding."""
    if not _conflict_tables_present(con):
        return None
    scope = _conflict_scope(con, contact_entry, year, exclude_chat_ids)
    if scope is None:
        return None
    where, params = scope
    period_width = 7 if year is not None else 4
    base = _conflict_base(con, where, params, period_width)
    if base is None:  # no conflict windows in scope -> skip
        return None

    stats, narrate, visuals = {}, {}, []
    # Deterministic-finder fallback (never gated): always survives here.
    stats["conflict_windows"] = narrate["conflict_windows"] = base["conflict_windows"]
    if base["typical_gap_hours"] is not None:
        secs = round(base["typical_gap_hours"] * 3600)
        stats["typical_cool_off_seconds"] = secs
        narrate["typical_cool_off"] = wrapped_html.format_duration(secs)
    if base["volume_collapses"]:
        narrate["windows_with_volume_collapse"] = base["volume_collapses"]
    if len(base["windows_by_period"]) > 1:
        narrate["windows_by_period"] = base["windows_by_period"]
        chart = wrapped_html.render_hbars(
            base["windows_by_period"], "period", "windows", table=False)
        if chart:
            visuals.append(wrapped_html.block("Conflict windows over time", chart))

    # Gated stat 1: personal most-common horseman (per-class gate; the helper
    # already restricts to gate-cleared classes).
    horseman = _personal_horseman(con, where, params)
    if horseman is not None:
        stats["your_top_horseman"] = horseman["horseman"]
        narrate["your_most_common_horseman"] = horseman
        if len(horseman["breakdown"]) > 1:
            chart = wrapped_html.render_hbars(
                horseman["breakdown"], "horseman", "your_turns")
            if chart:
                visuals.append(wrapped_html.block("Your escalation moves", chart))

    # Gated stats 2 & 3: most-repaired relationship AND recovery-time trend both
    # ride the single 'repair' gate -- if it has not cleared, BOTH are omitted.
    if wrapped_sections._conflict_class_cleared(con, "repair"):
        repaired = _most_repaired(con, where, params)
        if repaired is not None:
            stats["most_repaired_relationship"] = repaired["with"]
            narrate["most_repaired_relationship"] = repaired
        trend = _recovery_trend(con, where, params, period_width)
        if len(trend) > 1:
            narrate["recovery_time_trend"] = trend
            chart = wrapped_html.render_hbars(
                trend, "period", "recovery_seconds", table=False)
            if chart:
                visuals.append(wrapped_html.block("Recovery time over time", chart))
        elif trend:
            # One period is not a trend, but its recovery time is still a real,
            # gated number worth stating.
            narrate["typical_recovery"] = wrapped_html.format_duration(
                trend[0]["recovery_seconds"])

    return {
        "title": "Repair Report",
        "stats": stats,
        "narrate_stats": narrate,
        "extra_html": "".join(visuals),
        "caption": ("Conflict windows are tense stretches a local model flagged "
                    "from deterministic signals (reply-latency withdrawal, "
                    "one-sided sessions, volume collapse) -- a read, not a "
                    "verdict. Escalation and repair labels appear only where "
                    "they cleared an agreement check, and off-the-text stretches "
                    "are left out of recovery times."),
        "hint": ("Conflict windows are deterministic tense-stretch flags, never "
                 "certainties -- describe them gently and never assign blame or "
                 "name a 'winner'. The horseman is the reader's OWN most-common "
                 "escalation move in their turns. The most-repaired relationship "
                 "and recovery times appear only when those labels passed their "
                 "gate; recovery time is how long a repaired window lasted. Keep "
                 "it warm and non-judgmental, and name at most one relationship."),
    }
# --- end feature: conflict ---


# --- feature: wellbeing ---
# W4 (RES-68): the opt-in wellbeing trajectory section. Built ONLY when
# build_sections is called with wellbeing=True (wrapped.py resolves the
# --wellbeing flag / WRAPPED_WELLBEING config), and even then it renders only
# when get_language_trajectory reports a flagged, sustained cross-family shift
# in the user's OWN language -- otherwise it returns None and the section is
# dropped. It is deliberately self-scoped (the user's history, pooled across
# chats): per-contact trajectories are consent-gated and are not surfaced in a
# shareable keepsake. Two safety invariants, both owned by CODE not the model:
#   * the fixed support footer (wellbeing.support_block) is rendered into the
#     section HTML, so it ships on every flagged window regardless of narration;
#   * every builder-authored string here passes wellbeing.sanitize_condition_names
#     before it can render, and the LLM blurb is scrubbed the same way in
#     wrapped.compose_report (via the sanitize_narrative flag below).

# Internal marker name -> a plain, non-clinical reader label. Never names a
# condition; describes the behavior a marker measures.
_WELLBEING_MARKER_LABELS = {
    "i_count": "“I” focus",
    "absolutist_count": "Absolute words",
    "fw_negation_rate": "Negations",
    "late_night_share": "Late-night messaging",
    "initiation_share": "Reaching out first",
    "laugh_rate": "Laughing",
    "emoji_rate": "Emoji",
}


def _sec_wellbeing():
    """Self trajectory with the flagged window shaded, plus the code-emitted
    support footer. Returns None (section omitted) when the tool is unavailable,
    below the data floor, or reports no sustained departure."""
    import wellbeing  # dependency-free; imported locally to keep the top clean

    result, _note = wrapped._tool("get_language_trajectory")
    if result is None:
        return None
    windows = result.get("windows") or []
    if not windows:  # no sustained cross-family shift -> nothing to shade
        return None
    window = windows[0]
    start, end = window.get("start"), window.get("end")
    if not (start and end):
        return None

    series_map = result.get("series") or {}
    departed = set(window.get("departed_families") or [])
    ordered = list(wellbeing.MARKERS)
    # Prefer the markers whose families actually departed; fall back to whatever
    # series exist so the chart is never empty when a window fired.
    chosen = [name for name in ordered
              if wellbeing.MARKERS[name][0] in departed and series_map.get(name)]
    if not chosen:
        chosen = [name for name in ordered if series_map.get(name)]
    markers = [{"label": _WELLBEING_MARKER_LABELS.get(name, name),
                "points": series_map[name]} for name in chosen]

    def clean(text):
        return wellbeing.sanitize_condition_names(text or "")[0]

    span = f"{wrapped_html.display_scalar(start)} to {wrapped_html.display_scalar(end)}"
    framing = clean(window.get("framing")
                    or "A stretch where several of your own signals moved together.")
    caption = clean(
        "This looks back at how your own texting shifted over your history; the "
        "shaded stretch is where several signals moved together at once. It is a "
        "gentle heads-up, never a diagnosis.")
    support = wellbeing.support_block()  # fixed, code-emitted, never the model's

    return {
        "title": "A Gentle Check-In",
        "stats": {},
        "narrate_stats": {
            "the_stretch": span,
            "what_this_is": framing,
            "signals_that_shifted": [m["label"] for m in markers],
        },
        "extra_html": wrapped_html.render_wellbeing(markers, (start, end), support),
        "caption": caption,
        "sanitize_narrative": True,  # wrapped.compose_report scrubs the LLM blurb
        "hint": clean(
            "This is a sensitive, opt-in section about the reader's own texting "
            "over time. Write two or three warm, plain second-person sentences. "
            "Name the stretch (the_stretch) and that a few signals shifted "
            "together -- never diagnose, never name any condition, never give "
            "medical advice, and stay gentle and non-alarming. Support "
            "information is shown separately below your blurb."),
    }
# --- end feature: wellbeing ---


# --- feature: nudges ---
# Human-facing labels for each loose-thread kind, so the board reads as a
# keepsake and never leaks a detector's snake_case name. Unknown kinds fall
# back to a de-underscored form.
_OPEN_LOOP_LABELS = {
    "faded_initiator": "they used to reach out first",
    "dormant_tie": "gone quiet",
    "unanswered_question": "a question left hanging",
    "inferred_occasion": "a date coming back around",
    "dropped_plan": "a plan that fizzled",
    "unreciprocated_goodnews": "good news that went unanswered",
}


def _sec_open_loops(exclude_chat_ids=(), excluded_names=frozenset()):
    """Open Loops board: the current loose-thread nudge digest rendered as a
    keepsake. Composes from get_loose_threads (already post-Stage-0, so muted
    nudge_exclusions contacts never appear) and ADDITIONALLY drops any contact
    Wrapped itself excluded -- the silent-contact roster guard or --exclude,
    expanded to their 1:1 chats in wrapped.main, plus the lowercase-name
    fallback for unsaved handles -- so an excluded person can never resurface
    here. The dormant-tie batch-hide notice and the digest_meta completeness
    sentinel (non-thread markers) are skipped."""
    threads, note = wrapped._tool("get_loose_threads", status="open",
                                  limit=config.NUDGE_MAX_RESULTS)
    if threads is None:
        return {"title": "Open Loops", "stats": None, "note": note}
    exclude_chat_ids = set(exclude_chat_ids)
    rows = []
    for t in threads:
        if not isinstance(t, dict) or "notice" in t or "digest_meta" in t:
            continue  # skip the dormant-tie batch-hide notice AND the
            # get_loose_threads digest_meta completeness sentinel -- neither is a
            # board item (both can trail the real threads in the same response)
        if t.get("chat_id") in exclude_chat_ids:
            continue
        name = t.get("contact")
        if name and str(name).lower() in excluded_names:
            continue
        rows.append({
            "who": name or "Someone",
            "kind": _OPEN_LOOP_LABELS.get(
                t.get("kind"), str(t.get("kind") or "").replace("_", " ")),
            "nudge": t.get("nudge_text"),
            "since": t.get("anchor_date"),
            "due": t.get("due_by"),
        })
    if not rows:
        return {"title": "Open Loops", "stats": None,
                "note": "no open loops in scope"}
    return {
        "title": "Open Loops",
        "stats": {"open_loops": len(rows)},
        "narrate_stats": {"open_loops": rows},
        "extra_html": wrapped_html.render_open_loops(rows),
        "caption": ("Threads you let drift -- someone who went quiet, a question "
                    "left hanging, a birthday coming back around. Muted contacts "
                    "and anyone kept out of this report never appear here."),
        "hint": ("Each item is one open loop: the person, why it's open, and for "
                 "a recurring occasion when it next comes up. Nudge gently toward "
                 "one or two; never scold, and don't list them all."),
    }
# --- end feature: nudges ---
