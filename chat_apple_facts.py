"""Fact derivers for the Apple FM chat path: turn one structured tool result
into clean plain-text sentences the on-device model can narrate without doing
arithmetic or reading a nested aggregate.

Split out of chat_apple.py. A NON-EMPTY return also ends that path's staged
loop -- the exact answer is in hand, and the raw result's digest is noise the
small model was observed to confabulate from. Every specialized analytics tool
routes through a deriver here so the reply narrates a code-computed fact rather
than inventing one.

Every name is re-exported from chat_apple, so `chat_apple._derive_summary_facts`
(etc.) keeps resolving for the tests.
"""

import re

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday")


_FIRST_MSG_RE = re.compile(
    r"first (text|message|messaged|texted)|who .*(texted|messaged|reached out) .*first"
    r"|when did .* first", re.I)


def _derive_summary_facts(name, result, question=""):
    """Pre-compute the arithmetic/reading a small model cannot do reliably over a
    structured aggregate, returning clean plain-text facts to seed alongside the
    digest. A NON-EMPTY return also ends the loop -- the exact answer is in hand,
    and the raw result's digest is noise the small model was observed (on the
    Apple FM eval sweep) to confabulate from. Every specialized
    analytics tool routes through a deriver here so the reply narrates a
    code-computed fact rather than inventing one; anything not special-cased (or a
    deriver that cannot read its result) returns [] and falls back to DIGEST."""
    q = question or ""
    if isinstance(result, list):
        # List-shaped results (loose threads, participant stats, and the
        # group_by_chat=True per-chat leaderboard of get_chat_summary_stats).
        if name == "get_loose_threads":
            return _derive_loose_threads_facts(result)
        if name == "get_chat_participant_stats":
            return _derive_participant_facts(result, q)
        if name == "get_chat_summary_stats":
            return _derive_summary_stats_facts(result, q)
        return []
    if not isinstance(result, dict):
        return []
    if result.get("error") or result.get("duplicate_call"):
        return []
    dispatch = {
        "get_activity_patterns": _derive_activity_facts,
        "get_response_dynamics": _derive_response_facts,
        "get_milestones": _derive_milestone_facts,
        "get_reaction_stats": _derive_reaction_facts,
        "get_word_trends": _derive_word_trends_facts,
        "get_word_frequency": _derive_word_frequency_facts,
        "get_group_dynamics": _derive_group_facts,
        "get_voiceprint": _derive_voiceprint_facts,
        "get_chat_summary_stats": _derive_summary_stats_facts,
        "get_fadeout_watchlist": _derive_fadeout_facts,
        "find_conflicts": _derive_conflict_facts,
    }
    fn = dispatch.get(name)
    if fn is None:
        return []
    try:
        return fn(result, q)
    except Exception:
        return []  # a shape surprise must degrade to DIGEST, never crash the turn


def _msg_snippet(m):
    """A one-line 'sender: "text" (day)' from a milestone/message dict."""
    if not isinstance(m, dict):
        return ""
    sender = m.get("sender") or ("You" if m.get("is_from_me") else "the other person")
    text = _clip((m.get("text") or "").strip(), 90)
    day = str(m.get("date") or "")[:10]
    return f'{sender}: "{text}" ({day})' if text else f"{sender} ({day})"


def _derive_milestone_facts(result, question=""):
    """get_milestones has four sub-sections; pick the one the question asks for so
    a 'when did we first / who first / 10,000th message / longest silence / how
    far back' answer quotes the right landmark instead of the model conflating
    two near-identical timestamps (RC-4)."""
    q = question.lower()
    firsts = result.get("firsts") if isinstance(result.get("firsts"), dict) else {}
    facts = []
    # Each specific-intent branch commits: it returns its fact, or [] when the
    # section is absent -- never the first/who-first default, which would answer a
    # different question (a 10,000th-message ask must not get the first-text date).
    # Longest silence / gone without talking.
    if re.search(r"\b(longest|silence|gap|without (talking|texting)|gone without)\b", q):
        sil = result.get("longest_silence")
        if isinstance(sil, dict) and sil.get("available") and sil.get("days") is not None:
            return [f"[derived] The longest silence was {sil['days']} days "
                    f"({str(sil.get('from'))[:10]} to {str(sil.get('to'))[:10]}), "
                    f"broken by {sil.get('broken_by', 'someone')}."]
        return []
    # Nth-message landmark (10,000th / 1,000th ...).
    if re.search(r"\b\d[\d,]*(st|nd|rd|th)\b", q):
        nth = result.get("nth")
        rows = nth.get("landmarks") if isinstance(nth, dict) else None
        if isinstance(rows, list) and rows:
            asked = None
            mnum = re.search(r"(\d[\d,]*)(?:st|nd|rd|th)", q)
            if mnum:
                asked = int(mnum.group(1).replace(",", ""))
            pick = None
            for row in rows:  # exact match if present, else the largest landmark
                if asked is not None and row.get("ordinal") == asked:
                    pick = row
                    break
            pick = pick or rows[-1]
            return [f"[derived] The {pick.get('ordinal')}th message was "
                    f"{_msg_snippet(pick.get('message'))}."]
        return []
    # Anniversaries.
    if "anniversar" in q:
        ann = result.get("anniversaries")
        if isinstance(ann, dict) and ann.get("next_anniversary"):
            return [f"[derived] Next anniversary: {ann['next_anniversary']} "
                    f"({ann.get('years', '?')} years since first contact, "
                    f"{ann.get('days_until', '?')} days away)."]
        return []
    # How far back / earliest.
    if re.search(r"\bhow far back|earliest|history go\b", q):
        first = firsts.get("first")
        if isinstance(first, dict) and first.get("date"):
            return [f"[derived] Message history goes back to {str(first['date'])[:10]} "
                    f"({_msg_snippet(first)})."]
        return []
    # Last message: ONLY firsts.last (the overall most recent message). Milestones
    # has no last-inbound field, and any fallback to a "first_*" receipt would
    # present the oldest message as the newest -- degrade to DIGEST instead.
    if re.search(r"\blast (thing|message)\b", q):
        last = firsts.get("last")
        if not (isinstance(last, dict) and last.get("date")):
            return []
        # "Last thing THEY sent": if the most recent message is mine, the
        # their-side answer is not in this result -- do not present my own
        # message as theirs.
        if last.get("is_from_me") and re.search(r"\b(they|she|he|them)\b", q):
            return []
        return [f"[derived] The last message was {_msg_snippet(last)}."]
    # Default: first / who-first.
    first = firsts.get("first") if isinstance(firsts.get("first"), dict) else {}
    fm = firsts.get("first_from_me") if isinstance(firsts.get("first_from_me"), dict) else {}
    ft = firsts.get("first_from_them") if isinstance(firsts.get("first_from_them"), dict) \
        else {}
    first_date = first.get("date")
    fm_date, ft_date = fm.get("date"), ft.get("date")
    if not first_date:
        return []
    # Keep the who-first verdict a single unambiguous clause -- including the two
    # near-identical raw timestamps confused the small model into inverting it.
    day = str(first_date)[:10]
    facts = [f'[derived] The first message between you and the other person was on {day}'
             f' ({_msg_snippet(first)}).']
    if fm_date and ft_date:  # ISO strings compare lexicographically = chronologically
        who = "You" if fm_date <= ft_date else "The other person (not you)"
        facts.append(f"[derived] {who} sent that very first message.")
    return facts


def _derive_activity_facts(result, question=""):
    """get_activity_patterns carries the heatmap plus late_night / streaks /
    busiest_days / volume_series; pick the section the question asks for."""
    q = question.lower()
    tz = result.get("timezone", "local")
    # Late-night / 3am: name the share and the top late-night contact (RC-3 fix
    # for the 'you text Chungus 10 times' confabulation).
    if re.search(r"\b(after midnight|3\s?am|late at night|at night)\b", q):
        ln = result.get("late_night")
        if isinstance(ln, dict) and ln.get("n") is not None:
            # Skip business/opaque handles (urn:...) so the "who" is a real person.
            top = [c for c in (ln.get("top_contacts") or [])
                   if isinstance(c, dict) and not str(c.get("contact", "")).startswith("urn:")]
            who = ""
            if top:
                names = ", ".join(f"{c.get('contact')} ({c.get('late_night')} late-night "
                                  f"messages)" for c in top[:3])
                who = (f" The people on the other end, most first: {names}.")
            return [f"[derived] You send {ln['n']} messages between "
                    f"{ln.get('start_hour', 0)}:00 and {ln.get('end_hour', 5)}:00 "
                    f"({round((ln.get('share') or 0) * 100, 1)}% of all your messages)."
                    + who]
    # Longest streak of consecutive days.
    if "streak" in q:
        st = result.get("streaks")
        ls = st.get("longest_streak") if isinstance(st, dict) else None
        if isinstance(ls, dict) and ls.get("days") is not None:
            return [f"[derived] The longest streak of consecutive days talking was "
                    f"{ls['days']} days ({ls.get('start')} to {ls.get('end')})."]
    # Single busiest day ever.
    if re.search(r"\bbusiest .*(day|date) (ever|of all)|single busiest\b", q):
        bd = result.get("busiest_days")
        if isinstance(bd, list) and bd and isinstance(bd[0], dict):
            top = bd[0]
            return [f"[derived] Your single busiest day was {top.get('day')} "
                    f"with {top.get('count')} messages."]
    # More/less this year vs last (year volume compare).
    if re.search(r"\b(more|less)\b.*\b(year|month)\b", q):
        vs = result.get("volume_series")
        rows = vs.get("rows") if isinstance(vs, dict) else None
        if isinstance(rows, list) and len(rows) >= 2:
            def tot(r):
                return (r.get("sent") or 0) + (r.get("received") or 0)
            last, prev = rows[-1], rows[-2]
            direction = ("more" if tot(last) > tot(prev)
                         else "less" if tot(last) < tot(prev) else "about the same")
            return [f"[derived] {last.get('period')}: {tot(last)} messages vs "
                    f"{prev.get('period')}: {tot(prev)} -> you are texting {direction}."]
    # Per-period volume series ("how many each month of 2025").
    if re.search(r"\beach (month|year|week)|per (month|year|week)|how many .*each\b", q):
        vs = result.get("volume_series")
        rows = vs.get("rows") if isinstance(vs, dict) else None
        if isinstance(rows, list) and rows:
            grand = sum((r.get("sent") or 0) + (r.get("received") or 0) for r in rows)
            per = "; ".join(
                f"{r.get('period')}: {r.get('sent')} sent / {r.get('received')} received"
                for r in rows if isinstance(r, dict))
            return [f"[derived] Volume by {vs.get('granularity', 'period')} "
                    f"({grand} total): {per}."]
    # Default: busiest hour + weekday from the 7x24 heatmap.
    heatmap = result.get("heatmap")
    counts = heatmap.get("counts") if isinstance(heatmap, dict) else None
    if not (isinstance(counts, list) and len(counts) == 7
            and all(isinstance(row, list) and len(row) == 24 for row in counts)):
        return []
    hour_totals = [sum(counts[d][h] for d in range(7)) for h in range(24)]
    day_totals = [sum(counts[d]) for d in range(7)]
    if not any(hour_totals):
        return []
    busiest_hour = max(range(24), key=lambda h: hour_totals[h])
    busiest_day = max(range(7), key=lambda d: day_totals[d])
    return [
        f"[derived] busiest hour of day = {busiest_hour:02d}:00 {tz} "
        f"({hour_totals[busiest_hour]} messages); busiest weekday = "
        f"{_WEEKDAY_NAMES[busiest_day]} ({day_totals[busiest_day]} messages)"
    ]


# Reply-speed cues that justify the get_response_dynamics reply-latency verdict (who
# replies faster / response time / quickest). Gates the default so a non-speed
# question routed here does not get a spurious latency fact.
_REPLY_SPEED_RE = re.compile(
    r"\b(fast(er|est)?|quick(er|est)?|slow(er|est)?|repl(y|ies|ied)|respond|responds"
    r"|response|get(s)? back|text(s)? back|first)\b", re.I)

# Read-receipt coverage below this share carries no usable "left on read" signal
# (the caveat is the honest answer); at/above it, the measured read->reply
# figures are worth stating -- still never a left-on-read COUNT, because the
# tool records read-and-replied pairs only.
_READ_COVERAGE_MIN = 0.5


def _derive_response_facts(result, question=""):
    q = question.lower()
    # Initiation: who starts more.
    if re.search(r"\b(start|starts|initiat)", q):
        init = result.get("initiation")
        if isinstance(init, dict) and isinstance(init.get("me"), (int, float)) \
                and isinstance(init.get("them"), (int, float)):
            who = ("You" if init["me"] > init["them"]
                   else "The other person" if init["them"] > init["me"] else "Both equally")
            return [f"[derived] Conversation starts: you initiated {init['me']}, the "
                    f"other person {init['them']} -> {who} start(s) more."]
    # Double-texting (a follow-up you send before the other person replies). Routed
    # here by the \bdouble.?text rule -- without this branch it fell to the reply-
    # latency default and answered a different question.
    if re.search(r"\bdouble.?text", q):
        dt = result.get("double_text")
        if isinstance(dt, dict) and isinstance(dt.get("me"), (int, float)) \
                and isinstance(dt.get("them"), (int, float)):
            who = ("You double-text more" if dt["me"] > dt["them"]
                   else "The other person double-texts more" if dt["them"] > dt["me"]
                   else "You both double-text about equally")
            return [f"[derived] Double-texts (a follow-up sent before a reply): you "
                    f"{dt['me']}, the other person {dt['them']} -> {who}."]
    # Left on read. The tool has no read-but-never-replied count -- only my
    # read->reply medians plus per-side read-receipt coverage. Pick the side the
    # question asks about: "leaves ME on read" is about THEIR receipts of my
    # outbound messages; anything else is about my receipts of their inbound.
    if re.search(r"\bon read\b", q):
        r2r = result.get("read_to_reply")
        if isinstance(r2r, dict):
            cov = r2r.get("coverage") or {}
            about_me = re.search(r"\b(leave|left|leaves)\s+me\b", q)
            if about_me:
                got = cov.get("outbound_with_date_read") or 0
                tot = cov.get("outbound_total") or 0
                side = "your messages to them"
            else:
                got = cov.get("inbound_with_date_read") or 0
                tot = cov.get("inbound_total") or 0
                side = "their messages to you"
            pct = round(got / tot * 100, 1) if tot else 0
            if not tot or got / tot < _READ_COVERAGE_MIN:
                return [f"[derived] Read receipts cover only {pct}% of {side} "
                        f"({got}/{tot}), so 'left on read' cannot be measured "
                        f"reliably -- state this caveat rather than a verdict."]
            fact = f"[derived] Read receipts cover {pct}% of {side} ({got}/{tot})"
            med = r2r.get("median_deliberation_seconds")
            # The read->reply medians measure MY replies to messages I read, so
            # they only bear on the "do I leave them on read" side.
            if not about_me and isinstance(med, (int, float)):
                fact += (f"; you reply a median of {med}s after reading "
                         f"({r2r.get('pairs', 0)} read->reply pairs)")
            return [fact + ". Only read-and-replied pairs are recorded -- a "
                    "read-but-never-answered count is not available, so report "
                    "what is measured, not a left-on-read verdict."]
    # Long conversations vs quick bursts.
    if re.search(r"\b(long conversations|quick bursts|bursts)\b", q):
        sess = result.get("sessions")
        if isinstance(sess, dict) and sess.get("median_messages") is not None:
            med = sess.get("median_messages") or 0
            shape = "longer conversations" if med >= 8 else "quick bursts"
            return [f"[derived] {sess.get('count')} conversation sessions, median "
                    f"{med} messages each -> {shape}."]
    # Ghosted / never replied. only_me lives under the "ghosted" section (a count of
    # one-sided sessions), NOT "sessions" (which has no such field).
    if re.search(r"\b(ghost|not repl|never repl)", q):
        ghost = result.get("ghosted")
        only_me = ghost.get("only_me") if isinstance(ghost, dict) else None
        if only_me is not None:
            sess = result.get("sessions") or {}
            return [f"[derived] {only_me} conversation(s) you started got no reply at "
                    f"all (out of {sess.get('count', '?')} sessions)."]
    # Default: reply-latency comparison.
    lat = result.get("reply_latency")
    if not isinstance(lat, dict):
        return []
    me, them = lat.get("me"), lat.get("them")
    if not (isinstance(me, dict) and isinstance(them, dict)):
        return []
    me_med, them_med = me.get("median_seconds"), them.get("median_seconds")
    if not (isinstance(me_med, (int, float)) and isinstance(them_med, (int, float))):
        return []
    # "How long does <them> take to reply" asks for THEIR median specifically --
    # lead with it so the small model does not quote your side by mistake (S2.2).
    if re.search(r"\bhow (long|fast|quick)\b", q) and not re.search(r"\bwho\b", q):
        them_p90 = them.get("p90_seconds")
        p90 = f", 90th percentile {them_p90}s" if isinstance(them_p90, (int, float)) else ""
        # State ONLY their median -- naming your own number here made the small
        # model quote the wrong side (S2.2). The answer is a single figure.
        return [f"[derived] The other person takes a median of {them_med}s to reply "
                f"to you{p90}."]
    # The reply-latency verdict must answer only a reply-SPEED question -- a question
    # that reached this tool for another reason (double-text, a future intent) would
    # otherwise get a bogus "who replies faster" fact. Gate it, but only when a
    # question is present (a no-question call keeps the default -- production always
    # supplies one, so this affects tests/edge callers only).
    if q and not _REPLY_SPEED_RE.search(q):
        return []
    if me_med < them_med:
        verdict = "you reply faster (lower median reply time)"
    elif them_med < me_med:
        verdict = "the other person replies faster (lower median reply time)"
    else:
        verdict = "you both reply about equally fast"
    return [
        f"[derived] median reply time: you = {me_med}s, the other person = "
        f"{them_med}s -> {verdict}"
    ]


# --- Tapback / reaction labels for friendly narration -----------------------
_REACT_LABEL = {
    "love": "Loved (heart)", "like": "Liked (thumbs up)", "dislike": "Disliked",
    "laugh": "Laughed (haha)", "emphasize": "Emphasized (!!)",
    "question": "Questioned (?)", "custom-emoji": "custom emoji", "sticker": "sticker",
}


def _top_by_type(by_type):
    """(label, count) of the highest-count reaction type, or None."""
    if not isinstance(by_type, dict) or not by_type:
        return None
    top = max(by_type.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)
    return _REACT_LABEL.get(top[0], top[0]), top[1]


def _derive_reaction_facts(result, question=""):
    q = question.lower()
    # Funniest leaderboard.
    if "funn" in q:
        fun = result.get("funniest")
        if isinstance(fun, list) and fun and isinstance(fun[0], dict):
            f = fun[0]
            return [f"[derived] {f.get('participant')} laughs at your messages most: "
                    f"{f.get('laughs')} laughs ({f.get('laughs_per_100_my_messages')} "
                    f"per 100 of your messages)."]
    # Most-reacted single message (hall of fame).
    if re.search(r"\bmost reactions|which .*message\b", q):
        hof = result.get("hall_of_fame")
        if isinstance(hof, list) and hof and isinstance(hof[0], dict):
            h = hof[0]
            return [f"[derived] Your most-reacted message: {_msg_snippet(h)} — "
                    f"{h.get('reaction_count')} reactions."]
    # How quickly people react.
    if re.search(r"\bhow (quickly|fast)|react.*quick", q):
        ttr = result.get("time_to_react")
        them = ttr.get("them_reacting") if isinstance(ttr, dict) else None
        if isinstance(them, dict) and them.get("median_seconds") is not None:
            return [f"[derived] People react to your messages in a median of "
                    f"{them['median_seconds']} seconds."]
    # Loved my messages (received love count).
    if re.search(r"\b(love|loved|heart)\b", q):
        rec = result.get("received")
        love = (rec.get("by_type") or {}).get("love") if isinstance(rec, dict) else None
        if love is not None:
            return [f"[derived] You received {love} 'Loved' reactions on your messages "
                    f"in this scope."]
    # React more than / reciprocity.
    if re.search(r"\bmore than|reciprocit|react.*more\b", q):
        rec = result.get("reciprocity")
        if isinstance(rec, dict) and rec.get("ratio") is not None:
            return [f"[derived] You gave {rec.get('given')} reactions and received "
                    f"{rec.get('received')} (ratio {rec.get('ratio')}): you react "
                    f"{'more' if (rec.get('ratio') or 0) > 1 else 'less'} than they do."]
    # Different reactions between us (given vs received breakdown).
    if re.search(r"\bdifferent|what reactions do\b", q):
        given = _top_by_type((result.get("given") or {}).get("by_type"))
        recv = _top_by_type((result.get("received") or {}).get("by_type"))
        if given or recv:
            parts = []
            if given:
                parts.append(f"you most often send '{given[0]}' ({given[1]})")
            if recv:
                parts.append(f"you most often receive '{recv[0]}' ({recv[1]})")
            return [f"[derived] Reaction use: {'; '.join(parts)}."]
    # Default: top tapback given.
    given = _top_by_type((result.get("given") or {}).get("by_type"))
    if given:
        return [f"[derived] Your most-used tapback is '{given[0]}' ({given[1]} times)."]
    return []


def _derive_word_trends_facts(result, question=""):
    facts = []
    pt = result.get("phrase_trend")
    if isinstance(pt, dict) and isinstance(pt.get("series"), list) and pt["series"]:
        series = pt["series"]

        def tot(r):
            return (r.get("me") or 0) + (r.get("them") or 0)
        first, last = series[0], series[-1]
        direction = ("rising" if tot(last) > tot(first)
                     else "falling" if tot(last) < tot(first) else "flat")
        facts.append(
            f"[derived] '{pt.get('phrase')}' appears in {pt.get('total_messages')} "
            f"messages; {first.get('period')}={tot(first)} -> {last.get('period')}="
            f"{tot(last)} ({direction}). First used by "
            f"{pt.get('coined_by', 'unknown')} on {str(pt.get('date') or '')[:10]}."
            if pt.get("coined_by") else
            f"[derived] '{pt.get('phrase')}' appears in {pt.get('total_messages')} "
            f"messages; {first.get('period')}={tot(first)} -> {last.get('period')}="
            f"{tot(last)} ({direction}).")
    emoji = result.get("emoji")
    if isinstance(emoji, dict) and isinstance(emoji.get("top"), list) and emoji["top"]:
        top = ", ".join(f"{e.get('emoji')} ({e.get('count')})"
                        for e in emoji["top"][:3] if isinstance(e, dict))
        facts.append(f"[derived] Top emoji: {top}.")
    dt = result.get("distinctive_terms")
    if isinstance(dt, dict) and isinstance(dt.get("terms"), list) and dt["terms"]:
        terms = "; ".join(
            f"{t.get('term')} ({t.get('verdict')})"
            + ("" if t.get("curated") else " [uncurated]")
            for t in dt["terms"][:5] if isinstance(t, dict))
        facts.append(f"[derived] Distinctive shared terms: {terms}.")
    lsm = result.get("lsm")
    if isinstance(lsm, dict) and lsm.get("score") is not None:
        s = lsm["score"]
        reading = ("very similar" if s >= 0.9 else "fairly similar" if s >= 0.8
                   else "moderately similar" if s >= 0.6 else "not very similar")
        facts.append(f"[derived] Language style matching score = {s} ({reading}); "
                     f"1.0 is identical style.")
    emo = result.get("emotions")
    if isinstance(emo, dict) and isinstance(emo.get("top"), list) and emo["top"]:
        top = ", ".join(f"{e.get('emotion')} ({e.get('count')})"
                        for e in emo["top"][:3] if isinstance(e, dict))
        facts.append(f"[derived] Top emotions: {top}.")
    elif isinstance(emo, dict) and not emo.get("top"):
        facts.append("[derived] Emotion analysis is not available for this scope.")
    val = result.get("valence")
    if isinstance(val, dict) and isinstance(val.get("series"), list) and val["series"]:
        s = val["series"]
        facts.append(f"[derived] Mood (mean valence) {s[0].get('period')}="
                     f"{s[0].get('mean_valence')} -> {s[-1].get('period')}="
                     f"{s[-1].get('mean_valence')}.")
    elif isinstance(val, dict) and val.get("available") is False:
        facts.append("[derived] Mood/valence analysis is not available for this scope.")
    rates = result.get("rates")
    if isinstance(rates, dict) and isinstance(rates.get("series"), list) and rates["series"]:
        s = rates["series"]
        first_me = (s[0].get("me") or {}) if isinstance(s[0], dict) else {}
        last_me = (s[-1].get("me") or {}) if isinstance(s[-1], dict) else {}
        facts.append(f"[derived] Your laughs per 100 messages: {s[0].get('period')}="
                     f"{first_me.get('laugh_per_100')} -> {s[-1].get('period')}="
                     f"{last_me.get('laugh_per_100')}.")
    return facts


def _derive_word_frequency_facts(result, question=""):
    rows = result.get("rows")
    if not (isinstance(rows, list) and rows):
        return []
    top = ", ".join(f"{r.get('ngram')} ({r.get('count')})"
                    for r in rows[:10] if isinstance(r, dict))
    kind = "phrases" if (result.get("ngram") or 1) >= 2 else "words"
    return [f"[derived] Most common {kind} (scanned {result.get('messages_scanned', '?')} "
            f"messages): {top}."]


def _derive_group_facts(result, question=""):
    q = question.lower()
    if re.search(r"\bcarr|dominates|dominated by|balanced|most active\b", q):
        members = result.get("members")
        if isinstance(members, list) and members and isinstance(members[0], dict):
            m = members[0]
            gini = (result.get("gini") or {}).get("gini")
            lead = (f"{m.get('member')} carries the group with "
                    f"{round((m.get('share') or 0) * 100, 1)}% of messages")
            if gini is not None:
                bal = "fairly balanced" if gini < 0.3 else "dominated by a few"
                lead += f"; Gini {gini} ({bal})"
            return [f"[derived] {lead}."]
    if "lurk" in q:
        lurkers = result.get("lurkers")
        if isinstance(lurkers, list):
            names = ", ".join(str(x.get("member") if isinstance(x, dict) else x)
                              for x in lurkers) if lurkers else "none"
            return [f"[derived] Lurkers (members who mostly read, rarely post): {names}."]
    if re.search(r"\b(starts|start|igniter|initiat)", q):
        ig = result.get("igniters")
        lb = ig.get("leaderboard") if isinstance(ig, dict) else None
        if isinstance(lb, list) and lb and isinstance(lb[0], dict):
            top = lb[0]
            return [f"[derived] {top.get('member')} starts the most conversations "
                    f"({round((top.get('share') or 0) * 100, 1)}%, "
                    f"{top.get('sessions_started')} of {ig.get('n_sessions')} sessions)."]
    if re.search(r"\btalks? to whom|who .*talks?\b", q):
        adj = result.get("adjacency")
        pairs = adj.get("pairs") if isinstance(adj, dict) else None
        if isinstance(pairs, list) and pairs and isinstance(pairs[0], dict):
            p = pairs[0]
            return [f"[derived] The most active pair is {p.get('replier')} <-> "
                    f"{p.get('replied_to')} ({p.get('count')} exchanges)."]
    if re.search(r"\bstands? out\b", q):
        gl = result.get("group_lsm")
        members = gl.get("members") if isinstance(gl, dict) else None
        if isinstance(members, list) and members:
            odd = min(members, key=lambda m: m.get("lsm_score", 1)
                      if isinstance(m, dict) else 1)
            return [f"[derived] {odd.get('member')}'s style stands out most from the "
                    f"group (lowest LSM {odd.get('lsm_score')})."]
    # Default: top member by share.
    members = result.get("members")
    if isinstance(members, list) and members and isinstance(members[0], dict):
        m = members[0]
        return [f"[derived] Most active member: {m.get('member')} "
                f"({round((m.get('share') or 0) * 100, 1)}% of messages)."]
    return []


def _derive_participant_facts(result, question=""):
    """get_chat_participant_stats: a per-member list. Answer the 'who gets the most
    reactions per message, among members with >= N sent' question by applying the
    floor the prompt states, then ranking reactions_received / messages_sent."""
    if not (isinstance(result, list) and result):
        return []
    floor = 0
    m = re.search(r"at least (\d+)|(\d+)\+? messages", question.lower())
    if m:
        floor = int(m.group(1) or m.group(2))
    eligible = [r for r in result if isinstance(r, dict)
                and (r.get("messages_sent") or 0) >= floor]
    if not eligible:
        return [f"[derived] No members sent at least {floor} messages."]

    def rate(r):
        sent = r.get("messages_sent") or 0
        return (r.get("reactions_received") or 0) / sent if sent else 0
    top = max(eligible, key=rate)
    return [f"[derived] Among members with >= {floor} messages, {top.get('participant')} "
            f"gets the most reactions per message: {top.get('reactions_received')} "
            f"reactions / {top.get('messages_sent')} sent = {round(rate(top), 3)}."]


def _derive_voiceprint_facts(result, question=""):
    feats = result.get("features")
    if not isinstance(feats, dict):
        return []
    conf = result.get("confidence", "unknown")
    n = result.get("n_messages", "?")
    bits = []
    lc = feats.get("lowercase_ratio")
    if isinstance(lc, (int, float)):
        bits.append(f"{'mostly lowercase' if lc >= 0.6 else 'mixed case'} "
                    f"({round(lc * 100)}% lowercase)")
    burst = feats.get("burst_size")
    if isinstance(burst, (int, float)):
        bits.append(f"{'tends to double-text' if burst >= 1.8 else 'one message at a time'} "
                    f"(avg burst {round(burst, 1)})")
    ex = feats.get("exclaim_rate")
    if isinstance(ex, (int, float)) and ex:
        bits.append(f"exclamation rate {round(ex, 3)}")

    def phrases(key):
        items = result.get(key)
        if isinstance(items, list) and items:
            return ", ".join(f"'{p.get('phrase')}'" for p in items[:3]
                             if isinstance(p, dict) and p.get("phrase"))
        return ""
    openers, closers = phrases("openers"), phrases("closers")
    if openers:
        bits.append(f"common openers {openers}")
    if closers:
        bits.append(f"common closers {closers}")
    caveat = (" This rests on few messages (LOW confidence) -- describe it tentatively."
              if conf == "low" else "")
    return [f"[derived] Texting voice ({conf} confidence, {n} messages): "
            f"{'; '.join(bits)}.{caveat}"]


def _derive_summary_stats_facts(result, question=""):
    q = question.lower()
    # group_by_chat leaderboard (list): the top thread by volume.
    if isinstance(result, list):
        rows = [r for r in result if isinstance(r, dict)]
        if not rows:
            return []
        top = max(rows, key=lambda r: r.get("message_count") or 0)
        # A derived fact ends the loop, so no follow-up call can resolve the id
        # to a name -- present the id honestly instead of asking the model to.
        return [f"[derived] Your highest-volume thread is chat_id={top.get('chat_id')} "
                f"with {top.get('message_count')} messages (a chat id -- the thread's "
                f"name is not in this result)."]
    if not isinstance(result, dict):
        return []
    if re.search(r"\b(photo|picture|image|attachment)", q):
        ac = result.get("attachment_count")
        if ac is not None:
            return [f"[derived] {ac} attachments have been sent in this scope "
                    f"(individual attachments, not messages)."]
    mc = result.get("message_count")
    if mc is not None:
        return [f"[derived] {mc} messages in this scope "
                f"({result.get('from_me', '?')} from you, "
                f"{result.get('from_them', '?')} from them)."]
    return []


def _derive_fadeout_facts(result, question=""):
    wl = result.get("watchlist")
    if not (isinstance(wl, list) and wl):
        return []
    items = []
    for r in wl[:6]:
        if not isinstance(r, dict):
            continue
        driver = (r.get("top_driver") or {}).get("signal", "?")
        items.append(f"{r.get('contact')} ({r.get('phase')}, driver: {driver})")
    return [f"[derived] Drifting/fading ties (iMessage-only signal, not a verdict): "
            f"{'; '.join(items)}."]


def _derive_loose_threads_facts(result, question=""):
    if not (isinstance(result, list) and result):
        return ["[derived] No open loose threads / nudges were found."]
    items = []
    for r in result[:8]:
        if not isinstance(r, dict):
            continue
        kind = r.get("kind", "?")
        txt = _clip((r.get("nudge_text") or "").strip(), 70)
        items.append(f"[{kind}] {txt}" if txt else f"[{kind}]")
    return [f"[derived] {len(result)} open loose thread(s): " + " | ".join(items)]


# Defer (no derive/stop) ONLY for an apology/draft ask -- draft_apology needs a
# conflict_id the model must carry forward from find_conflicts, so stopping on a
# derived summary would prevent the draft. A diagnosis ask ("break down the tensest
# stretch") does NOT defer: the model does not reliably chain to diagnose_conflict,
# so a grounded find_conflicts summary (dates/signals, no-fault) beats letting the
# digest confabulate an episode.
_CONFLICT_DEFER_RE = re.compile(r"\bapolog|\b(draft|write)\b", re.I)


def _derive_conflict_facts(result, question=""):
    if _CONFLICT_DEFER_RE.search(question or ""):
        return []
    total = result.get("total_matching")
    rows = result.get("rows")
    if total is None or not isinstance(rows, list):
        return []
    if total == 0 or not rows:
        return ["[derived] No tense stretches / conflicts were detected in this thread."]
    top = rows[0] if isinstance(rows[0], dict) else {}
    signals = top.get("signals")
    sig = ", ".join(map(str, signals)) if isinstance(signals, list) else "?"
    # NOTE: deliberately no who-started / aggressor field exists -- do not infer one.
    return [f"[derived] {total} tense stretch(es) detected (deterministic text-only "
            f"signals, not confirmed fights). The strongest: around "
            f"{str(top.get('first_date') or '')[:10]}, signals: {sig}, corroboration: "
            f"{top.get('corroboration')}. These signals do NOT identify who was at "
            f"fault or who started it."]


# A who-started / whose-fault / who's-to-blame question about a conflict. The
# finder has NO aggressor field, so any fault verdict is a confabulation the
# no-aggressor design (Master Spec C2) forbids -- yet the small model reliably
# states one anyway when asked directly. So on this question the answer is
# COMPOSED IN CODE from the derived facts, never generated, mirroring how the DV
# floor and humility disclaimer are code-enforced rather than prompted.
_FAULT_QUESTION_RE = re.compile(
    r"\bwho (started|caused|instigated|is to blame|was at fault|escalated)\b"
    r"|\bwhose fault\b|\bwho'?s to blame\b|\bwho (was|is) the aggressor\b", re.I)


def _compose_no_fault_answer(scratchpad):
    """A code-composed reply for a who-started/fault conflict question: it relays
    the deterministic count/signals but never assigns blame. None when no conflict
    fact was gathered (then the normal answer path runs)."""
    conflict = next((f for f in scratchpad if "tense stretch" in f.lower()), None)
    if conflict is None:
        return None
    # No conflict was found: nothing to assign a start/fault to. Compose a clean
    # reply rather than splicing the negative fact into "what the signals show".
    if "no tense stretch" in conflict.lower():
        return (
            "I couldn't find any tense stretches or conflicts in this thread, so "
            "there's nothing to assign a start or fault to — and I wouldn't guess "
            "even if there were. These signals are a text-only read (reply gaps, "
            "one-sided sessions, tapbacks) that can't see calls, tone, or in-person "
            "context, so treat their absence as reassuring but not definitive.")
    body = conflict.split("]", 1)[-1].strip() if conflict.startswith("[") else conflict
    body = body.split("These signals do NOT")[0].strip().rstrip(".")
    return (
        "I can't say who started more of them, and I won't guess. The conflict "
        "signals are a text-only read (reply gaps, one-sided sessions, tapbacks) "
        "with no 'who started it' or fault information — it can't see calls, tone, "
        "or in-person context. "
        + (f"What the signals do show: {body}. " if body else "")
        + "Treat this as a prompt to reflect on the pattern, not a verdict on either "
        "person.")

# Bottom import: this module is one half of a cycle -- `chat_apple` re-exports the
# names below, so importing it from the TOP here makes chat_apple_facts unimportable
# on its own (the parent would run its own re-export against a still-empty
# chat_apple_facts). Down here every name above is already bound, so the cycle
# resolves whichever module a caller reaches first. Only used inside functions,
# so the late binding costs nothing.
from chat_apple import _clip  # noqa: E402
