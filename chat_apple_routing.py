"""Deterministic pre-resolution and intent routing for the Apple FM chat path.

Split out of chat_apple.py: the capitalized-name scan that turns "how often does
Will text me" into resolved contact/chat ids before the staged loop starts, and
the pattern table that forces the loop's FIRST tool call when a question's
intent is unambiguous. Both exist because a 4,096-token on-device model cannot
be trusted to route across ~32 tools on its own, and a wrong scope is worse than
a slow answer -- an unscoped analytics call misattributes whole-corpus stats to
one person.

Every name here is re-exported from chat_apple, so
`chat_apple._preresolve_entities` / `._ROUTE_RULES` (etc.) keep resolving for
the tests. _TOOL_SPECS is read back through chat_apple rather than imported by
value, the way the other split modules in this repo do it.
"""

import re

import chat_apple

# --- Deterministic name->id pre-resolution (before the staged loop) ----------
# Question words / imperative verbs that a capitalized-token scan picks up but
# that must never be looked up as a contact.
_NAME_STOPWORDS = frozenset({
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "did", "do", "does", "is", "are", "was", "were", "has", "have", "had",
    "draft", "find", "break", "based", "help", "looking", "if", "the", "a",
    "an", "and", "or", "but", "my", "me", "i", "you", "we", "us", "show",
    "tell", "give", "list", "count", "compare",
    # Sentence-initial verbs / auxiliaries / imperatives a capitalized-token scan
    # sweeps in ("Am I texting…" -> "Am", "Show me…" -> "Show"). Each would
    # otherwise fuzzy-resolve to a real contact (Am -> Durham/Graham), poison the
    # scratchpad, and -- with the unresolved-name guard -- suppress a correct
    # forced route; measured on the Apple FM eval sweep as its single largest
    # class of wrong-scope answers.
    "am", "could", "would", "should", "shall", "might",
    "must", "been", "being", "be", "get", "got", "make", "made", "let", "see",
    "look", "there", "here", "any", "some", "no", "not", "over", "about",
    "around", "between", "with", "for", "from", "of", "on", "in", "at", "to",
    "does", "whats", "hows", "wheres", "im", "ive", "id",
})
# Auxiliaries that are ALSO real first names (Will, May, Can). Unconditionally
# stopwording them swallows the contact ("How often does Will text me?" -> no
# candidate), which ALSO disarms the unresolved-name guard (no candidates -> no
# guard) so a forced contact-scoped intent runs unscoped and misattributes
# whole-corpus stats to the person -- the S7.6 confabulation. Capitalization
# disambiguates: mid-sentence these are capitalized only as names, while an
# auxiliary is capitalized only sentence-initially ("Will I ever…"). So they are
# trimmed ONLY as a span's first token at a sentence start.
_POSITIONAL_STOPWORDS = frozenset({"will", "may", "can"})
# Nothing but whitespace / an opening quote/paren between the span and either the
# start of the question or the previous sentence's terminator.
_SENTENCE_INITIAL_RE = re.compile(r"(?:^|[.?!;])[\s\"'“”‘’(\[]*$")
_NAME_CANDIDATE_RE = re.compile(r"[A-Z][a-zA-Z'’]+(?:\s+[A-Z][a-zA-Z'’]+)*")
_MAX_PRERESOLVE_NAMES = 4
_MAX_PRERESOLVE_CONTACTS = 2
_MAX_PRERESOLVE_CHATS = 3


def _extract_candidate_names(question):
    """Pull the capitalized name-shaped spans out of the question (single or
    multi-word), dropping the sentence-initial question words / verbs a naive
    scan also catches. Deterministic and model-free -- a false candidate simply
    resolves to no contact and is skipped. Order-preserving, de-duplicated."""
    question = question or ""
    out, seen = [], set()
    for m in _NAME_CANDIDATE_RE.finditer(question):
        # Trim leading/trailing question-words the scan swept in ("And Kyle" ->
        # "Kyle", "Who" -> "") so a real name is kept clean and dedups with its
        # bare form; an all-stopword span trims to nothing and is dropped. A
        # name-colliding auxiliary (_POSITIONAL_STOPWORDS) is trimmed only as the
        # span's opening token at a sentence start -- mid-sentence, or after a
        # trimmed opener ("Can Will text back?"), it is a name and is kept; the
        # trailing trim never touches it ("Jenna May" stays whole).
        words = m.group(0).split()
        sentence_initial = bool(_SENTENCE_INITIAL_RE.search(question[:m.start()]))
        lead = 0
        while lead < len(words):
            head = words[lead].lower()
            if head in _NAME_STOPWORDS or (
                    lead == 0 and sentence_initial and head in _POSITIONAL_STOPWORDS):
                lead += 1
            else:
                break
        words = words[lead:]
        while words and words[-1].lower() in _NAME_STOPWORDS:
            words.pop()
        name = " ".join(words)
        if len(name) < 2:
            continue
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out[:_MAX_PRERESOLVE_NAMES]


def _preresolve_entities(chat, question):
    """Resolve person names in the question to labeled id facts BEFORE the staged
    loop, using only the public directory tools. This hands the FM model the
    chat_id / handle_id / contact_id it otherwise fabricates (or resolves
    unreliably across several loop turns), each clearly labeled so it cannot
    conflate a contact_id with a chat_id. Best-effort: any failure degrades to no
    seed. Deterministic infra, so it does not consume APPLE_MAX_TOOL_ITERATIONS
    and is not offered to the _recorder (which counts model-initiated calls).

    Returns (facts, resolved): `facts` are the [resolved] scratchpad seed strings;
    `resolved` is the structured list ({"contact_id","chats"}) that lets a forced
    contact-scoped intent be filled deterministically (_primary_contact) rather
    than trusting the FM to reuse the right labeled id."""
    find_contacts = chat.TOOL_FUNCTIONS.get("search_contacts_by_name")
    find_chats = chat.TOOL_FUNCTIONS.get("list_chats")
    if find_contacts is None:
        return [], []
    facts, resolved = [], []
    for name in _extract_candidate_names(question):
        try:
            contacts = find_contacts(name=name, limit=5)
        except Exception:
            continue
        if not isinstance(contacts, list):
            continue
        for c in contacts[:_MAX_PRERESOLVE_CONTACTS]:
            if not isinstance(c, dict) or c.get("id") is None:
                continue
            cid = c["id"]
            handle_ids = [h.get("id") for h in (c.get("handles") or [])
                          if isinstance(h, dict) and h.get("id") is not None]
            label = " ".join(str(c.get(k) or "").strip()
                             for k in ("first_name", "last_name")).strip() or name
            direct_chats = []
            if find_chats is not None:
                try:
                    rows = find_chats(contact_id=cid, limit=50)
                    direct_chats = [r["id"] for r in (rows or [])
                                    if isinstance(r, dict) and not r.get("is_group")
                                    and r.get("id") is not None][:_MAX_PRERESOLVE_CHATS]
                except Exception:
                    direct_chats = []
            fact = (f'"{name}" -> {label}: contact_id={cid}'
                    + (f", handle_id={handle_ids}" if handle_ids else "")
                    + (f", 1:1 chat_id={direct_chats}" if direct_chats else ""))
            facts.append(f"[resolved] {fact}")
            resolved.append({"contact_id": cid, "chats": direct_chats})
            chat.console.print(f"  [resolve] {fact}", style="dim cyan", markup=False)
    return facts, resolved


# Forced-intent tools whose scope is naturally a contact_id -- when the question
# named exactly one clear person, we fill that scope deterministically instead of
# trusting the FM (which conflates contact_id with chat_id even when the seed
# labels both). No-name questions leave `resolved` empty and fall through to the
# normal FILL (whole-corpus for the aggregates).
_CONTACT_SCOPED_INTENT_TOOLS = frozenset({
    "get_milestones", "get_response_dynamics", "get_chat_summary_stats",
    "get_activity_patterns",
})


def _primary_contact(resolved):
    """The single unambiguous contact to scope a forced aggregate by: the one
    resolved contact that has a known 1:1 chat (a real conversation), or the sole
    resolved contact if only one matched. None when ambiguous or empty -- then the
    FM fills the scope itself."""
    if not resolved:
        return None
    with_chat = [r for r in resolved if r.get("chats")]
    if len(with_chat) == 1:
        return with_chat[0]["contact_id"]
    if not with_chat and len(resolved) == 1:
        return resolved[0]["contact_id"]
    return None


# Forced-intent tools whose scope is a GROUP chat_id. get_group_dynamics /
# get_chat_participant_stats error without one, and _preresolve_entities only
# resolves PEOPLE -- so a group named in the question ("Real Tahoe") went
# unscoped and the tool errored, then the model confabulated from search. These
# get their chat_id filled from a group pre-resolution instead.
_GROUP_SCOPED_INTENT_TOOLS = frozenset({
    "get_group_dynamics", "get_chat_participant_stats",
})


def _preresolve_groups(chat, question):
    """Resolve a group chat named in the question to its chat_id BEFORE the loop,
    via the public list_chats(query=...) directory search -- the group analogue of
    _preresolve_entities. Returns (facts, group_ids): the [resolved] scratchpad
    seed lines and the list of matched group chat_ids (deduped). Best-effort: any
    failure degrades to no seed. Named after the same candidate-name scan people
    use, so a person name that matches no group simply yields nothing."""
    find_chats = chat.TOOL_FUNCTIONS.get("list_chats")
    if find_chats is None:
        return [], []
    facts, group_ids, seen = [], [], set()
    for name in _extract_candidate_names(question):
        try:
            rows = find_chats(query=name, limit=5)
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for r in rows[:_MAX_PRERESOLVE_CHATS]:
            if not (isinstance(r, dict) and r.get("is_group") and r.get("id") is not None):
                continue
            # list_chats(query=) matches on PARTICIPANTS too, so a person's name
            # returns every group they are in (mostly unnamed). Require the group's
            # own display_name to match the candidate -- that is the case where the
            # question actually named the GROUP (e.g. "Real Tahoe"), not a member.
            dn = (r.get("display_name") or "").strip()
            nl, dl = name.lower(), dn.lower()
            if not dn or (nl not in dl and dl not in nl):
                continue
            gid = r["id"]
            if gid in seen:
                continue
            seen.add(gid)
            group_ids.append(gid)
            fact = f'"{name}" -> group "{dn}": chat_id={gid}'
            facts.append(f"[resolved] {fact}")
            chat.console.print(f"  [resolve] {fact}", style="dim cyan", markup=False)
    return facts, group_ids


# --- Deterministic intent -> tool routing (the loop's forced first action) ---
# The small FM model routes fine once pointed at the right tool but reliably
# MIS-picks the canonical aggregate/milestone tools -- reaching for
# search_conversations / search_messages_sql on "how many messages", "what hour",
# or "when did we first text" (which search_messages_sql cannot answer: it is
# newest-first only). Each rule maps a high-precision question pattern to the
# canonical tool and forces it as iteration 0's action, bypassing _route (also
# saving one model round-trip). Arg-filling still goes through _fill_args, so a
# rule decides only the tool NAME. A miss falls back to model routing -- no harm.
_ROUTE_RULES = [
    # --- Specific intents FIRST. First rule wins: any specific intent whose
    # phrasing can also match a generic frequency/volume rule below ("how often
    # do i", "when do i text", ...) must sit ABOVE the activity block, or the
    # generic rule steals it ("how often do I ghost people?" is a ghost ask). ---
    (re.compile(r"\bdouble.?text", re.I), "get_response_dynamics"),
    (re.compile(r"\bghost", re.I), "get_response_dynamics"),
    (re.compile(r"\b(leave|left|leaves) .* on read\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bnot replied to a conversation\b", re.I), "get_response_dynamics"),
    # --- Milestones (firsts / nth / anniversaries / gaps / history bounds) ---
    (re.compile(r"\bfirst (text|message|messaged|texted)\b", re.I), "get_milestones"),
    (re.compile(r"\bwho (texted|messaged|reached out|contacted) .*first\b", re.I),
     "get_milestones"),
    (re.compile(r"\bwhen did .* first\b", re.I), "get_milestones"),
    (re.compile(r"\b\d[\d,]*(st|nd|rd|th) message\b", re.I), "get_milestones"),
    (re.compile(r"\banniversar", re.I), "get_milestones"),
    (re.compile(r"\bhow far back\b", re.I), "get_milestones"),
    (re.compile(r"\blast (thing|message) .*(sent|texted|said)\b", re.I), "get_milestones"),
    (re.compile(r"\blongest (silence|gap|stretch)\b", re.I), "get_milestones"),
    (re.compile(r"\bwithout (talking|texting)\b", re.I), "get_milestones"),
    # --- Activity patterns (heatmap / streaks / late-night / volume) ---
    (re.compile(r"\blongest streak\b", re.I), "get_activity_patterns"),
    (re.compile(r"\b(after midnight|3\s?am|late at night)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\b(what|which) (hour|time of day)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bbusiest (\w+ )?(day|hour|time|night)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bwhen (am|do) (i|we) .*(active|text|message)\b", re.I),
     "get_activity_patterns"),
    (re.compile(r"\bwhen do (i|we) (text|message)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bhow often do (i|we)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\btext(ing)? (more|less)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\btext(ed|ing)? (much|a lot) (with|in)\b", re.I), "get_activity_patterns"),
    (re.compile(r"\beach (month|year|week) of \d{4}\b", re.I), "get_activity_patterns"),
    (re.compile(r"\bevolved\b", re.I), "get_activity_patterns"),
    # --- Response dynamics ---
    (re.compile(r"\bwho (replies|responds|texts|gets back) .*(faster|fastest|first)\b",
                re.I), "get_response_dynamics"),
    (re.compile(r"\bquickest to respond\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bhow (long|fast|quick).*(reply|respond|get back|text back)\b", re.I),
     "get_response_dynamics"),
    (re.compile(r"\bresponse time\b", re.I), "get_response_dynamics"),
    (re.compile(r"\bwho (starts|initiates) .*(me or|\bmore\b)", re.I),
     "get_response_dynamics"),
    (re.compile(r"\b(long conversations|quick bursts)\b", re.I), "get_response_dynamics"),
    # --- Per-member (chat_participant_stats) BEFORE reactions/group (more specific) ---
    (re.compile(r"\breactions? per message\b", re.I), "get_chat_participant_stats"),
    (re.compile(r"\bat least \d+ messages\b", re.I), "get_chat_participant_stats"),
    # --- Reactions ---
    (re.compile(r"\btapback", re.I), "get_reaction_stats"),
    (re.compile(r"\bwhat reactions do\b", re.I), "get_reaction_stats"),
    (re.compile(r"\bfunniest\b", re.I), "get_reaction_stats"),
    (re.compile(r"\bfinds? me funn", re.I), "get_reaction_stats"),
    (re.compile(r"\breact(s|ed)? to (my|your|me|them|their|his|her)\b", re.I),
     "get_reaction_stats"),
    (re.compile(r"\bmost reactions\b", re.I), "get_reaction_stats"),
    (re.compile(r"\b(love|loved|like|liked|laugh|laughed) (at )?my (message|text)", re.I),
     "get_reaction_stats"),
    (re.compile(r"\bhow (quickly|fast) .*react\b", re.I), "get_reaction_stats"),
    # --- Group dynamics ---
    (re.compile(r"\bwho (carries|lurks?|dominates)\b", re.I), "get_group_dynamics"),
    (re.compile(r"\b(dominated by|balanced)\b", re.I), "get_group_dynamics"),
    (re.compile(r"\bwho (starts|talks|texts) .* (in|inside)\b", re.I), "get_group_dynamics"),
    (re.compile(r"\btalks? to whom\b", re.I), "get_group_dynamics"),
    (re.compile(r"\bstands? out\b", re.I), "get_group_dynamics"),
    # --- Words / language (trends) ---
    (re.compile(r"\binside jokes?\b", re.I), "get_word_trends"),
    (re.compile(r"\bunique to (me|us)\b", re.I), "get_word_trends"),
    (re.compile(r"\bemoji\b", re.I), "get_word_trends"),
    (re.compile(r"\buse of the (word|phrase)\b", re.I), "get_word_trends"),
    (re.compile(r"\bstart(ed)? saying\b", re.I), "get_word_trends"),
    (re.compile(r"\bhave i ever (said|used|texted)\b", re.I), "get_word_trends"),
    (re.compile(r"\b(laugh a lot|we laugh)\b", re.I), "get_word_trends"),
    (re.compile(r"\bmood of (my|our|the) (chats?|texts?)\b", re.I), "get_word_trends"),
    (re.compile(r"\bemotions\b", re.I), "get_word_trends"),
    (re.compile(r"\bhow similar .* style\b", re.I), "get_word_trends"),
    # --- Word frequency ---
    (re.compile(r"\bmost common words\b", re.I), "get_word_frequency"),
    (re.compile(r"\b(two.word|2.word|two word) phrase", re.I), "get_word_frequency"),
    (re.compile(r"\bphrases? (do )?(i|you) (say|use)\b", re.I), "get_word_frequency"),
    # --- Voiceprint ---
    (re.compile(r"\bvoiceprint\b", re.I), "get_voiceprint"),
    (re.compile(r"\b(describe|what.?s) my .*(texting )?(style|voice)\b", re.I),
     "get_voiceprint"),
    # --- Summary stats (counts / totals / attachments / who-most leaderboard) ---
    (re.compile(r"\bhow many (photos|pictures|images|attachments)\b", re.I),
     "get_chat_summary_stats"),
    (re.compile(r"\bhow many (total )?(messages|texts)\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\btotal (number of )?(messages|texts)\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\bmessage count\b", re.I), "get_chat_summary_stats"),
    (re.compile(r"\bwho do (i|you) (text|message|talk to) (the )?most\b", re.I),
     "get_chat_summary_stats"),
    # --- Nudges / loose ends ---
    (re.compile(r"\bowe .*(a )?(reply|text|response)\b", re.I), "get_loose_threads"),
    (re.compile(r"\bleft hanging\b", re.I), "get_loose_threads"),
    (re.compile(r"\bnever (got back|replied|responded)\b", re.I), "get_loose_threads"),
    (re.compile(r"\bnudges?\b", re.I), "get_loose_threads"),
    (re.compile(r"\bunanswered\b", re.I), "get_loose_threads"),
    # --- Drift / reconnection (fadeout gated behind DRIFT_ENABLED) ---
    (re.compile(r"\b(fad(e|ing|ed)|drift(ing|ed)?|growing apart)\b", re.I),
     "get_fadeout_watchlist"),
    (re.compile(r"\b(lost touch|reconnect|break the ice)\b", re.I), "get_loose_threads"),
    # --- Conflict (find_conflicts; apology/draft guarded out in _suggest_first_action) ---
    (re.compile(r"\b(rough patch(es)?|tense stretch(es)?|tensest|falling out)\b", re.I),
     "find_conflicts"),
    (re.compile(r"\b(fight|argument|arguments|conflict)\b", re.I), "find_conflicts"),
    # --- Persona / wellbeing ---
    (re.compile(r"\bsound like (something )?(i|me) would write\b", re.I),
     "check_style_match"),
    (re.compile(r"\bwhat (might|would) .*(text|say|respond) back\b", re.I),
     "simulate_contact"),
    (re.compile(r"\b(mood or mental health|mental health)\b", re.I),
     "get_language_trajectory"),
    (re.compile(r"\b(changed|shifted).*(way (i|she|he|they) (write|text)|sustained)\b",
                re.I), "get_language_trajectory"),
    (re.compile(r"\bsustained (way|change|departure)\b", re.I), "get_language_trajectory"),
]

# Intents whose canonical tool takes an id only a PRIOR call can supply
# (draft_apology/diagnose_conflict need a conflict_id) or that the model already
# routes well on its own (draft_reply needs the inbound message first) -- forcing
# a first action here would either misfire or cut the loop short before the
# follow-up call. When matched, skip deterministic routing and let the model
# stage the chain (it lands the second tool once find_conflicts/retrieval seeds
# the id/context).
_ROUTE_SKIP_RE = re.compile(r"\bapolog|\b(draft|write) (a |me a )?reply", re.I)


def _suggest_first_action(question):
    """The canonical tool for a recognized question intent, or None. First rule
    wins; only fires for a tool the FM model reliably mis-picks. Returns a name in
    chat_apple._TOOL_SPECS or None (guarded so a rule can never name an unregistered tool --
    a flag-gated tool absent from the catalog simply falls through to model
    routing). Apology/draft-reply intents are skipped (see _ROUTE_SKIP_RE)."""
    question = question or ""
    if _ROUTE_SKIP_RE.search(question):
        return None
    for pattern, tool in _ROUTE_RULES:
        if tool in chat_apple._TOOL_SPECS and pattern.search(question):
            return tool
    return None
