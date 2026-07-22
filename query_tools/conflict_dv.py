"""DV power-and-control screen (Master Spec A10, SOFTENED / code-enforced).

Split out of conflict.py into a single home so the DV floor is unambiguously
one source: `_dv_screen` and the transient `set_dv_screen_bypass` flag live
here and are imported by BOTH diagnose_conflict (query_tools.conflict) and
draft_apology (query_tools.conflict_apology). The scan is read-only and never
persisted. Generic public screening vocabulary only -- no personal data.
"""

import config
from function_words import tokenize


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
