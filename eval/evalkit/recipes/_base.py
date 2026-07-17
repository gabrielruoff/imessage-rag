"""P0 ground truth: the @recipe registry (plan §4.2, §6); roster resolution
lives in roster.py.

Every recipe computes what a correct answer must contain by calling
query_tools directly (no Ollama) and returns
{"values": {name: {"value": v, "tol": "exact|round|pct1"}}, "raw": {...}}.
App imports live inside functions -- TARGET_DB must be settable in the
environment before config.py first loads (env-before-import invariant)."""

RECIPES = {}

# Minimum messages sent to make a group member eligible for S6.8's
# reactions-per-message leaderboard. Keep in sync with S6.8's prompt in
# eval/cases/S6.json, which states the same floor -- the model can only apply a
# filter it is told about.
PARTICIPANT_REACTIONS_MIN_MESSAGES = 20


def recipe(name):
    def register(fn):
        RECIPES[name] = fn
        return fn

    return register


def _qt():
    import query_tools

    return query_tools


def _connect():
    import db
    import query_tools

    return db.connect_default(query_tools.DB_PATH, vec=False)


def _v(value, tol):
    return {"value": value, "tol": tol}


def _scope_kwargs(entry):
    """Dyad-faithful scope: the roster entry's chat_ids (a contact's 1:1
    twins, a group's split threads). contact_id scoping would differ two
    ways: the analytics tools resolve person -> ALL their chats (group chats
    included), and get_chat_summary_stats/get_word_frequency scope by
    *sender* -- either way, not the dyad the scoped prompts ask about."""
    chat_ids = entry.get("chat_ids")
    if not chat_ids:
        raise ValueError(f"roster entry {entry.get('display_name')!r} has no chat_ids")
    return {"chat_id": chat_ids}


def _year_bounds(year):
    return f"{year}-01-01", f"{year}-12-31"


def _real():
    """Real conversation messages only, matching query_tools._shared's
    _REAL_MESSAGE_CLAUSES plus the dated requirement every aggregate applies.
    Derived from the app's tuple at call time (lazy import, so the
    env-before-import invariant holds) -- the two can never diverge."""
    from query_tools._shared import _REAL_MESSAGE_CLAUSES

    return "m.date IS NOT NULL AND " + " AND ".join(_REAL_MESSAGE_CLAUSES)
