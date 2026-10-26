"""Small helpers shared between ingest.py, chunking.py, query_tools, and
reset_db.py: sqlite-vec extension loading, the contact-label SQL fragment,
the ISO-date -> day-number conversion used by chunk_vectors' metadata
columns, and the ingest_state key/value accessors."""

import datetime
import importlib
import os
import re
import sqlite3
import sys

import sqlite_vec


def load_vec_extension(con):
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)


def connect(path, vec=True, rows=True):
    """Open the analysis DB the one shared way. Callers keep the path in a
    module-level attribute (their test patch seam) and pass it in; vec/rows
    opt out of the sqlite-vec extension and sqlite3.Row for the callers that
    don't use them."""
    con = sqlite3.connect(path)
    if vec:
        load_vec_extension(con)
    if rows:
        con.row_factory = sqlite3.Row
    return con


def connect_default(path, *, vec=True, rows=True):
    """The one home for the batch CLIs' connection defaults. The per-module
    ``_connect()`` wrappers (annotate.py, topics.py, wrapped.py,
    eval/evalkit recipes) are documented test-patch seams: they keep their
    module-level DB path attribute and delegate here, so the defaults are
    defined once."""
    return connect(path, vec=vec, rows=rows)


def run_cli(main):
    """Shared ``__main__`` tail for the batch CLIs: exit with main()'s return
    code (None -> 0) and turn Ctrl-C / EOF into the conventional exit 130.

    A script is a second module object beside the one its children import, so
    ``main`` is re-read off the real module (DECISIONS: "a script is a second
    copy of its own module")."""
    if main.__module__ == "__main__":
        g = main.__globals__
        spec = g.get("__spec__")  # set under `python -m x`, None for `python x.py`
        name = spec.name if spec else os.path.splitext(os.path.basename(g["__file__"]))[0]
        main = getattr(importlib.import_module(name), main.__name__)
    try:
        sys.exit(main() or 0)
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted, exiting.")
        sys.exit(130)


# Human-readable label for a contact/handle: nickname > "First Last" > raw
# handle_identifier. Assumes the enclosing query joins handles AS h and
# contacts AS c (on h.contact_id).
CONTACT_LABEL_SQL = (
    "COALESCE("
    "NULLIF(c.nickname, ''), "
    "NULLIF(TRIM(COALESCE(c.first_name, '') || ' ' || COALESCE(c.last_name, '')), ''), "
    "h.handle_identifier)"
)


# chat.db's associated_message_guid points at a message *part*
# ('p:0/<guid>', 'bp:<guid>'); messages.guid is always bare. Reactions are
# stored with the prefix stripped -- at insert time by ingest, and for old
# rows by create_schema's migration -- so they join messages.guid. Both
# sites use this one function so they can never disagree.
_REACTION_GUID_PREFIX = re.compile(r"^(?:p:\d+/|bp:)")


def strip_reaction_guid_prefix(guid):
    return _REACTION_GUID_PREFIX.sub("", guid) if guid else guid


# Apple tapback type codes as stored in reactions.associated_message_type.
# 2000-2007 are tapbacks being added; 3000-3007 remove the corresponding
# tapback. Other codes (2, 3, 1000, ...) are non-tapback associated messages;
# counting queries filter to BETWEEN 2000 AND 2999 so removals and strays
# never inflate reaction counts.
TAPBACK_NAMES = {
    2000: "love",
    2001: "like",
    2002: "dislike",
    2003: "laugh",
    2004: "emphasize",
    2005: "question",
    2006: "custom-emoji",
    2007: "sticker",
}


def tapback_name(code):
    """Human-readable name for a reactions.associated_message_type code."""
    if code in TAPBACK_NAMES:
        return TAPBACK_NAMES[code]
    if code is not None and 3000 <= code <= 3007 and code - 1000 in TAPBACK_NAMES:
        return "removed-" + TAPBACK_NAMES[code - 1000]
    return f"unknown-{code}"


def get_state(con, key, default=None):
    """Read one ingest_state value (last_message_rowid, embed_model, ...)."""
    row = con.execute("SELECT value FROM ingest_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_state(con, key, value):
    con.execute(
        "INSERT INTO ingest_state(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def get_float_state(con, key):
    """A positive float previously banked under `key`, else None. Anything
    unparseable or non-positive reads as None: these keys hold measured rates
    that prime a budget's arithmetic, and a garbage value must degrade to "no
    measurement" rather than poison a deadline."""
    raw = get_state(con, key)
    try:
        value = float(raw) if raw else None
    except (TypeError, ValueError):
        return None
    return value if value and value > 0 else None


def set_float_state(con, key, value):
    """Bank a MEASURED float under `key` and commit. A falsy value writes
    nothing -- every caller here measures a rate, and storing a zero (or the
    cold prior a run started from) would freeze a non-measurement into the DB
    where nothing would ever correct it."""
    if not value:
        return
    set_state(con, key, repr(value))
    con.commit()


def check_model_guard(con, *, key, model, table, message, predicate="1", record=True):
    """The repo's one model guard: refuse to mix generation models within one
    derived artifact (the same shape as chunking.build_vec_table's embedding
    guard), and record the model that produced it.

    `key` is the ingest_state key, `table`/`predicate` count only the rows this
    model actually produced (dyad_terms holds uncurated rows too), and `message`
    is the refusal text, rendered with {n} (rows found), {prev}, {model} and
    {table}. Each pipeline passes its own wording -- the guards' shape is
    shared, the remedy sentence is not.

    `record=False` runs the REFUSAL half only. Callers use it for an early call
    that fires before anything destructive (conflict_detect's --relabel clear
    and detection sweep, loose_ends' dropped_plan detection), so a run that will
    be refused is refused before it destroys or pays for anything -- while the
    key still means "the model that wrote these rows", set by the pass that
    actually writes them."""
    prev = get_state(con, key)
    if prev and prev != model:
        n = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {predicate}").fetchone()[0]
        if n:
            print(message.format(n=n, prev=prev, model=model, table=table))
            raise SystemExit(1)
    if record:
        set_state(con, key, model)
        con.commit()


def resolve_budget_minutes(flag, env_value, default, *, zero_means, flag_name,
                           env_name, default_name, flag_error, env_error):
    """(minutes, provenance) for a wall-clock budget. Precedence: flag > env >
    default -- the one order both budgeted pipelines use.

    `zero_means` is the only thing that genuinely differs between them:
      * "unset"      -- 0 falls through to the next source (annotate's hybrid
                        tier: the budget IS the tier, so there is no such thing
                        as a zero-minute run and 0 must mean the default).
      * "unbudgeted" -- 0 is a real value meaning "no deadline" (conflict
                        labeling: the budget is additive, so switching it off
                        has to restore the unbounded path, not a default).

    A NEGATIVE value is refused from either source, never clamped. Passed
    through it is a deadline already in the past, which every phase declines
    while reporting a clean exit; clamped to 0 it silently buys either the
    default or the unbounded run the CLI refuses a negative flag for. The
    caller supplies `flag_error`/`env_error` (rendered with {value}) so the
    refusal names the knob it came from in that pipeline's own words."""
    if flag is not None:
        if flag < 0:
            raise SystemExit(flag_error.format(value=flag))
        if flag or zero_means == "unbudgeted":
            return flag, flag_name
    if env_value < 0:
        raise SystemExit(env_error.format(value=env_value))
    if env_value or zero_means == "unbudgeted":
        return env_value, env_name
    return default, default_name


def like_escape(value):
    r"""Escape LIKE wildcards for use with ESCAPE '\'. Lives here rather than in
    either caller because both `annotate` and `annotate_events` need it and
    importing one from the other is an import cycle -- `db` is the module
    everything may import."""
    return (value.replace("\\", "\\\\")
                 .replace("%", "\\%")
                 .replace("_", "\\_"))


def iso_to_day(iso_date: str) -> int:
    """Calendar day number (proleptic Gregorian ordinal) of an ISO date or
    datetime string. Written into chunk_vectors' first_day/last_day INTEGER
    metadata columns at ingest time and compared against by query_tools'
    date-range filters -- both sides must use this same function."""
    return datetime.date.fromisoformat(iso_date[:10]).toordinal()
