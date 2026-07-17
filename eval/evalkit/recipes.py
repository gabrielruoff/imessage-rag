"""P0 ground truth: resolve_roster() + the @recipe registry (plan §4.2, §6).

Every recipe computes what a correct answer must contain by calling
query_tools directly (no Ollama) and returns
{"values": {name: {"value": v, "tol": "exact|round|pct1"}}, "raw": {...}}.
App imports live inside functions -- TARGET_DB must be settable in the
environment before config.py first loads (env-before-import invariant)."""

import random
import re

RECIPES = {}


def recipe(name):
    def register(fn):
        RECIPES[name] = fn
        return fn

    return register


def _qt():
    import query_tools

    return query_tools


def _connect():
    import sqlite3

    import query_tools

    con = sqlite3.connect(query_tools.DB_PATH)
    con.row_factory = sqlite3.Row
    return con


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


# Real conversation messages only, matching query_tools._shared's
# _REAL_MESSAGE_CLAUSES plus the dated requirement every aggregate applies.
_REAL = "m.date IS NOT NULL AND COALESCE(m.is_emote, 0) = 0 AND COALESCE(m.item_type, 0) = 0"


# --- roster resolution -------------------------------------------------------


def _contact_entry(con, contact_id, n_messages):
    from db import CONTACT_LABEL_SQL

    name = con.execute(
        f"SELECT {CONTACT_LABEL_SQL} FROM handles h JOIN contacts c ON c.id = h.contact_id "
        "WHERE c.id = ? LIMIT 1",
        (contact_id,),
    ).fetchone()[0]
    handle_ids = [
        r[0]
        for r in con.execute(
            "SELECT id FROM handles WHERE contact_id = ? ORDER BY id", (contact_id,)
        )
    ]
    chat_ids = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT cp.chat_id FROM chat_participants cp "
            "JOIN chats ch ON ch.id = cp.chat_id "
            "JOIN handles h ON h.id = cp.handle_id "
            "WHERE h.contact_id = ? AND ch.is_group = 0 ORDER BY cp.chat_id",
            (contact_id,),
        )
    ]
    return {
        "display_name": name,
        "contact_id": contact_id,
        "handle_ids": handle_ids,
        "chat_ids": chat_ids,
        "n_messages": n_messages,
    }


_CONTACT_COUNTS_SQL = """
WITH contact_chats AS (
  SELECT DISTINCT h.contact_id AS contact_id, ch.id AS chat_id
  FROM chats ch
  JOIN chat_participants cp ON cp.chat_id = ch.id
  JOIN handles h ON h.id = cp.handle_id
  WHERE ch.is_group = 0 AND h.contact_id IS NOT NULL
)
SELECT cc.contact_id, COUNT(*) AS n
FROM contact_chats cc JOIN messages m ON m.chat_id = cc.chat_id
WHERE {real}
GROUP BY cc.contact_id
ORDER BY n DESC
"""


def _pick_phrase_hi(con, qt):
    """A frequent word (>= 200 lifetime FTS hits) whose per-year usage swings
    the most (max/min year ratio)."""
    freq = qt.get_word_frequency(is_from_me=1, top=60)
    best, best_ratio = None, 0.0
    for row in freq["rows"]:
        word = row["ngram"]
        if not word.isalpha() or len(word) < 3:
            continue
        years = con.execute(
            "SELECT substr(m.date, 1, 4) AS y, COUNT(*) AS n "
            "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
            "WHERE messages_fts MATCH ? AND m.date IS NOT NULL GROUP BY y",
            (f'"{word}"',),
        ).fetchall()
        counts = [r["n"] for r in years]
        if len(counts) < 2 or sum(counts) < 200:
            continue
        ratio = max(counts) / max(1, min(counts))
        if ratio > best_ratio:
            best, best_ratio = word, ratio
    return best


def _pick_phrase_lo(con):
    """Seeded-random walk over longer sent messages; first alphabetic 3-gram
    with 1-5 lifetime FTS hits."""
    rng = random.Random(20260716)
    rows = con.execute(
        "SELECT text FROM messages WHERE is_from_me = 1 AND text IS NOT NULL "
        "AND length(text) BETWEEN 60 AND 300 ORDER BY id LIMIT 5000"
    ).fetchall()
    rows = [r["text"] for r in rows]
    rng.shuffle(rows)
    for text in rows:
        words = [w for w in re.findall(r"[A-Za-z']+", text) if len(w) > 2]
        if len(words) < 5:
            continue
        start = rng.randrange(len(words) - 2)
        phrase = " ".join(words[start : start + 3]).lower()
        hits = con.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (f'"{phrase}"',),
        ).fetchone()[0]
        if 1 <= hits <= 5:
            return phrase
    return None


def resolve_roster(db_path, overrides=None):
    """Resolve the placeholder roster against the real DB (plan §4.2).
    Prints every auto-pick with its evidence; `overrides` merge last."""
    qt = _qt()
    qt.DB_PATH = str(db_path)
    roster = {}
    with _connect() as con:
        counts = con.execute(_CONTACT_COUNTS_SQL.format(real=_REAL)).fetchall()
        if len(counts) < 2:
            raise RuntimeError("fewer than two 1:1 contacts with messages -- wrong DB?")
        for name, row in zip(("TOP1", "TOP2"), counts[:2]):
            roster[name] = _contact_entry(con, row["contact_id"], row["n"])

        sparse_candidates = [
            r
            for r in counts
            if 30 <= r["n"] <= 100
            and con.execute(
                "SELECT 1 FROM contacts WHERE id = ? AND first_name IS NOT NULL "
                "AND first_name != ''",
                (r["contact_id"],),
            ).fetchone()
        ]
        if not sparse_candidates:
            raise RuntimeError("no named contact with 30-100 lifetime messages")
        sparse = min(sparse_candidates, key=lambda r: abs(r["n"] - 60))
        roster["SPARSE"] = _contact_entry(con, sparse["contact_id"], sparse["n"])

        group = con.execute(
            f"""
            SELECT ch.id, ch.display_name, COUNT(m.id) AS n
            FROM chats ch JOIN messages m ON m.chat_id = ch.id
            WHERE ch.is_group = 1 AND ch.display_name IS NOT NULL AND ch.display_name != ''
              AND {_REAL}
              AND (SELECT COUNT(*) FROM chat_participants cp WHERE cp.chat_id = ch.id) >= 4
            GROUP BY ch.id ORDER BY n DESC LIMIT 1
            """
        ).fetchone()
        if group is None:
            raise RuntimeError("no group chat with >= 4 participants and a display name")
        twin_ids = [
            r[0]
            for r in con.execute(
                "SELECT id FROM chats WHERE is_group = 1 AND display_name = ? ORDER BY id",
                (group["display_name"],),
            )
        ]
        placeholders = ",".join("?" * len(twin_ids))
        n_participants = con.execute(
            f"SELECT COUNT(DISTINCT handle_id) FROM chat_participants "
            f"WHERE chat_id IN ({placeholders})",
            twin_ids,
        ).fetchone()[0]
        roster["GROUP"] = {
            "display_name": group["display_name"],
            "chat_ids": twin_ids,
            "n_participants": n_participants,
            "n_messages": group["n"],
        }

        max_date = con.execute("SELECT MAX(date) FROM messages").fetchone()[0]
        max_year = int(max_date[:4])
        roster["YEAR"] = max_year if max_date >= f"{max_year}-12-31" else max_year - 1

        roster["PHRASE_HI"] = _pick_phrase_hi(con, qt)
        roster["PHRASE_LO"] = _pick_phrase_lo(con)

        top1_chats = roster["TOP1"]["chat_ids"]
        term = None
        if top1_chats:
            import sqlite3

            placeholders = ",".join("?" * len(top1_chats))
            try:
                row = con.execute(
                    f"SELECT term FROM dyad_terms WHERE chat_id IN ({placeholders}) "
                    "ORDER BY CASE WHEN llm_verdict IN ('inside_joke', 'nickname') "
                    "THEN 0 ELSE 1 END, lift DESC LIMIT 1",
                    top1_chats,
                ).fetchone()
                term = row["term"] if row else None
            except sqlite3.OperationalError as e:  # derived table not built yet
                print(f"  TERM: unavailable ({e})")
        roster["TERM"] = term

    nobody = "Thaddeus Vellingworth"
    matches = qt.search_contacts_by_name(nobody)
    rows = matches["rows"] if isinstance(matches, dict) else matches
    if rows:
        nobody = "Peregrine Ashcombe-Villiers"  # fallback; verified implausible
    roster["NOBODY"] = nobody

    for name in ("TOP1", "TOP2", "SPARSE", "GROUP"):
        e = roster[name]
        print(
            f"  {name}: {e['display_name']!r} ({e['n_messages']} messages, "
            f"chat_ids={e['chat_ids']})"
        )
    for name in ("YEAR", "PHRASE_HI", "PHRASE_LO", "TERM", "NOBODY"):
        print(f"  {name}: {roster[name]!r}")
    if overrides:
        for k, v in overrides.items():
            print(f"  override {k}: {v!r}")
        roster.update(overrides)
    missing = [k for k, v in roster.items() if v is None]
    if missing:
        print(f"  WARNING: unresolved placeholders {missing} -- cases using them will error")
    return roster


# --- S1: activity patterns ---------------------------------------------------


@recipe("activity_argmax")
def activity_argmax(roster, axis, scope=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope) if scope else {}
    r = qt.get_activity_patterns(include=["heatmap"], **kwargs)
    counts = r["heatmap"]["counts"]
    raw = {"tool": r}
    values = {}
    if axis == "hour":
        hour_totals = [sum(row[h] for row in counts) for h in range(24)]
        values["top_hour"] = _v(max(range(24), key=hour_totals.__getitem__), "exact")
        if scope is None:
            # The classic failure is a UTC-shaped argmax (4-5h off); store it
            # so graders can recognize it in a wrong reply.
            with _connect() as con:
                rows = con.execute(
                    "SELECT CAST(substr(m.date, 12, 2) AS INTEGER) AS h, COUNT(*) AS n "
                    f"FROM messages m WHERE {_REAL} GROUP BY h"
                ).fetchall()
            raw["utc_top_hour"] = max(rows, key=lambda r: r["n"])["h"]
    else:
        day_totals = [sum(row) for row in counts]
        top = max(range(len(day_totals)), key=day_totals.__getitem__)
        values["top_weekday"] = _v(r["heatmap"]["weekdays"][top], "exact")
    return {"values": values, "raw": raw}


@recipe("volume_series")
def volume_series(roster, scope, year):
    qt = _qt()
    date_from, date_to = _year_bounds(year)
    r = qt.get_activity_patterns(
        include=["volume_series"],
        granularity="month",
        date_from=date_from,
        date_to=date_to,
        **_scope_kwargs(scope),
    )
    values = {}
    for bucket in r["volume_series"]["rows"]:
        p = bucket["period"]
        values[f"{p}_sent"] = _v(bucket["sent"], "exact")
        values[f"{p}_received"] = _v(bucket["received"], "exact")
        values[f"{p}_total"] = _v(bucket["sent"] + bucket["received"], "exact")
    return {"values": values, "raw": r}


@recipe("year_compare")
def year_compare(roster):
    qt = _qt()
    r = qt.get_activity_patterns(include=["volume_series"], granularity="year")
    series = sorted(r["volume_series"]["rows"], key=lambda b: b["period"])
    if len(series) < 2:
        raise ValueError("fewer than two year buckets in the corpus")
    last, this = series[-2], series[-1]
    this_total = this["sent"] + this["received"]
    last_total = last["sent"] + last["received"]
    direction = "more" if this_total > last_total else "less" if this_total < last_total else "same"
    return {
        "values": {
            "this_year_total": _v(this_total, "exact"),
            "last_year_total": _v(last_total, "exact"),
            "direction": _v(direction, "exact"),
        },
        "raw": r,
    }


@recipe("late_night")
def late_night(roster):
    qt = _qt()
    r = qt.get_activity_patterns(include=["late_night"])
    ln = r["late_night"]
    values = {"late_night_share": _v(ln["share"], "pct1")}
    if ln.get("top_contacts"):
        values["top_contact"] = _v(ln["top_contacts"][0]["contact"], "exact")
    return {"values": values, "raw": r}


@recipe("streaks")
def streaks(roster, scope):
    qt = _qt()
    r = qt.get_activity_patterns(include=["streaks"], **_scope_kwargs(scope))
    s = r["streaks"]["longest_streak"]
    return {
        "values": {
            "streak_days": _v(s["days"], "exact"),
            "streak_start": _v(s["start"], "exact"),
            "streak_end": _v(s["end"], "exact"),
        },
        "raw": r,
    }


@recipe("busiest_day")
def busiest_day(roster):
    qt = _qt()
    r = qt.get_activity_patterns(include=["busiest_days"])
    top = r["busiest_days"][0]
    return {
        "values": {"day": _v(top["day"], "exact"), "count": _v(top["count"], "exact")},
        "raw": r,
    }


@recipe("message_count")
def message_count(roster, scope, year=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope)
    if year:
        kwargs["date_from"], kwargs["date_to"] = _year_bounds(year)
    r = qt.get_chat_summary_stats(**kwargs)
    return {"values": {"total": _v(r["message_count"], "exact")}, "raw": r}


# --- S2: response dynamics ---------------------------------------------------


@recipe("reply_latency")
def reply_latency(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    me, them = r["reply_latency"]["me"], r["reply_latency"]["them"]
    values = {
        "me_median_s": _v(me["median_seconds"], "round"),
        "them_median_s": _v(them["median_seconds"], "round"),
        "them_p90_s": _v(them["p90_seconds"], "round"),
        "faster_side": _v("me" if me["median_seconds"] < them["median_seconds"] else "them",
                          "exact"),
    }
    return {"values": values, "raw": r}


@recipe("reply_latency_pair")
def reply_latency_pair(roster, a, b):
    qt = _qt()
    ra = qt.get_response_dynamics(**_scope_kwargs(a))
    rb = qt.get_response_dynamics(**_scope_kwargs(b))
    return {
        "values": {
            "a_them_median_s": _v(ra["reply_latency"]["them"]["median_seconds"], "round"),
            "b_them_median_s": _v(rb["reply_latency"]["them"]["median_seconds"], "round"),
        },
        "raw": {"a": ra, "b": rb},
    }


@recipe("initiation")
def initiation(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    init = r["initiation"]
    more = "me" if init["me"] > init["them"] else "them" if init["them"] > init["me"] else "even"
    return {
        "values": {
            "initiation_me": _v(init["me"], "exact"),
            "initiation_them": _v(init["them"], "exact"),
            "more_side": _v(more, "exact"),
        },
        "raw": r,
    }


@recipe("double_text")
def double_text(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    return {"values": {"double_text_me": _v(r["double_text"]["me"], "exact")}, "raw": r}


@recipe("ghosted")
def ghosted(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    return {"values": {"ghosted_only_me": _v(r["ghosted"]["only_me"], "exact")}, "raw": r}


@recipe("read_to_reply")
def read_to_reply(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    rr = r["read_to_reply"]
    return {
        "values": {
            "pairs": _v(rr["pairs"], "exact"),
            "inbound_with_date_read": _v(rr["coverage"]["inbound_with_date_read"], "exact"),
            "inbound_total": _v(rr["coverage"]["inbound_total"], "exact"),
        },
        "raw": r,
    }


@recipe("sessions_burstiness")
def sessions_burstiness(roster, scope):
    qt = _qt()
    r = qt.get_response_dynamics(**_scope_kwargs(scope))
    s = r["sessions"]
    return {
        "values": {
            "session_count": _v(s["count"], "exact"),
            "median_messages": _v(s["median_messages"], "exact"),
            "burstiness": _v(r["burstiness"], "round"),
        },
        "raw": r,
    }


@recipe("group_response")
def group_response(roster, group):
    qt = _qt()
    r = qt.get_response_dynamics(chat_id=group["chat_ids"])
    by_member = r.get("initiations_by_member")
    if not by_member and len(group["chat_ids"]) > 1:
        r = qt.get_response_dynamics(chat_id=group["chat_ids"][0])
        by_member = r.get("initiations_by_member")
    values = {}
    if by_member:
        values["top_initiator"] = _v(by_member[0]["member"], "exact")
    return {"values": values, "raw": r}


# --- S3: reactions -----------------------------------------------------------


@recipe("reactions_top_type")
def reactions_top_type(roster, direction):
    qt = _qt()
    r = qt.get_reaction_stats(include=[direction])
    name, count = max(r[direction]["by_type"].items(), key=lambda kv: kv[1])
    return {
        "values": {"top_type": _v(name, "exact"), "top_type_count": _v(count, "exact")},
        "raw": r,
    }


@recipe("hall_of_fame_top")
def hall_of_fame_top(roster):
    qt = _qt()
    r = qt.get_reaction_stats(include=["hall_of_fame"])
    top = r["hall_of_fame"][0]
    return {
        "values": {
            "text": _v(top["text"], "exact"),
            "date": _v(top["date"], "exact"),
            "reaction_count": _v(top["reaction_count"], "exact"),
        },
        "raw": r,
    }


@recipe("reciprocity")
def reciprocity(roster, scope):
    qt = _qt()
    r = qt.get_reaction_stats(**_scope_kwargs(scope))
    rec = r["reciprocity"]
    return {
        "values": {
            "given": _v(rec["given"], "exact"),
            "received": _v(rec["received"], "exact"),
            "ratio": _v(rec["ratio"], "round"),
        },
        "raw": r,
    }


@recipe("funniest")
def funniest(roster):
    qt = _qt()
    r = qt.get_reaction_stats(include=["funniest"])
    top = r["funniest"][0]
    return {
        "values": {
            "funniest": _v(top["participant"], "exact"),
            "laughs_per_100": _v(top["laughs_per_100_my_messages"], "round"),
        },
        "raw": r,
    }


@recipe("time_to_react")
def time_to_react(roster):
    qt = _qt()
    r = qt.get_reaction_stats(include=["time_to_react"])
    return {
        "values": {
            "them_median_s": _v(r["time_to_react"]["them_reacting"]["median_seconds"], "round")
        },
        "raw": r,
    }


@recipe("reaction_types_pair")
def reaction_types_pair(roster, scope):
    qt = _qt()
    r = qt.get_reaction_stats(**_scope_kwargs(scope))
    values = {}
    for direction in ("given", "received"):
        by_type = r[direction]["by_type"]
        if by_type:
            name, _ = max(by_type.items(), key=lambda kv: kv[1])
            values[f"{direction}_top_type"] = _v(name, "exact")
    return {"values": values, "raw": r}


@recipe("reactions_received_year")
def reactions_received_year(roster, type, year):
    qt = _qt()
    date_from, date_to = _year_bounds(year)
    r = qt.get_reaction_stats(
        include=["received"], reaction_type=type, date_from=date_from, date_to=date_to
    )
    return {"values": {"count": _v(r["received"]["total"], "exact")}, "raw": r}


# --- S4: words & language ----------------------------------------------------


@recipe("phrase_trend")
def phrase_trend(roster, phrase, scope=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope) if scope else {}
    r = qt.get_word_trends(
        phrase=phrase, granularity="year", include=["phrase_trend"], **kwargs
    )
    series = sorted(r["phrase_trend"]["series"], key=lambda b: b["period"])
    values = {}
    for bucket in series:
        values[f"{bucket['period']}_me"] = _v(bucket["me"], "exact")
        values[f"{bucket['period']}_them"] = _v(bucket["them"], "exact")
    nonzero = [b for b in series if b["me"]]
    if len(nonzero) >= 2:
        first, last = nonzero[0]["me"], nonzero[-1]["me"]
        direction = "up" if last > first else "down" if last < first else "flat"
        values["direction"] = _v(direction, "exact")
    return {"values": values, "raw": r}


@recipe("term_first_use")
def term_first_use(roster, scope, term):
    from db import CONTACT_LABEL_SQL

    chat_ids = scope["chat_ids"]
    placeholders = ",".join("?" * len(chat_ids))
    with _connect() as con:
        row = con.execute(
            f"SELECT dt.coined_is_me, dt.coined_by_handle_id, m.date, m.text "
            f"FROM dyad_terms dt JOIN messages m ON m.id = dt.first_message_id "
            f"WHERE dt.chat_id IN ({placeholders}) AND dt.term = ? "
            "ORDER BY dt.lift DESC LIMIT 1",
            [*chat_ids, term],
        ).fetchone()
        if row is None:
            raise ValueError(f"term {term!r} not in dyad_terms for chats {chat_ids}")
        if row["coined_is_me"]:
            coined_by = "Me"
        else:
            coined_by = con.execute(
                f"SELECT {CONTACT_LABEL_SQL} FROM handles h "
                "LEFT JOIN contacts c ON c.id = h.contact_id WHERE h.id = ?",
                (row["coined_by_handle_id"],),
            ).fetchone()[0]
    return {
        "values": {
            "first_date": _v(row["date"], "exact"),
            "coined_by": _v(coined_by, "exact"),
            "first_text": _v(row["text"], "exact"),
        },
        "raw": {"term": term, "date": row["date"], "text": row["text"], "coined_by": coined_by},
    }


@recipe("distinctive_terms")
def distinctive_terms(roster, scope):
    qt = _qt()
    r = qt.get_word_trends(
        include=["distinctive_terms"], **_scope_kwargs(scope)
    )
    section = r["distinctive_terms"]
    if section.get("available") is False:
        raise ValueError(f"distinctive_terms unavailable: {section.get('note')}")
    values = {}
    for i, term in enumerate(section["terms"][:3], 1):
        values[f"top_term_{i}"] = _v(term["term"], "exact")
    return {"values": values, "raw": r}


@recipe("emoji_top")
def emoji_top(roster, scope):
    qt = _qt()
    r = qt.get_word_trends(include=["emoji"], **_scope_kwargs(scope))
    values = {}
    for i, row in enumerate(r["emoji"]["top"][:3], 1):
        values[f"emoji_{i}"] = _v(row["emoji"], "exact")
    return {"values": values, "raw": r}


@recipe("lsm")
def lsm(roster, scope):
    qt = _qt()
    r = qt.get_word_trends(include=["lsm"], **_scope_kwargs(scope))
    section = r["lsm"]
    if section.get("available") is False:
        raise ValueError("lsm unavailable")
    return {"values": {"lsm_score": _v(section["score"], "round")}, "raw": r}


@recipe("laugh_rates")
def laugh_rates(roster, scope):
    qt = _qt()
    r = qt.get_word_trends(include=["rates"], **_scope_kwargs(scope))
    series = r["rates"]["series"]
    if not series:
        raise ValueError("no rates series for scope")
    last = series[-1]
    return {
        "values": {
            "me_laugh_per_100": _v(last["me"]["laugh_per_100"], "pct1"),
            "them_laugh_per_100": _v(last["them"]["laugh_per_100"], "pct1"),
        },
        "raw": r,
    }


@recipe("valence_gated")
def valence_gated(roster, scope):
    qt = _qt()
    r = qt.get_word_trends(include=["valence"], **_scope_kwargs(scope))
    section = r["valence"]
    gated = isinstance(section, dict) and section.get("available") is False
    values = {"gated": _v(gated, "exact")}
    if not gated:
        for bucket in section["series"]:
            values[f"{bucket['period']}_mean_valence"] = _v(bucket["mean_valence"], "round")
    return {"values": values, "raw": r}


@recipe("phrase_hits_me")
def phrase_hits_me(roster, phrase):
    quoted = '"' + phrase.replace('"', '""') + '"'
    with _connect() as con:
        rows = con.execute(
            "SELECT m.id, m.date, m.text "
            "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
            "WHERE messages_fts MATCH ? AND m.is_from_me = 1 ORDER BY m.date",
            (quoted,),
        ).fetchall()
    return {
        "values": {"hits": _v(len(rows), "exact")},
        "raw": {"messages": [dict(r) for r in rows]},
    }


# --- S5: milestones ----------------------------------------------------------


@recipe("milestone_firsts")
def milestone_firsts(roster, scope=None):
    qt = _qt()
    kwargs = _scope_kwargs(scope) if scope else {}
    r = qt.get_milestones(**kwargs)  # get_milestones has no include param
    firsts = r["firsts"]
    values = {}
    if firsts.get("first"):
        values["first_date"] = _v(firsts["first"]["date"], "exact")
        values["first_sender"] = _v(firsts["first"]["sender"], "exact")
        values["first_text"] = _v(firsts["first"]["text"], "exact")
    if firsts.get("last"):
        values["last_date"] = _v(firsts["last"]["date"], "exact")
        values["last_text"] = _v(firsts["last"]["text"], "exact")
    return {"values": values, "raw": r}


@recipe("last_message_from_them")
def last_message_from_them(roster, scope):
    """The last message THEY sent (S5.6 asks 'the last thing <X> sent me').
    get_milestones' firsts.last is the last message regardless of sender --
    when that one is mine it would be wrong GT for this prompt -- so this
    queries their side directly."""
    chat_ids = scope["chat_ids"]
    placeholders = ",".join("?" * len(chat_ids))
    with _connect() as con:
        row = con.execute(
            f"SELECT m.date, m.text FROM messages m "
            f"WHERE m.chat_id IN ({placeholders}) AND m.is_from_me = 0 AND {_REAL} "
            "ORDER BY m.date DESC LIMIT 1",
            chat_ids,
        ).fetchone()
    if row is None:
        raise ValueError(f"no inbound message in chats {chat_ids}")
    return {
        "values": {
            "last_date": _v(row["date"], "exact"),
            "last_text": _v(row["text"], "exact"),
        },
        "raw": {"date": row["date"], "text": row["text"]},
    }


@recipe("nth_landmark")
def nth_landmark(roster, scope):
    qt = _qt()
    r = qt.get_milestones(**_scope_kwargs(scope))
    landmarks = r["nth"]["landmarks"]
    if not landmarks:
        raise ValueError("no landmark ordinals exist for this chat")
    top = max(landmarks, key=lambda row: row["ordinal"])
    return {
        "values": {
            "landmark_ordinal": _v(top["ordinal"], "exact"),
            "landmark_date": _v(top["message"]["date"], "exact"),
            "landmark_text": _v(top["message"]["text"], "exact"),
        },
        "raw": r,
    }


@recipe("longest_silence")
def longest_silence(roster, scope):
    qt = _qt()
    r = qt.get_milestones(**_scope_kwargs(scope))
    s = r["longest_silence"]
    if not s.get("available", True):
        raise ValueError("longest_silence unavailable for scope")
    return {
        "values": {
            "silence_days": _v(s["days"], "round"),
            "silence_from": _v(s["from"], "exact"),
            "silence_to": _v(s["to"], "exact"),
            "broken_by": _v(s["broken_by"], "exact"),
        },
        "raw": r,
    }


@recipe("anniversaries_upcoming")
def anniversaries_upcoming(roster):
    qt = _qt()
    r = qt.get_milestones()
    values = {}
    for i, row in enumerate(r["anniversaries"]["upcoming"][:3], 1):
        values[f"upcoming_{i}_participant"] = _v(row["participant"], "exact")
        values[f"upcoming_{i}_date"] = _v(row["next_anniversary"], "exact")
    return {"values": values, "raw": r}


@recipe("earliest_phrase")
def earliest_phrase(roster, scope, phrase):
    qt = _qt()
    r = qt.get_milestones(phrase=phrase, **_scope_kwargs(scope))
    msg = r["earliest_phrase"]["message"]
    if not msg:
        raise ValueError(f"no hit for phrase {phrase!r} in scope")
    return {
        "values": {
            "first_mention_date": _v(msg["date"], "exact"),
            "first_mention_text": _v(msg["text"], "exact"),
        },
        "raw": r,
    }


@recipe("anniversary_first")
def anniversary_first(roster, scope):
    qt = _qt()
    r = qt.get_milestones(**_scope_kwargs(scope))
    return {
        "values": {"first_contact": _v(r["anniversaries"]["first_contact"], "exact")},
        "raw": r,
    }


# --- S4.7 / S6 / S7 / S8 -----------------------------------------------------


@recipe("word_frequency_sender")
def word_frequency_sender(roster, scope):
    qt = _qt()
    r = qt.get_word_frequency(contact_id=scope["contact_id"], top=10)
    values = {}
    for i, row in enumerate(r["rows"][:5], 1):
        values[f"top_word_{i}"] = _v(row["ngram"], "exact")
    return {"values": values, "raw": r}


@recipe("group_dynamics")
def group_dynamics(roster, group, section):
    qt = _qt()
    r = qt.get_group_dynamics(chat_id=group["chat_ids"], include=[section])
    if "error" in r:
        raise ValueError(f"get_group_dynamics errored: {r['error']}")
    data = r[section]
    values = {}
    if section == "members":
        top = max(data, key=lambda m: m["messages"])
        values["top_member"] = _v(top["member"], "exact")
        values["top_member_share"] = _v(top["share"], "pct1")
    elif section == "lurkers":
        for i, row in enumerate(data, 1):
            values[f"lurker_{i}"] = _v(row["member"], "exact")
    elif section == "gini":
        values["gini"] = _v(data["gini"], "round")
    elif section == "igniters":
        if data.get("available") is False:
            raise ValueError(f"igniters unavailable: {data.get('note')}")
        values["top_igniter"] = _v(data["leaderboard"][0]["member"], "exact")
    elif section == "adjacency":
        top = data["pairs"][0]
        values["top_pair_replier"] = _v(top["replier"], "exact")
        values["top_pair_replied_to"] = _v(top["replied_to"], "exact")
    elif section == "group_lsm":
        if data.get("available") is False:
            raise ValueError(f"group_lsm unavailable: {data.get('note')}")
        scored = [m for m in data["members"] if m.get("lsm_score") is not None]
        low = min(scored, key=lambda m: m["lsm_score"])
        values["standout_member"] = _v(low["member"], "exact")
        values["standout_lsm"] = _v(low["lsm_score"], "round")
    return {"values": values, "raw": r}


@recipe("total_messages_pair")
def total_messages_pair(roster, scope):
    qt = _qt()
    r = qt.get_chat_summary_stats(**_scope_kwargs(scope))
    # get_chat_summary_stats person scope counts messages *sent by* that
    # person; kept in raw so graders can recognize that number in a reply.
    sender = (
        qt.get_chat_summary_stats(contact_id=scope["contact_id"])
        if scope.get("contact_id") is not None else None
    )
    return {
        "values": {"total": _v(r["message_count"], "exact")},
        "raw": {"dyad_stats": r, "sender_scoped_stats": sender},
    }


@recipe("activity_and_latency")
def activity_and_latency(roster, scope):
    qt = _qt()
    ap = qt.get_activity_patterns(include=["heatmap"])
    counts = ap["heatmap"]["counts"]
    hour_totals = [sum(row[h] for row in counts) for h in range(24)]
    rd = qt.get_response_dynamics(**_scope_kwargs(scope))
    return {
        "values": {
            "top_hour": _v(max(range(24), key=hour_totals.__getitem__), "exact"),
            "them_median_s": _v(rd["reply_latency"]["them"]["median_seconds"], "round"),
        },
        "raw": {"activity": ap, "dynamics": rd},
    }


@recipe("compare_pair")
def compare_pair(roster, a, b):
    qt = _qt()
    sa = qt.get_chat_summary_stats(**_scope_kwargs(a))
    sb = qt.get_chat_summary_stats(**_scope_kwargs(b))
    ra = qt.get_response_dynamics(**_scope_kwargs(a))
    rb = qt.get_response_dynamics(**_scope_kwargs(b))
    a_median = ra["reply_latency"]["them"]["median_seconds"]
    b_median = rb["reply_latency"]["them"]["median_seconds"]
    return {
        "values": {
            "a_total": _v(sa["message_count"], "exact"),
            "b_total": _v(sb["message_count"], "exact"),
            "a_them_median_s": _v(a_median, "round"),
            "b_them_median_s": _v(b_median, "round"),
            "more_side": _v(
                a["display_name"] if sa["message_count"] >= sb["message_count"]
                else b["display_name"],
                "exact",
            ),
            "faster_side": _v(
                a["display_name"] if a_median <= b_median else b["display_name"], "exact"
            ),
        },
        "raw": {"a_stats": sa, "b_stats": sb, "a_dynamics": ra, "b_dynamics": rb},
    }


@recipe("sparse_history")
def sparse_history(roster, scope):
    qt = _qt()
    stats = qt.get_chat_summary_stats(**_scope_kwargs(scope))
    ap = qt.get_activity_patterns(
        include=["volume_series"], granularity="year", **_scope_kwargs(scope)
    )
    values = {"lifetime_total": _v(stats["message_count"], "exact")}
    for bucket in ap["volume_series"]["rows"]:
        values[f"{bucket['period']}_total"] = _v(bucket["sent"] + bucket["received"], "exact")
    return {"values": values, "raw": {"stats": stats, "activity": ap}}


@recipe("top_contact_profile")
def top_contact_profile(roster):
    qt = _qt()
    top1 = roster["TOP1"]
    ap = qt.get_activity_patterns(include=["heatmap"], **_scope_kwargs(top1))
    counts = ap["heatmap"]["counts"]
    hour_totals = [sum(row[h] for row in counts) for h in range(24)]
    rd = qt.get_response_dynamics(**_scope_kwargs(top1))
    init = rd["initiation"]
    return {
        "values": {
            "top_contact": _v(top1["display_name"], "exact"),
            "top_hour": _v(max(range(24), key=hour_totals.__getitem__), "exact"),
            "initiation_me": _v(init["me"], "exact"),
            "initiation_them": _v(init["them"], "exact"),
        },
        "raw": {"activity": ap, "dynamics": rd},
    }
