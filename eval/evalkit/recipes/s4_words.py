"""S4 recipes: words & language (phrase trends, dyad terms, emoji, LSM,
laugh rates, the annotation-gated valence/emotions sections, sender-scoped
phrase frequency)."""

from ._base import _connect, _qt, _scope_kwargs, _v, recipe

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


def _gated_section(r, name):
    """A get_word_trends section that rides the chunk_annotations gate: returns
    (section, gated, values-seeded-with-the-gate-flag). An empty table must be
    reported as unavailable, never narrated, so `gated` is always GT."""
    section = r[name]
    gated = isinstance(section, dict) and section.get("available") is False
    return section, gated, {"gated": _v(gated, "exact")}


@recipe("valence_gated")
def valence_gated(roster, scope):
    qt = _qt()
    r = qt.get_word_trends(include=["valence"], **_scope_kwargs(scope))
    section, gated, values = _gated_section(r, "valence")
    if not gated:
        for bucket in section["series"]:
            values[f"{bucket['period']}_mean_valence"] = _v(bucket["mean_valence"], "round")
    return {"values": values, "raw": r}


@recipe("emotions_gated")
def emotions_gated(roster, scope):
    """emotions rides the same chunk_annotations gate as valence, but has its
    own shape (a GoEmotions leaderboard, not a trajectory), so it needs its own
    probe even though the gate is shared."""
    qt = _qt()
    r = qt.get_word_trends(include=["emotions"], **_scope_kwargs(scope))
    section, gated, values = _gated_section(r, "emotions")
    if not gated:
        for i, row in enumerate(section["top"][:3], 1):
            values[f"emotion_{i}"] = _v(row["emotion"], "exact")
            values[f"emotion_{i}_count"] = _v(row["count"], "exact")
        values["chunks_annotated"] = _v(section["chunks_annotated"], "exact")
    return {"values": values, "raw": r}


@recipe("phrase_frequency_me")
def phrase_frequency_me(roster):
    """get_word_frequency's other half: my own bigrams, corpus-wide. Scoping
    is by SENDER here (is_from_me), unlike every other analytics tool."""
    qt = _qt()
    r = qt.get_word_frequency(is_from_me=1, ngram=2, top=10)
    values = {}
    for i, row in enumerate(r["rows"][:3], 1):
        values[f"top_phrase_{i}"] = _v(row["ngram"], "exact")
        values[f"top_phrase_{i}_count"] = _v(row["count"], "exact")
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
