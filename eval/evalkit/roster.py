"""P0 roster resolution (plan §4.2): resolve_roster() maps the suite files'
placeholders (TOP1/TOP2/SPARSE/GROUP, YEAR, PHRASE_HI/LO, TERM, NOBODY,
HANDLE, LONG_MSG, REACTED_MSG) to the real corpus, printing every auto-pick
with its evidence.

App imports live inside functions -- TARGET_DB must be settable in the
environment before config.py first loads (env-before-import invariant).
Importing .recipes at module level is safe: it is equally import-light."""

import random
import re

from .recipes import _connect, _qt, _real


def _contact_entry(con, contact_id, n_messages):
    from db import CONTACT_LABEL_SQL

    name = con.execute(
        f"SELECT {CONTACT_LABEL_SQL} FROM handles h JOIN contacts c ON c.id = h.contact_id "
        "WHERE c.id = ? LIMIT 1",
        (contact_id,),
    ).fetchone()[0]
    handles = con.execute(
        "SELECT id, handle_identifier FROM handles WHERE contact_id = ? ORDER BY id",
        (contact_id,),
    ).fetchall()
    handle_ids = [r["id"] for r in handles]
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
        "handle_identifiers": [r["handle_identifier"] for r in handles],
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


def _message_entry(row, extra=None):
    """A message placeholder: `text_head` is what a prompt can quote back
    without becoming a wall of text, `day` what it can date the message by.
    text_head collapses whitespace and drops attachment placeholders -- it
    gets substituted into a one-line prompt, where a raw newline would split
    the quoted excerpt and a U+FFFC would quote an invisible character."""
    head = " ".join(row["text"].replace("￼", " ").split())[:60]
    entry = {
        "display_name": head,
        "message_id": row["id"],
        "guid": row["guid"],
        "chat_id": row["chat_id"],
        "date": row["date"],
        "day": row["date"][:10],
        "text": row["text"],
        "text_head": head,
    }
    entry.update(extra or {})
    return entry


def _pick_long_msg(con, chat_ids):
    """A message I sent, comfortably longer than query_tools' 300-char
    TEXT_PREVIEW_CHARS, so any search hit truncates it and the full text is
    reachable only via get_full_messages."""
    from query_tools._shared import TEXT_PREVIEW_CHARS

    placeholders = ",".join("?" * len(chat_ids))
    row = con.execute(
        f"SELECT m.id, m.guid, m.chat_id, m.date, m.text FROM messages m "
        f"WHERE m.chat_id IN ({placeholders}) AND m.is_from_me = 1 AND m.date IS NOT NULL "
        f"  AND length(m.text) > ? AND {_real()} "
        "ORDER BY length(m.text) DESC LIMIT 1",
        [*chat_ids, TEXT_PREVIEW_CHARS + 100],
    ).fetchone()
    return _message_entry(row, {"length": len(row["text"])}) if row else None


def _pick_reacted_msg(con, chat_ids):
    """The message I sent this contact that drew the most reactions, among
    those a prompt can actually quote back. Attachment rows carry U+FFFC
    object-replacement chars instead of words -- stripping them and demanding
    20 real characters keeps the excerpt meaningful. Retractions (type >= 3000)
    are excluded: they are rows, not reactions anyone can still see."""
    placeholders = ",".join("?" * len(chat_ids))
    row = con.execute(
        f"SELECT m.id, m.guid, m.chat_id, m.date, m.text, COUNT(r.id) AS n_reactions "
        f"FROM messages m JOIN reactions r ON r.target_message_guid = m.guid "
        f"  AND r.associated_message_type < 3000 "
        f"WHERE m.chat_id IN ({placeholders}) AND m.is_from_me = 1 AND m.date IS NOT NULL "
        f"  AND m.text IS NOT NULL AND length(trim(replace(m.text, char(65532), ''))) >= 20 "
        f"  AND {_real()} "
        "GROUP BY m.id ORDER BY n_reactions DESC, m.id LIMIT 1",
        chat_ids,
    ).fetchone()
    return _message_entry(row, {"n_reactions": row["n_reactions"]}) if row else None


def resolve_roster(db_path, overrides=None):
    """Resolve the placeholder roster against the real DB (plan §4.2).
    Prints every auto-pick with its evidence; `overrides` merge last."""
    qt = _qt()
    qt.DB_PATH = str(db_path)
    roster = {}
    with _connect() as con:
        counts = con.execute(_CONTACT_COUNTS_SQL.format(real=_real())).fetchall()
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
              AND {_real()}
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

        identifiers = roster["TOP1"]["handle_identifiers"]
        roster["HANDLE"] = identifiers[0] if identifiers else None
        roster["LONG_MSG"] = _pick_long_msg(con, top1_chats) if top1_chats else None
        roster["REACTED_MSG"] = _pick_reacted_msg(con, top1_chats) if top1_chats else None

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
    for name in ("YEAR", "PHRASE_HI", "PHRASE_LO", "TERM", "NOBODY", "HANDLE"):
        print(f"  {name}: {roster[name]!r}")
    for name in ("LONG_MSG", "REACTED_MSG"):
        e = roster[name]
        if e is None:
            print(f"  {name}: unresolved")
            continue
        detail = f"{e['length']} chars" if name == "LONG_MSG" else f"{e['n_reactions']} reactions"
        print(f"  {name}: message {e['message_id']} on {e['day']} ({detail})")
    if overrides:
        for k, v in overrides.items():
            print(f"  override {k}: {v!r}")
        roster.update(overrides)
    missing = [k for k, v in roster.items() if v is None]
    if missing:
        print(f"  WARNING: unresolved placeholders {missing} -- cases using them will error")
    return roster
