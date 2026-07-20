# imessage-rag

Makes iMessage history queryable and analyzable by an LLM. Ingests the
macOS Messages `chat.db` into a normalized SQLite store
(`imessage_analysis.db`) with keyword search (FTS5) and semantic search
over conversation chunks (`sqlite-vec` + local Ollama embeddings), then
exposes that store as tool calls for a local Ollama-hosted chat model to
query — instead of pasting message history into a prompt.

See [`docs/DECISIONS.md`](docs/DECISIONS.md) for full architecture, rationale,
schema, and known environment quirks ([`CLAUDE.md`](CLAUDE.md) is the short
agent-facing orientation).

## Privacy first — read this before running anything

This tool reads your entire iMessage history: every message you have ever
sent or received, and every message anyone has ever sent *you*.

- **The other people in your threads never consented to this analysis.**
  Treat the corpus, the derived DB, and anything you generate from it as
  their private data as much as yours.
- **Everything stays local.** The only network calls go to your own Ollama
  instance on `localhost`. No message content is sent to any hosted API.
- **`imessage_analysis.db` is unencrypted** and contains full message text.
  It is gitignored — keep it that way, and don't sync it to cloud storage.
- **Generated Wrapped reports quote real messages verbatim**, including
  names and inside jokes. `wrapped_*.html` is gitignored for that reason.
  Think hard before sharing one.
- **Don't commit eval artifacts.** `eval/results/` and
  `eval/analytics_eval_manifest*.json` embed real names and message text
  and are gitignored.

Everything in this repo — docs, tests, eval cases — uses placeholder names.
Keep it that way if you contribute back.

## Prerequisites

- **macOS.** The ingest path reads Apple's Messages and Contacts databases;
  there is no equivalent on Linux or Windows.
- **[Homebrew](https://brew.sh).** Ollama is installed and supervised as a
  brew service, and the default interpreter below is Homebrew's.
- **Homebrew Python.** `make install` builds the venv from
  `/opt/homebrew/bin/python3`. python.org builds ship a SQLite without
  loadable-extension support, which `sqlite-vec` requires. Override with
  `make install PYTHON=/path/to/python3` if yours lives elsewhere (Intel
  Macs: `/usr/local/bin/python3`).
- **Full Disk Access.** Reading `~/Library/Messages/chat.db` requires
  granting your terminal (or IDE) Full Disk Access in System Settings →
  Privacy & Security → Full Disk Access, then restarting it. `make doctor`
  reports whether the grant is actually live, and `make source-data` prints
  these instructions if it isn't.
- **[Ollama](https://ollama.com)** with the models pulled — `make
  install-ollama` does both (Homebrew install, then pulls every model
  `config.py` names, so it follows your `.env`). By default that is
  `nomic-embed-text` for embeddings and `gemma4:26b-a4b-it-qat` for
  chat/annotation. The chat model is a 26B MoE; expect roughly 20 GB of free
  RAM and seconds-to-minutes per answer. Point `CHAT_MODEL` at something
  smaller if that's too heavy — see [`.env.example`](.env.example).

## Quickstart

```sh
make bootstrap        # everything below, in order — then `make chat`
```

Or one step at a time:

```sh
make install          # creates .venv, installs pinned deps
make install-ollama   # installs Ollama, starts it, pulls the models config.py names
make source-data      # copies chat.db + Contacts in (needs Full Disk Access)
make ingest           # snapshot chat.db -> imessage_analysis.db, chunk, embed
make chat             # interactive CLI chat over your iMessage history
make chat ARGS='-p "who did I text most in June?"'   # one-shot: answer and exit
```

`make` on its own lists every target. `make doctor` checks the whole setup —
venv, Ollama, models, source data, ingested DB — and names whatever is
missing, which is the fastest way to diagnose anything below.

`ingest.py` reads `source_data/chat.db` (not the live database), so copy
yours in first — along with your AddressBook, if you want contact names
instead of raw phone numbers. `make source-data` does both, refuses to
overwrite an existing copy, and if macOS blocks the read it tells you it's
Full Disk Access instead of failing with a bare `Operation not permitted`.

If you'd rather copy by hand, bring the `-wal`/`-shm` sidecars along:

```sh
mkdir -p source_data
cp ~/Library/Messages/chat.db     source_data/chat.db
cp ~/Library/Messages/chat.db-wal source_data/chat.db-wal   # or you lose every
cp ~/Library/Messages/chat.db-shm source_data/chat.db-shm   # unflushed message
cp ~/Library/Application\ Support/AddressBook/Sources/*/AddressBook-v22.abcddb \
   source_data/address_book.abcddb    # optional: contact names
```

Messages keeps `chat.db` in WAL mode while it runs, so your most recent
messages may live only in those sidecars until ingest checkpoints them —
copying `chat.db` alone silently drops them.

Both paths are overridable via `SOURCE_DB` / `ADDRESS_BOOK_PATH` in `.env`
if you'd rather point at the originals. Either way ingest works on a
snapshot copy — the live `chat.db` is never opened directly.
`source_data/` is gitignored.

`make ingest` needs Ollama up for the embedding step; run
`make ingest INGEST_ARGS=--no-embed` instead to skip it and get a searchable
DB without semantic search.

## Analytics

`ingest.py` builds derived analytics tables after chunking, so the six
aggregate tools below work as soon as an ingest finishes — no extra step.
Ask for them in plain language in `chat.py`; the model picks the tool.

| Tool | Answers | Scope |
| --- | --- | --- |
| `get_activity_patterns` | when do we text — heatmap, volume over time, late-night share, busiest days, streaks | optional |
| `get_response_dynamics` | reply speed, read-to-reply, who initiates, ghosting, double-texts | chat or person required |
| `get_reaction_stats` | tapbacks given/received, reciprocity, time-to-react, most-loved messages | optional |
| `get_milestones` | first message, nth-message landmarks, anniversaries, longest silence | optional |
| `get_word_trends` | phrase usage over time, emoji, inside jokes with who coined them, style matching (LSM) | optional |
| `get_group_dynamics` | who dominates, who lurks, who ignites, who replies to whom | group `chat_id` required |

Scoping by a person means *that person's chats*. Each tool returns several
sections in one call, so prefer them over counting search results.

```
> when do I text Alice, and who texts first?
> what emoji did we start using in 2025 that we never used before?
> who lurks in the family group chat?
```

### Optional LLM passes

Two separate batch CLIs. Nothing above depends on them — they add extra
depth. Both commit as they go and are safe to Ctrl-C: re-running resumes
where you left off rather than redoing work.

To cover everything:

```sh
make annotate        # annotate every chunk, then curate inside-joke candidates
make topics          # cluster + label topics
```

Run them one at a time. They can't run in parallel — each holds a SQLite write
lock across many LLM calls, so a concurrent run dies with `database is locked`.
(Within one job, LLM requests do fan out: `make ollama-start` runs Ollama with
4 parallel slots and `annotate.py` keeps `ANNOTATE_WORKERS` requests in flight,
worth ~2x on the LLM passes.) `make pipeline` runs ingest, annotation, and
topics back to back in the right order (and stays sequential even under
`make -j`).

**`annotate.py`** has two modes, and one run only does one of them:

- Bare, it annotates *chunks* (valence, emotions, depth, events). `--quality`
  picks how (default `ANNOTATE_QUALITY`, `distilled`):
  - `distilled` LLM-labels a random seed sample once (`ANNOTATE_SEED_TARGET`,
    default 450 — roughly half an hour, and only ever paid once), then
    classifies every remaining chunk from its already-computed embedding by
    nearest-seed vote: the whole corpus in about a minute, always corpus-wide.
    Valence and depth match LLM fidelity, emotions run a little rougher, and
    *events are left empty* — extracting quotes is the one thing the shortcut
    can't do.
  - `fast` / `balanced` / `max` are the real-LLM-call-per-chunk tiers (each
    slower and more faithful than the last; hours to days corpus-wide), and
    the only source of events. They cover the top `ANNOTATE_TOP_DYADS` chats
    (default 20) bare, honor `--all` / `--top-dyads N`, and with
    `--upgrade-distilled` also replace distilled rows in scope — so an
    overnight `--quality fast --upgrade-distilled` is how events get filled
    in for the chats that matter.
- `--terms` instead curates `dyad_terms` verdicts, which is what gives
  `get_word_trends` its inside-joke section. It's a mode switch, so a bare run
  never touches terms — this is the second run you need. It is *not* a
  `--quality` tier and won't accept one: classifying a term is a four-way
  choice answered in one word, so it batches `ANNOTATE_TERMS_BATCH_SIZE` terms
  per request and skips the reasoning pass (`ANNOTATE_TERMS_THINK`, default
  false — on gemma4 that was 282.6s vs 26.2s per term for the same verdict).
  A term whose curation keeps failing is parked after
  `ANNOTATE_TERMS_MAX_FAILURES` attempts rather than retried by every later
  run; `--retry-failed` gives those terms another chance, and each run names
  the terms it parked. A run in which the model answers *nothing* aborts
  without recording any failures — that's a misconfiguration, not a corpus of
  bad terms, and it shouldn't be able to park everything.

(`make annotate-all` passes `--all`, which only changes anything if you've set
an LLM quality tier — under the default `distilled` both targets already cover
the whole corpus.)

`--dry-run` prints the pending scope and exits without calling Ollama or
writing anything, which is worth doing before an LLM-tier `--all` to see how
many chunks you're signing up for. It previews `--retry-failed` too: the
scope it prints is what the real run would curate, but the unparking itself
is held back. `--limit N` caps a single run. `annotate.py`
records its model and refuses to mix models into a non-empty
`chunk_annotations` table (distilled rows count — they inherit the seed
model's bias), so changing `ANNOTATE_MODEL` means emptying that table first.

**`topics.py`** needs no flags — one bare run clusters every embedded chunk
into `TOPIC_K` clusters (default 50) and then labels them. It requires an
ingest that included embeddings; with an empty `chunk_vectors` it just says so
and exits. `--k N` overrides the cluster count, and `--relabel` re-runs only
the naming pass over existing clusters, for when the labels are bad but the
clustering is fine.

### Wrapped report

A year-in-review HTML report, composed deterministically from the tools
above — the LLM only narrates numbers it was given. The page is a single
self-contained file (inline CSS + JS, no network — nothing leaves the
machine): stat tiles, bar and split-bar charts, a weekday × hour heatmap, a
word cloud, inside-joke cards, and the year's most-reacted messages as
iMessage-style bubbles, in light or dark to match the reader's system theme.
Sections reveal and bars grow as you scroll (static under reduced-motion or
with JS off). Open it in a browser; print to PDF to keep a copy.

```sh
make wrapped ARGS='--year 2025'                      # or drive it through make
.venv/bin/python3 wrapped.py --year 2025             # previews the featured roster, then generates
.venv/bin/python3 wrapped.py --contact "Alice"       # single-relationship deep dive
.venv/bin/python3 wrapped.py --year 2025 --no-llm    # stats only, no narratives
```

Contacts who have gone quiet are auto-excluded from the roster; the preview
shows who made the cut before anything is generated. Override with
`--exclude`/`--include`, skip the prompt with `--yes`, choose a path with
`--out` (default `wrapped_<year>_<scope>.html`).

Tunables (session gap, late-night hours, `LOCAL_TZ`, dyad-term thresholds,
annotation quality/workers/seed pool, topic `k`, wrapped roster size) all
live in `.env` — see `.env.example`.

### Progress output

Every job reports progress as it works. On a terminal that's a live bar with a
percentage and an ETA; piped to a file it degrades to periodic
`  ...N/Total (42%)  eta 3m10s` lines with no escape codes, so
`make annotate-all > run.log` stays readable. `PROGRESS_STYLE=plain` forces the
line style, `PROGRESS_STYLE=off` silences progress entirely.

## Other commands

```sh
make                  # list every target
make doctor           # check venv, Ollama, models, source data, ingested DB
make lint             # ruff check .
make test             # test suite, no Ollama/chat.db needed
make check            # everything CI enforces: lint + tests + compile check
make env              # create .env from .env.example (optional — all defaults work)
make reset-db         # wipe imessage_analysis.db's data, keep schema
```
