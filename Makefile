.PHONY: help bootstrap source-data env doctor \
        ingest annotate annotate-all topics pipeline eval eval-dry \
        chat wrapped \
        lint test check reset-db \
        install install-ollama ollama-start ollama-stop

# Without this, `make` runs the first target in the file. That used to be
# `ingest`, so the most natural first command on a fresh clone was
# `.venv/bin/python3 ingest.py` against a .venv that doesn't exist yet:
# "no such file or directory", no context, nothing to act on. Bare `make`
# now prints the map instead. (macOS ships GNU make 3.81, which is exactly
# the version .DEFAULT_GOAL was added in.)
.DEFAULT_GOAL := help

# --- Getting started ---------------------------------------------------------

# Self-documenting: descriptions come from the `## ` on each target line, and
# the section headings from this file's own `# --- Name ---` rules, so help
# can't drift from reality the way a hand-maintained echo block would. A
# target with no `## ` is deliberately absent from help.
#
# POSIX awk only -- macOS ships the one-true-awk, not gawk, so no gensub and
# no lazy `.*?` (the greedy FS below is the portable idiom). No ANSI color:
# this repo already takes the position that output degrades to plain when it
# isn't a terminal (see PROGRESS_STYLE), and `make help | less` is normal.
help:  ## Show this help
	@awk 'BEGIN { FS = ":.*## " } \
	     /^# --- / { h = $$0; sub(/^# --- /, "", h); sub(/[ -]+$$/, "", h); \
	                 printf "\n%s\n", h; next } \
	     /^[a-zA-Z0-9_-]+:.*## / { printf "  %-16s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "Fresh clone:  make bootstrap"
	@echo "Broken:       make doctor"
	@echo ""

# Recipe lines, not prerequisites -- the same rule as `pipeline` below, for a
# stronger reason: install-ollama reads config.py *through* the .venv that
# install creates, and source-data must land before ingest. Under `make -j`
# prerequisites are free to run in any order; recipe lines cannot.
#
# Every step is idempotent, so re-running after a failure resumes rather than
# redoing: install skips an existing .venv, `ollama pull` no-ops on a present
# model, source-data refuses to overwrite an existing copy, and ingest is
# incremental. Fix whatever doctor reports, then run it again.
#
# No stdin prompt on purpose: targets that block on input break under CI,
# under `make -j`, and under any wrapper. The banner is the warning.
bootstrap:  ## Fresh clone -> working setup: install, Ollama, source data, ingest
	@echo "This will: create .venv, install Ollama and pull ~15GB of models,"
	@echo "copy your Messages database into source_data/, and run a full ingest."
	@echo "The ingest is the long pole -- tens of minutes on a large corpus."
	@echo ""
	$(MAKE) install
	$(MAKE) install-ollama
	$(MAKE) source-data
	$(MAKE) ingest
	$(MAKE) doctor
	@echo ""
	@echo "Ready -- run 'make chat'"

# Copies the live Messages DB (and Contacts) into source_data/. ingest.py never
# opens the originals -- Messages keeps chat.db in WAL mode while running -- so
# this is a real prerequisite, not a convenience, and until now it existed only
# as prose in the README.
#
# The -wal/-shm sidecars come too, main file first. Copying chat.db alone
# loses every message still sitting in Messages.app's live WAL:
# snapshot_source() copies whatever sidecars it finds next to SOURCE_DB and
# then checkpoints them into the main file, so bringing them along is exactly
# what makes the snapshot current.
#
# We attempt the copy and then read cp's stderr, rather than probing with
# `test -r` first: without Full Disk Access macOS denies stat() on the path,
# so a readability probe reports "missing" and would send the user hunting for
# a file that is sitting right there.
#
# Hyphenated `source-data`, not `source_data`: a target sharing a name with
# the real directory would be considered up to date the moment the directory
# exists, and silently stop running.
source-data:  ## Copy chat.db + AddressBook into source_data/ (needs Full Disk Access)
	@if [ -e source_data/chat.db ]; then \
	  echo "source_data/chat.db already exists -- refusing to overwrite."; \
	  echo "  For a fresh copy:  rm -f source_data/chat.db* && make source-data"; \
	else \
	  mkdir -p source_data; \
	  src="$$HOME/Library/Messages/chat.db"; \
	  echo ">>> copying $$src (this can be hundreds of MB)"; \
	  err=$$(cp "$$src" source_data/chat.db 2>&1) || { \
	    case "$$err" in \
	      *"Operation not permitted"*|*"Permission denied"*) \
	        echo "error: macOS blocked access to $$src."; \
	        echo "  This is Full Disk Access, not file permissions. Grant it to"; \
	        echo "  whichever program launched make -- Terminal, iTerm, VS Code:"; \
	        echo "    System Settings -> Privacy & Security -> Full Disk Access"; \
	        echo "  Then QUIT AND REOPEN that program: the grant only applies to"; \
	        echo "  processes started after it was given. Re-run there.";; \
	      *) echo "error: copying chat.db failed: $$err";; \
	    esac; \
	    rm -f source_data/chat.db; \
	    exit 1; }; \
	  for s in -wal -shm; do \
	    cp "$$src$$s" "source_data/chat.db$$s" 2>/dev/null || true; \
	  done; \
	  echo "    -> source_data/chat.db"; \
	fi
	@if [ -e source_data/address_book.abcddb ]; then \
	  echo "source_data/address_book.abcddb already exists -- left alone."; \
	else \
	  ab=$$(ls -S "$$HOME/Library/Application Support/AddressBook/Sources"/*/AddressBook-v22.abcddb 2>/dev/null | head -1); \
	  if [ -z "$$ab" ]; then \
	    echo "warning: no AddressBook found. Ingest still works, but every"; \
	    echo "  participant stays a raw phone number instead of a name, and it"; \
	    echo "  does that silently. If you expected names, check Full Disk Access."; \
	  else \
	    cp "$$ab" source_data/address_book.abcddb; \
	    for s in -wal -shm; do \
	      cp "$$ab$$s" "source_data/address_book.abcddb$$s" 2>/dev/null || true; \
	    done; \
	    echo "    -> source_data/address_book.abcddb"; \
	  fi; \
	fi

env:  ## Create .env from .env.example (optional -- every value has a default)
	@if [ -f .env ]; then echo ".env already exists -- left alone."; \
	else cp .env.example .env; \
	  echo "created .env from .env.example (all values commented out)"; fi

doctor:  ## Check the environment: venv, brew, Ollama, models, source data, DB
	@ok()   { printf '  [ ok ] %s\n' "$$1"; }; \
	 warn() { printf '  [warn] %s\n' "$$1"; }; \
	 bad()  { printf '  [fail] %s\n' "$$1"; fail=1; }; \
	 fail=0; \
	 echo "Environment"; \
	 if [ -x .venv/bin/python3 ]; then ok ".venv present ($$(.venv/bin/python3 -V 2>&1))"; \
	 else bad "no .venv -- run: make install"; fi; \
	 if .venv/bin/python3 -c 'import dotenv, sqlite_vec' 2>/dev/null; then ok "deps importable (dotenv, sqlite_vec)"; \
	 else bad "venv missing deps -- run: make install"; fi; \
	 command -v brew >/dev/null 2>&1 && ok "brew on PATH" \
	   || warn "no brew -- install-ollama/ollama-start cannot work"; \
	 [ -f .env ] && ok ".env present (overriding config.py defaults)" \
	   || warn "no .env -- using config.py defaults (fine; see .env.example)"; \
	 echo ""; \
	 echo "Full Disk Access"; \
	 if head -c 16 "$$HOME/Library/Messages/chat.db" >/dev/null 2>&1; then \
	   ok "granted -- ~/Library/Messages/chat.db is readable"; \
	 else \
	   warn "DENIED (or Messages never used) -- 'make source-data' will fail. Grant it in System Settings -> Privacy & Security -> Full Disk Access, then restart this terminal"; \
	 fi; \
	 echo ""; \
	 echo "Ollama"; \
	 command -v ollama >/dev/null 2>&1 && ok "ollama binary present" \
	   || warn "no ollama binary -- run: make install-ollama"; \
	 addr=$(OLLAMA_ADDR); \
	 if curl -fsS --max-time 2 "http://$$addr/api/tags" >/dev/null 2>&1; then \
	   ok "reachable at $$addr"; \
	   for m in $$($(MODEL_LIST)); do \
	     if ollama list 2>/dev/null | awk '{print $$1}' | grep -qx -e "$$m" -e "$$m:latest"; then ok "model $$m"; \
	     else bad "model $$m NOT pulled -- run: make install-ollama"; fi; \
	   done; \
	 else bad "not reachable at $$addr -- run: make ollama-start"; fi; \
	 echo ""; \
	 echo "Source data"; \
	 src=$(SOURCE_DB_PATH); \
	 if [ ! -e "$$src" ]; then bad "$$src missing -- run: make source-data"; \
	 elif n=$$(sqlite3 "$$src" 'select count(*) from message' 2>/dev/null); then ok "$$src ($$n messages)"; \
	 else bad "$$src exists but has no 'message' table -- not a Messages database. Re-copy: rm -f source_data/chat.db* && make source-data"; fi; \
	 ab=$$(.venv/bin/python3 -c 'import config; print(config.ADDRESS_BOOK_PATH)' 2>/dev/null || echo source_data/address_book.abcddb); \
	 [ -e "$$ab" ] && ok "address book present (contact names available)" \
	   || warn "no address book at $$ab -- ingest falls back to raw phone numbers, silently"; \
	 echo ""; \
	 echo "Analysis DB"; \
	 tgt=$$(.venv/bin/python3 -c 'import config; print(config.TARGET_DB)' 2>/dev/null || echo imessage_analysis.db); \
	 : "chunk_vectors is a vec0 virtual table -- the system sqlite3 CLI cannot"; \
	 : "open it at all (no such module: vec0), so counting it through the CLI"; \
	 : "reports 0 and warns that semantic search is dead on a fully embedded"; \
	 : "corpus. It has to go through the venv with the extension loaded."; \
	 if [ ! -e "$$tgt" ]; then warn "$$tgt not built yet -- run: make ingest"; \
	 else \
	   m=$$(sqlite3 "$$tgt" 'select count(*) from messages' 2>/dev/null || echo 0); \
	   c=$$(sqlite3 "$$tgt" 'select count(*) from chunks' 2>/dev/null || echo 0); \
	   v=$$(.venv/bin/python3 -c 'import sqlite3, config, db; con = sqlite3.connect(config.TARGET_DB); db.load_vec_extension(con); print(con.execute("select count(*) from chunk_vectors").fetchone()[0])' 2>/dev/null || echo 0); \
	   [ "$$m" -gt 0 ] && ok "$$m messages, $$c chunks ingested" \
	     || warn "$$tgt exists but is empty -- run: make ingest"; \
	   [ "$$v" -gt 0 ] && ok "$$v chunks embedded (semantic search live)" \
	     || warn "0 embedded chunks -- semantic search returns nothing. Ollama up? then: make ingest"; \
	 fi; \
	 echo ""; \
	 [ $$fail -eq 0 ] && echo "All required checks passed." \
	   || echo "Some required checks failed (see [fail] above)."; \
	 exit $$fail

# Preflight probes ------------------------------------------------------------
# (Not a `# --- ` section: these are variables, not targets, so there is
# nothing for help to list under a heading.)
#
# Each probe is defined once and used two ways: the REQUIRE_* wrappers hard-
# fail a pipeline target, while `doctor` runs the bare probe and reports. That
# is why they are variables rather than targets -- a `require-ollama` *target*
# invoked from doctor would `exit 1` and kill doctor's whole recipe line,
# which is precisely what a diagnostic must not do.
#
# Paths and host/port resolve through config.py so a .env override is honored,
# via a subshell inside the recipe and never a top-level $(shell ...) -- that
# would fork Python on every `make lint` too (same reason as install-ollama's
# model list below).
MODEL_LIST = .venv/bin/python3 -c 'import config; print(" ".join(sorted({config.EMBED_MODEL, config.CHAT_MODEL, config.ANNOTATE_MODEL, config.TOPIC_MODEL, config.WRAPPED_MODEL})))'
OLLAMA_ADDR = $$(.venv/bin/python3 -c 'import config; print(f"{config.OLLAMA_HOST}:{config.OLLAMA_PORT}")' 2>/dev/null || echo localhost:11434)
SOURCE_DB_PATH = $$(.venv/bin/python3 -c 'import config; print(config.SOURCE_DB)' 2>/dev/null || echo source_data/chat.db)

REQUIRE_BREW = command -v brew >/dev/null 2>&1 || { \
    echo "error: Homebrew not found -- https://brew.sh"; \
    echo "  Ollama is installed and supervised as a brew service here."; \
    exit 1; }

REQUIRE_OLLAMA = addr=$(OLLAMA_ADDR); \
  curl -fsS --max-time 2 "http://$$addr/api/tags" >/dev/null 2>&1 || { \
    echo "error: no Ollama reachable at $$addr."; \
    echo "  Start it:           make ollama-start"; \
    echo "  Not installed yet:  make install-ollama"; \
    echo "  Full report:        make doctor"; \
    exit 1; }

# `select count(*) from chat`, not `test -f`: an empty or truncated chat.db
# passes any existence check and then dies three stages into ingest.
REQUIRE_SOURCE = src=$(SOURCE_DB_PATH); \
  sqlite3 "$$src" 'select count(*) from chat' >/dev/null 2>&1 || { \
    echo "error: $$src is missing, empty, or not a Messages database."; \
    echo "  Copy it in:  make source-data     (needs Full Disk Access)"; \
    exit 1; }

# --- Pipelines (all need Ollama up: make ollama-start) -----------------------
# These share one SQLite file and one serialized Ollama instance, so they must
# run one at a time -- see the `pipeline` target below.

# Preflight, not prerequisites: both checks are cheap, and the point is to fail
# in two seconds rather than after the ingest, chunk, session and lexicon
# stages have all run. detect_embed_dim() is the first thing to touch Ollama
# and it runs LAST, so a stopped Ollama used to cost the entire local pipeline
# before anything said so.
#
# INGEST_ARGS exists so --no-embed is reachable through make. The Ollama gate
# stays unconditional rather than sniffing --no-embed out of INGEST_ARGS: a
# spuriously blocked run costs two seconds and prints its own fix, which is
# cheaper to read than a conditional guard.
INGEST_ARGS ?=

ingest:  ## Snapshot chat.db -> imessage_analysis.db, chunk, embed
	@$(REQUIRE_SOURCE)
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 ingest.py $(INGEST_ARGS)

# Both annotation passes at the default chunk scope (top ANNOTATE_TOP_DYADS
# chats). --terms is a separate mode, not an addition, hence two commands.
annotate:  ## Annotate chunks, then curate inside-joke candidates
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 annotate.py
	.venv/bin/python3 annotate.py --terms

# Same, but annotating every chunk in the corpus rather than the top chats.
annotate-all:  ## Like annotate, but over every chunk in the corpus
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 annotate.py --all
	.venv/bin/python3 annotate.py --terms

topics:  ## Cluster embedded chunks into topics and label them
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 topics.py

# Sub-makes in a recipe, NOT prerequisites: prerequisites would let `make -j`
# run these concurrently, and concurrent runs deadlock on the SQLite write
# lock (each holds it across many LLM calls) while gaining nothing -- Ollama
# serializes the inference anyway. Recipe lines run in order under any -j.
pipeline:  ## ingest -> annotate -> nudges -> drift -> topics, in order (stays serial under -j)
	$(MAKE) ingest
	$(MAKE) annotate
	$(MAKE) nudges          # --- feature: nudges (runs after annotate) ---
# feature: drift >>> (additive fence; kept off "# --- " so it isn't parsed as a
# help section) -- drift runs after annotate (its affective-cooling signal
# reads chunk_annotations) and, like the others, is a recipe line so it stays
# serial under -j. DRIFT_ENABLED (default false) decides inside drift.py.
	$(MAKE) drift
# feature: drift <<<
	$(MAKE) topics

# --- feature: nudges ---
# Standalone loose-ends detector pass. Runs AFTER annotate in `pipeline` because
# its Stage-0 exclusion gate reads chunk_annotations, which annotate populates.
# Needs Ollama only for the one-line nudge_text narration.
.PHONY: nudges
nudges:  ## Detect loose threads (nudges) -- run after annotate
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 loose_ends.py
# --- end feature: nudges ---

# feature: drift >>> (additive fence)
# Deterministic drift scoring: no Ollama gate (drift.py makes no LLM calls; it
# reads the annotations annotate.py already produced). DRIFT_ENABLED gates the
# build inside, so `make drift` with the flag off just prints a notice.
.PHONY: drift
drift:  ## Score relationship drift into dyad_drift (opt-in: DRIFT_ENABLED=true)
	.venv/bin/python3 drift.py
# feature: drift <<<

# --- Running -----------------------------------------------------------------
# ARGS passes flags through, e.g.:
#   make chat ARGS='-p "who did I text most in June?"'
#   make wrapped ARGS='--year 2025 --no-llm'
ARGS ?=

chat:  ## Interactive chat over your history (ARGS='-p "..."' for one-shot)
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 chat.py $(ARGS)

wrapped:  ## Year-in-review HTML report (ARGS='--year 2025')
	@$(REQUIRE_OLLAMA)
	.venv/bin/python3 wrapped.py $(ARGS)

# --- Analytics eval (evalkit) ------------------------------------------------
# EVAL_RUN_ID names the results dir under eval/results/; override to re-check or
# re-report an existing run: `make eval EVAL_RUN_ID=20260717T120000-baseline`.
EVALKIT := .venv/bin/python3 -m eval.evalkit
EVAL_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%S)-baseline

# Full P0->P4 sweep. Needs Ollama (make ollama-start). Recipe lines, not
# prerequisites: `run` holds the Ollama lane lock and the phases are ordered.
eval:  ## Full analytics eval sweep (needs Ollama)
	$(EVALKIT) selfcheck
	$(EVALKIT) manifest
	$(EVALKIT) run --run-id $(EVAL_RUN_ID) --all
	$(EVALKIT) check --run $(EVAL_RUN_ID)
	$(EVALKIT) report --run $(EVAL_RUN_ID)

# Everything the eval can do without Ollama: static validation, ground truth,
# and the resolved prompts the sweep would send.
eval-dry:  ## Eval phases that need no Ollama (validation, ground truth, prompts)
	$(EVALKIT) selfcheck
	$(EVALKIT) manifest
	$(EVALKIT) run --dry-run --all --run-id $(EVAL_RUN_ID)

# --- Development -------------------------------------------------------------

lint:  ## ruff check .
	.venv/bin/ruff check .

test:  ## Run the test suite (needs no Ollama or chat.db)
	.venv/bin/python3 -m unittest discover tests

# What CI enforces, in one command. The compileall exclusion is a deliberate
# divergence from .github/workflows/ci.yml: CI has no .venv, so a bare
# `compileall .` is safe there and would walk thousands of vendored files here.
check:  ## Everything CI enforces: lint, tests, compile check
	$(MAKE) lint
	$(MAKE) test
	.venv/bin/python3 -m compileall -q -x '(^|/)(\.venv|\.git|__pycache__|source_data)/' .

reset-db:  ## Wipe imessage_analysis.db's data, keep the schema
	.venv/bin/python3 reset_db.py -f

# --- Install -----------------------------------------------------------------

# PYTHON defaults to Homebrew's interpreter because python.org builds ship a
# SQLite compiled without loadable-extension support, which sqlite-vec needs.
# Override for Intel Macs (/usr/local/bin/python3) or any other prefix:
#   make install PYTHON=/usr/local/bin/python3
PYTHON ?= /opt/homebrew/bin/python3

install:  ## Create .venv and install pinned deps
	@test -x "$(PYTHON)" || { \
	  echo "error: no interpreter at $(PYTHON)"; \
	  echo "  Install Homebrew Python (brew install python@3.13), or re-run as:"; \
	  echo "    make install PYTHON=/path/to/python3"; \
	  exit 1; }
	test -d .venv || "$(PYTHON)" -m venv .venv
	@.venv/bin/python3 -c "import sqlite3, tempfile, os; \
	  p = os.path.join(tempfile.mkdtemp(), 'probe.db'); \
	  c = sqlite3.connect(p); \
	  c.enable_load_extension(True)" 2>/dev/null || { \
	  echo "error: $(PYTHON) was built without SQLite loadable-extension support."; \
	  echo "  sqlite-vec cannot work with it. Use Homebrew Python:"; \
	  echo "    brew install python@3.13 && rm -rf .venv && make install"; \
	  exit 1; }
	.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

# Ollama setup: the binary, the service, and every model config.py names.
# Deliberately NOT part of `make install` -- that target is fast, offline and
# needed for lint/tests/`ingest --no-embed`, none of which touch Ollama, while
# this one pulls ~15GB and installs a background service.
#
# brew, not `curl ollama.com/install.sh | sh`: on macOS that script installs
# /Applications/Ollama.app, which `ollama-start`'s `brew services` cannot
# manage (and which auto-starts at login, contending for port 11434).
#
# Models come from config.py rather than being spelled here, so a .env that
# repoints CHAT_MODEL/ANNOTATE_MODEL/etc. pulls what it actually uses -- hence
# the .venv dependency. `ollama pull` needs a reachable server, so ollama-start
# runs first; pull is a no-op for an already-present model, making re-runs
# cheap. The model list is computed inside the recipe, not in a `$(shell ...)`
# variable: a top-level one would shell out to Python on every `make lint` too.
install-ollama:  ## Install Ollama, start it, pull every model config.py names
	@$(REQUIRE_BREW)
	@command -v ollama >/dev/null || brew install ollama
	@test -d .venv || { echo "error: no .venv -- run 'make install' first (config.py names the models)"; exit 1; }
	$(MAKE) ollama-start
	@for m in $$($(MODEL_LIST)); do \
	  echo ">>> ollama pull $$m"; ollama pull "$$m" || exit 1; \
	done

# OLLAMA_NUM_PARALLEL=4: parallel request slots, feeding annotate.py's
# ANNOTATE_WORKERS pool (~2x on LLM annotation passes, measured 2026-07-17;
# RAM cost negligible, each slot keeps the full context window).
#
# `restart`, not `start`: the launchctl vars above are only read by ollama at
# process start, and `brew services start` is a NO-OP against an already-
# running service. That combination silently produced the worst case -- a
# server with no parallel slots and an ANNOTATE_WORKERS=4 pool queueing
# against it, looking configured while buying nothing. Ollama exposes no way
# to read NUM_PARALLEL back, so there is nothing to assert afterwards; the
# only fix is to make the restart unconditional.
ollama-start:  ## Start/restart Ollama with 4 parallel request slots
	@$(REQUIRE_BREW)
	launchctl setenv OLLAMA_KEEP_ALIVE 24h
	launchctl setenv OLLAMA_NUM_PARALLEL 4
	brew services restart ollama

ollama-stop:  ## Stop the Ollama service
	@$(REQUIRE_BREW)
	brew services stop ollama
