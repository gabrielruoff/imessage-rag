.PHONY: reset-db ollama-start ollama-stop install install-ollama lint \
        ingest annotate annotate-all topics pipeline eval eval-dry

# --- Pipelines (all need Ollama up: make ollama-start) -----------------------
# These share one SQLite file and one serialized Ollama instance, so they must
# run one at a time -- see the `pipeline` target below.

ingest:
	.venv/bin/python3 ingest.py

# Both annotation passes at the default chunk scope (top ANNOTATE_TOP_DYADS
# chats). --terms is a separate mode, not an addition, hence two commands.
annotate:
	.venv/bin/python3 annotate.py
	.venv/bin/python3 annotate.py --terms

# Same, but annotating every chunk in the corpus rather than the top chats.
annotate-all:
	.venv/bin/python3 annotate.py --all
	.venv/bin/python3 annotate.py --terms

topics:
	.venv/bin/python3 topics.py

# Sub-makes in a recipe, NOT prerequisites: prerequisites would let `make -j`
# run these concurrently, and concurrent runs deadlock on the SQLite write
# lock (each holds it across many LLM calls) while gaining nothing -- Ollama
# serializes the inference anyway. Recipe lines run in order under any -j.
pipeline:
	$(MAKE) ingest
	$(MAKE) annotate
	$(MAKE) topics

# --- Analytics eval (evalkit) ------------------------------------------------
# EVAL_RUN_ID names the results dir under eval/results/; override to re-check or
# re-report an existing run: `make eval EVAL_RUN_ID=20260717T120000-baseline`.
EVALKIT := .venv/bin/python3 -m eval.evalkit
EVAL_RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%S)-baseline

# Full P0->P4 sweep. Needs Ollama (make ollama-start). Recipe lines, not
# prerequisites: `run` holds the Ollama lane lock and the phases are ordered.
eval:
	$(EVALKIT) selfcheck
	$(EVALKIT) manifest
	$(EVALKIT) run --run-id $(EVAL_RUN_ID) --all
	$(EVALKIT) check --run $(EVAL_RUN_ID)
	$(EVALKIT) report --run $(EVAL_RUN_ID)

# Everything the eval can do without Ollama: static validation, ground truth,
# and the resolved prompts the sweep would send.
eval-dry:
	$(EVALKIT) selfcheck
	$(EVALKIT) manifest
	$(EVALKIT) run --dry-run --all --run-id $(EVAL_RUN_ID)

reset-db:
	.venv/bin/python3 reset_db.py -f

lint:
	.venv/bin/ruff check .

# PYTHON defaults to Homebrew's interpreter because python.org builds ship a
# SQLite compiled without loadable-extension support, which sqlite-vec needs.
# Override for Intel Macs (/usr/local/bin/python3) or any other prefix:
#   make install PYTHON=/usr/local/bin/python3
PYTHON ?= /opt/homebrew/bin/python3

install:
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
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

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
install-ollama:
	@command -v ollama >/dev/null || brew install ollama
	@test -d .venv || { echo "error: no .venv -- run 'make install' first (config.py names the models)"; exit 1; }
	$(MAKE) ollama-start
	@for m in $$(.venv/bin/python3 -c 'import config; print(" ".join(sorted({config.EMBED_MODEL, config.CHAT_MODEL, config.ANNOTATE_MODEL, config.TOPIC_MODEL, config.WRAPPED_MODEL})))'); do \
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
ollama-start:
	launchctl setenv OLLAMA_KEEP_ALIVE 24h
	launchctl setenv OLLAMA_NUM_PARALLEL 4
	brew services restart ollama

ollama-stop:
	brew services stop ollama
