.PHONY: reset-db ollama-start ollama-stop install lint \
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

install:
	test -d .venv || /opt/homebrew/bin/python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

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
