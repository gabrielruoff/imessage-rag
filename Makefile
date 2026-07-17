.PHONY: reset-db ollama-start ollama-stop install lint \
        ingest annotate annotate-all topics pipeline

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

reset-db:
	.venv/bin/python3 reset_db.py -f

lint:
	.venv/bin/ruff check .

install:
	test -d .venv || /opt/homebrew/bin/python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

ollama-start:
	launchctl setenv OLLAMA_KEEP_ALIVE 24h
	brew services start ollama

ollama-stop:
	brew services stop ollama
