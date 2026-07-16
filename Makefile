.PHONY: reset-db ollama-start ollama-stop install

reset-db:
	./reset-db.sh -f

install:
	test -d .venv || /opt/homebrew/bin/python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

ollama-start:
	launchctl setenv OLLAMA_KEEP_ALIVE 24h
	brew services start ollama

ollama-stop:
	brew services stop ollama
