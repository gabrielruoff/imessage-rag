"""FastAPI backend for the local, single-user web UI over the chat engine.

Thin transport layer: it persists conversation history in app.db and streams
``chat.run_turn_events`` to the browser over SSE. The chat engine itself stays
stateless -- every request rebuilds the messages list from stored turns.
"""
