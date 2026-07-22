"""Hermetic tests for the Apple Foundation Models transport (applefm.py) and
its ollama.py dispatch. No real shim, no Ollama, no network: the routing tests
patch the applefm functions / stub HTTPConnection, and the protocol tests run a
tiny FAKE Python shim standing in for the compiled Swift binary."""

import json
import os
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

import applefm
import ollama


class _StubHTTP:
    """Minimal HTTPConnection stub for the non-apple (Ollama) path, so those
    tests can assert applefm is NOT reached without contacting a server."""

    class _Resp:
        status = 200

        def read(self):
            return b'{"message": {"content": "ollama"}}'

        def __iter__(self):
            # chat_stream iterates the response line by line.
            yield b'{"message": {"content": "ollama"}, "done": true}\n'

    def __init__(self, *a, **k):
        pass

    def request(self, *a, **k):
        pass

    def getresponse(self):
        return self._Resp()

    def close(self):
        pass


class TestDispatchRouting(unittest.TestCase):
    """The apple sentinel in `model` (or CHAT_MODEL) routes ollama.py's three
    entry points to applefm; any other model stays on the Ollama HTTP path."""

    def test_chat_once_routes_to_apple_by_model_arg(self):
        with mock.patch.object(applefm, "chat_once",
                               return_value={"role": "assistant",
                                             "content": "x"}) as fake:
            out = ollama.chat_once([{"role": "user", "content": "hi"}],
                                   model="apple")
        fake.assert_called_once()
        self.assertEqual(out, {"role": "assistant", "content": "x"})

    def test_chat_once_routes_to_apple_by_chat_model_default(self):
        with mock.patch.object(ollama, "CHAT_MODEL", "apple"), \
             mock.patch.object(applefm, "chat_once",
                               return_value={"role": "assistant",
                                             "content": "x"}) as fake:
            ollama.chat_once([{"role": "user", "content": "hi"}])
        fake.assert_called_once()

    def test_chat_once_non_apple_stays_on_ollama(self):
        with mock.patch.object(ollama.http.client, "HTTPConnection", _StubHTTP), \
             mock.patch.object(applefm, "chat_once") as fake:
            out = ollama.chat_once([{"role": "user", "content": "hi"}],
                                   model="gemma4:26b-a4b-it-qat")
        fake.assert_not_called()
        self.assertEqual(out, {"content": "ollama"})

    def test_chat_once_json_routes_and_forwards_retry_nudge(self):
        with mock.patch.object(applefm, "chat_once_json",
                               return_value={"ok": 1}) as fake:
            out = ollama.chat_once_json([{"role": "user", "content": "hi"}],
                                        model="apple", retry_nudge="NUDGE")
        fake.assert_called_once()
        self.assertEqual(fake.call_args.kwargs["retry_nudge"], "NUDGE")
        self.assertEqual(out, {"ok": 1})

    def test_chat_once_json_non_apple_stays_on_ollama(self):
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": '{"a": 1}'}), \
             mock.patch.object(applefm, "chat_once_json") as fake:
            out = ollama.chat_once_json([{"role": "user", "content": "hi"}],
                                        model="gemma4:26b-a4b-it-qat")
        fake.assert_not_called()
        self.assertEqual(out, {"a": 1})

    def test_chat_stream_routes_to_apple_and_forwards_tools(self):
        def fake_stream(*a, **k):
            yield {"role": "assistant", "content": "a"}
            yield {"role": "assistant", "content": "b"}

        with mock.patch.object(applefm, "chat_stream",
                               side_effect=fake_stream) as fake:
            out = list(ollama.chat_stream([{"role": "user", "content": "hi"}],
                                          tools=[{"t": 1}], model="apple"))
        fake.assert_called_once()
        self.assertEqual(fake.call_args.kwargs["tools"], [{"t": 1}])
        self.assertEqual(out, [{"role": "assistant", "content": "a"},
                               {"role": "assistant", "content": "b"}])

    def test_chat_stream_non_apple_stays_on_ollama(self):
        with mock.patch.object(ollama.http.client, "HTTPConnection", _StubHTTP), \
             mock.patch.object(applefm, "chat_stream") as fake:
            out = list(ollama.chat_stream([{"role": "user", "content": "hi"}],
                                          model="gemma4:26b-a4b-it-qat"))
        fake.assert_not_called()
        self.assertEqual(out, [{"content": "ollama"}])


class TestMessageFlattening(unittest.TestCase):
    """chat_once/chat_stream collapse a message list into the shim's
    (system, prompt) pair: first system -> system, the rest role-labeled."""

    def test_first_system_and_role_labels(self):
        captured = {}

        def fake_respond(system, prompt, schema=None, stream=False,
                         timeout=None):
            captured.update(system=system, prompt=prompt, schema=schema)
            return "ok"

        with mock.patch.object(applefm, "respond", side_effect=fake_respond):
            out = applefm.chat_once([
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "bye"},
            ])
        self.assertEqual(captured["system"], "SYS")
        self.assertEqual(captured["prompt"],
                         "User: hello\n\nAssistant: hi there\n\nUser: bye")
        self.assertIsNone(captured["schema"])
        self.assertEqual(out, {"role": "assistant", "content": "ok"})

    def test_format_json_uses_no_schema_and_folds_a_json_directive(self):
        # An empty guided schema would pin the reply to "{}", so JSON mode
        # generates free text with a JSON directive folded into the system
        # instead -- chat_once_json then parses + nudge-retries.
        captured = {}

        def fake_respond(system, prompt, schema=None, stream=False,
                         timeout=None):
            captured.update(system=system, schema=schema)
            return "{}"

        with mock.patch.object(applefm, "respond", side_effect=fake_respond):
            applefm.chat_once([{"role": "system", "content": "SYS"},
                               {"role": "user", "content": "x"}], format="json")
        self.assertIsNone(captured["schema"])
        self.assertIn("SYS", captured["system"])
        self.assertIn("JSON object", captured["system"])

    def test_think_is_ignored_not_forwarded(self):
        # respond() has no think parameter; chat_once must never forward one.
        def fake_respond(system, prompt, schema=None, stream=False,
                         timeout=None):
            return "ok"

        with mock.patch.object(applefm, "respond", side_effect=fake_respond):
            # Would TypeError if think leaked into respond().
            applefm.chat_once([{"role": "user", "content": "x"}], think="high")


# A tiny Python stand-in for the compiled Swift shim: reads JSON-line requests
# and emits protocol responses. Behavior is driven off the request so tests can
# provoke each path (errors, streaming, crashes) through the prompt/op. The
# __SENTINEL__ placeholder is replaced with a per-test temp path used by the
# crash-once recovery case.
_FAKE_SHIM = r'''#!/usr/bin/env python3
import json, os, sys

SENTINEL = __SENTINEL__


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


for line in iter(sys.stdin.readline, ""):
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    rid = req.get("id")
    op = req.get("op")
    if op == "availability":
        emit({"id": rid, "type": "availability", "available": True,
              "reason": "Apple Intelligence is ready"})
    elif op == "embed":
        texts = req.get("texts", [])
        emit({"id": rid, "type": "embed",
              "vectors": [[float(len(t)), 0.0, 1.5] for t in texts]})
    elif op == "respond":
        prompt = req.get("prompt", "")
        if "CRASH_ALWAYS" in prompt:
            os._exit(1)
        if "CRASH_ONCE" in prompt:
            if not os.path.exists(SENTINEL):
                open(SENTINEL, "w").close()
                os._exit(1)
            emit({"id": rid, "type": "done", "content": "recovered"})
        elif "ERR_OVERFLOW" in prompt:
            emit({"id": rid, "type": "error", "kind": "context_overflow",
                  "message": "exceededContextWindowSize"})
        elif "ERR_UNAVAIL" in prompt:
            emit({"id": rid, "type": "error", "kind": "unavailable",
                  "message": "Apple Intelligence is not enabled"})
        elif req.get("stream"):
            for piece in ("hello ", "world"):
                emit({"id": rid, "type": "chunk", "content": piece})
            emit({"id": rid, "type": "done"})
        elif req.get("schema") is not None:
            emit({"id": rid, "type": "done",
                  "content": json.dumps({"echo": prompt})})
        elif "JSON object" in req.get("system", ""):
            # chat_once(format="json") folds a JSON directive into the system
            # and passes no schema; a compliant model returns a JSON object.
            emit({"id": rid, "type": "done",
                  "content": json.dumps({"echo": prompt})})
        else:
            emit({"id": rid, "type": "done", "content": "RESP:" + prompt})
'''


class TestShimProtocol(unittest.TestCase):
    """Round-trips against the fake shim over the real subprocess/JSON-lines
    machinery: framing, JSON parsing, error mapping, respawn, missing binary."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.sentinel = os.path.join(self.tmpdir, "crash-sentinel")
        shim_path = os.path.join(self.tmpdir, "fake-shim")
        with open(shim_path, "w") as fh:
            fh.write(_FAKE_SHIM.replace("__SENTINEL__", json.dumps(self.sentinel)))
        os.chmod(shim_path, 0o755)
        # Point the transport at the fake shim and reset its module state, then
        # guarantee the spawned subprocess is always torn down.
        self._orig_shim = applefm.SHIM_PATH
        applefm.SHIM_PATH = pathlib.Path(shim_path)
        applefm._proc = None
        applefm._id_counter = 0
        self.addCleanup(self._teardown)

    def _teardown(self):
        with applefm._lock:
            applefm._kill_proc()
        applefm.SHIM_PATH = self._orig_shim
        applefm._id_counter = 0

    def test_respond_nonstream_roundtrip(self):
        self.assertEqual(applefm.respond("sys", "hi there"), "RESP:hi there")

    def test_respond_with_schema_returns_json_string(self):
        out = applefm.respond("sys", "payload",
                              schema={"type": "object", "properties": {}})
        self.assertEqual(json.loads(out), {"echo": "payload"})

    def test_respond_stream_yields_chunks(self):
        gen = applefm.respond("sys", "please stream", stream=True)
        self.assertEqual(list(gen), ["hello ", "world"])

    def test_availability_roundtrip(self):
        self.assertEqual(
            applefm.availability(),
            {"available": True, "reason": "Apple Intelligence is ready"},
        )

    def test_embed_returns_float_vectors(self):
        vecs = applefm.embed_texts_apple(["a", "bb", "ccc"])
        self.assertEqual(vecs, [[1.0, 0.0, 1.5], [2.0, 0.0, 1.5],
                                [3.0, 0.0, 1.5]])
        # Coercion guarantee: every component is a float, never a stray int.
        self.assertTrue(all(isinstance(x, float) for v in vecs for x in v))

    def test_chat_once_flattens_and_roundtrips(self):
        out = applefm.chat_once([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "q"},
        ])
        self.assertEqual(out, {"role": "assistant", "content": "RESP:User: q"})

    def test_chat_once_json_parses_schema_output(self):
        data = applefm.chat_once_json([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "give json"},
        ])
        # format=json folds a JSON directive into the system; the fake shim
        # detects it and echoes {"echo": prompt}.
        self.assertEqual(data, {"echo": "User: give json"})

    def test_context_overflow_error_is_distinguishable(self):
        with self.assertRaises(RuntimeError) as ctx:
            applefm.respond("sys", "ERR_OVERFLOW now")
        self.assertIn("context_overflow", str(ctx.exception))

    def test_unavailable_error_surfaces_reason(self):
        with self.assertRaises(RuntimeError) as ctx:
            applefm.respond("sys", "ERR_UNAVAIL now")
        self.assertIn("Apple Intelligence is not enabled", str(ctx.exception))

    def test_crash_then_retry_recovers(self):
        # First attempt kills the shim after dropping the sentinel; the client
        # respawns and retries ONCE, and the fresh proc answers.
        self.assertEqual(applefm.respond("sys", "CRASH_ONCE please"),
                         "recovered")

    def test_persistent_crash_raises_then_transport_recovers(self):
        with self.assertRaises(RuntimeError) as ctx:
            applefm.respond("sys", "CRASH_ALWAYS")
        self.assertIn("crashed", str(ctx.exception).lower())
        # The dead proc was cleared, so the very next call spawns a fresh shim.
        self.assertEqual(applefm.respond("sys", "again"), "RESP:again")

    def test_missing_binary_points_to_make_target(self):
        applefm.SHIM_PATH = pathlib.Path(self.tmpdir) / "does-not-exist"
        applefm._proc = None
        with self.assertRaises(RuntimeError) as ctx:
            applefm.respond("sys", "hi")
        self.assertIn("make apple-shim", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
