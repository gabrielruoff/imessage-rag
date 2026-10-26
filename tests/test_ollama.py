"""Hermetic tests for ollama.py (request-body construction).
HTTPConnection is stubbed; no Ollama server is contacted."""

import json
import unittest
from unittest import mock

import ollama


class TestThinkTriState(unittest.TestCase):
    """think has four states: _UNSET (fall back to CHAT_THINK), None (omit the
    flag -- some models reject it), True, and False. None and False must not
    collapse into each other."""

    def _body(self, **kwargs):
        sent = {}

        class FakeResponse:
            status = 200

            def read(self):
                return b'{"message": {"content": "ok"}}'

        class FakeConn:
            def __init__(self, *a, **k):
                pass

            def connect(self):
                pass

            def request(self, _method, path, body=None, **_kwargs):
                sent.update(json.loads(body))

            def getresponse(self):
                return FakeResponse()

            def close(self):
                pass

        with mock.patch.object(ollama.http.client, "HTTPConnection", FakeConn):
            ollama.chat_once([{"role": "user", "content": "x"}], **kwargs)
        return sent

    def test_unset_defers_to_chat_think(self):
        with mock.patch.object(ollama, "CHAT_THINK", None):
            self.assertNotIn("think", self._body())
        with mock.patch.object(ollama, "CHAT_THINK", False):
            self.assertIs(self._body()["think"], False)

    def test_explicit_none_omits_flag_even_when_chat_think_set(self):
        with mock.patch.object(ollama, "CHAT_THINK", True):
            self.assertNotIn("think", self._body(think=None))

    def test_explicit_false_overrides_chat_think(self):
        with mock.patch.object(ollama, "CHAT_THINK", True):
            self.assertIs(self._body(think=False)["think"], False)


class TestEmptyThinkFalseDegrades(unittest.TestCase):
    """gpt-oss honours think=false by returning EMPTY content instead of
    erroring, so the reject-path degrade never fires and every caller records
    a per-row failure for what is one config mismatch. chat_once_json detects
    that signature from the reply and drops reasoning control instead."""

    def setUp(self):
        # Process-global degrade latch: isolate this test from the others.
        self.addCleanup(setattr, ollama, "_think_unsupported", False)
        ollama._think_unsupported = False

    def test_empty_reply_to_think_false_retries_without_the_flag(self):
        replies = [{"content": ""}, {"content": '{"verdict": "noise"}'}]
        with mock.patch.object(ollama, "chat_once",
                               side_effect=replies) as fake:
            data = ollama.chat_once_json([{"role": "user", "content": "x"}],
                                         think=False)
        self.assertEqual(data, {"verdict": "noise"})
        # Same messages, reissued with the flag gone -- not a nudge retry.
        self.assertEqual(fake.call_count, 2)
        self.assertIs(fake.call_args_list[0].kwargs["think"], False)
        self.assertIsNone(fake.call_args_list[1].kwargs["think"])
        self.assertEqual(fake.call_args_list[0].args[0],
                         fake.call_args_list[1].args[0])

    def test_degrade_latches_for_the_rest_of_the_run(self):
        with mock.patch.object(
                ollama, "chat_once",
                side_effect=[{"content": ""},
                             {"content": '{"a": 1}'},
                             {"content": '{"b": 2}'}]) as fake:
            ollama.chat_once_json([{"role": "user", "content": "x"}], think=False)
            ollama.chat_once_json([{"role": "user", "content": "y"}], think=False)
        # The second call never re-pays the empty round trip.
        self.assertEqual(fake.call_count, 3)
        self.assertIsNone(fake.call_args_list[2].kwargs["think"])

    def test_empty_reply_without_think_false_is_left_alone(self):
        # An empty reply on any other think value is just a bad reply: nudge
        # once, then give up. Degrading here would hide a real failure.
        with mock.patch.object(ollama, "chat_once",
                               return_value={"content": ""}) as fake:
            data = ollama.chat_once_json([{"role": "user", "content": "x"}],
                                         think="low")
        self.assertIsNone(data)
        self.assertEqual(fake.call_count, 2)
        self.assertFalse(ollama._think_unsupported)


if __name__ == "__main__":
    unittest.main()


class TestUnreachableIsClassifiedAtConnect(unittest.TestCase):
    """chat_once wraps every failure to OPEN a connection as OllamaUnreachable
    (callers treat it as a config error) and leaves a failure after connecting
    -- a read timeout -- as the raw transport error."""

    def test_every_connect_failure_is_unreachable(self):
        import socket
        for exc in (ConnectionRefusedError(), socket.gaierror("nodename"),
                    OSError(65, "No route to host"), TimeoutError()):
            with self.subTest(exc=type(exc).__name__):
                with mock.patch("http.client.HTTPConnection.connect",
                                side_effect=exc):
                    with self.assertRaises(ollama.OllamaUnreachable):
                        ollama.chat_once([{"role": "user", "content": "x"}],
                                         model="test-model")

    def test_a_timeout_after_connecting_stays_a_transport_failure(self):
        with mock.patch("http.client.HTTPConnection.connect"), \
                mock.patch("http.client.HTTPConnection.request",
                           side_effect=TimeoutError()):
            with self.assertRaises(TimeoutError) as caught:
                ollama.chat_once([{"role": "user", "content": "x"}],
                                 model="test-model")
        self.assertNotIsInstance(caught.exception, ollama.OllamaUnreachable)
