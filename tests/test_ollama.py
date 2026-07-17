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




if __name__ == "__main__":
    unittest.main()
