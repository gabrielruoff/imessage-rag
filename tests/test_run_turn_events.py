"""Tests for chat.run_turn_events -- the SSE/web event generator.

Hermetic: ollama.chat_stream is swapped for a scripted fake (no socket) and the
tool functions are stubbed in chat.TOOL_FUNCTIONS (no chat.db). Covers the event
sequence, the exact-repeat guard, the tools-withheld final turn, and the
load-bearing wellbeing buffered-sanitize path.

Run from the repo root:  python3 -m unittest tests.test_run_turn_events
"""

import copy
import unittest

import chat
import chat_floor
import ollama


def _content_chunks(*pieces):
    """A model turn that streams `pieces` as content deltas (no tool call)."""
    return [{"role": "assistant", "content": p} for p in pieces]


def _tool_call_turn(name, arguments=None):
    """A model turn that emits a single tool call and no content."""
    return [{
        "role": "assistant",
        "tool_calls": [{"function": {"name": name, "arguments": arguments or {}}}],
    }]


class RunTurnEventsTest(unittest.TestCase):
    def setUp(self):
        # Small budget keeps the fallback tests fast.
        self._saved_iterations = chat.MAX_TOOL_ITERATIONS
        chat.MAX_TOOL_ITERATIONS = 3
        self.addCleanup(setattr, chat, "MAX_TOOL_ITERATIONS", self._saved_iterations)

        # Scripted model turns; each is a list of chunk dicts to yield in order.
        self._turns = []
        self._stream_calls = []
        self._saved_stream = ollama.chat_stream
        ollama.chat_stream = self._fake_stream
        self.addCleanup(setattr, ollama, "chat_stream", self._saved_stream)

        # A fake tool, injected into the dispatch table; records how often it ran.
        self.tool_runs = []
        self._saved_tools = dict(chat.TOOL_FUNCTIONS)
        chat.TOOL_FUNCTIONS["fake_tool"] = self._fake_tool
        chat.TOOL_FUNCTIONS["get_language_trajectory"] = self._fake_trajectory
        self.addCleanup(self._restore_tools)

    def _restore_tools(self):
        chat.TOOL_FUNCTIONS.clear()
        chat.TOOL_FUNCTIONS.update(self._saved_tools)

    def _fake_stream(self, messages, tools=None, model=None):
        """Yield the next scripted turn's chunks. Records the `tools` it was
        called with so the tools-withheld final turn can be asserted."""
        self._stream_calls.append({"tools": tools, "messages": copy.deepcopy(messages)})
        chunks = self._turns.pop(0)
        for chunk in chunks:
            yield chunk

    def _fake_tool(self, **kwargs):
        self.tool_runs.append(kwargs)
        return {"rows": [{"id": 1}]}

    def _fake_trajectory(self, **kwargs):
        return {"rows": [], "_epilogue": "If you need support, reach out.",
                "support_block": "duplicate crisis text"}

    def _run(self, messages=None):
        messages = messages or [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "hi"}]
        return list(chat.run_turn_events(messages)), messages

    # --- basic streaming ---------------------------------------------------

    def test_streams_tokens_and_done(self):
        self._turns = [_content_chunks("Hello ", "world")]
        events, _ = self._run()
        self.assertEqual(
            events,
            [
                {"type": "token", "text": "Hello "},
                {"type": "token", "text": "world"},
                {"type": "done", "reply": "Hello world"},
            ],
        )

    def test_tool_call_then_answer(self):
        self._turns = [_tool_call_turn("fake_tool", {"q": "x"}),
                       _content_chunks("done")]
        events, messages = self._run()
        types = [e["type"] for e in events]
        self.assertEqual(types, ["tool_call", "token", "done"])
        tc = events[0]
        self.assertEqual(tc["name"], "fake_tool")
        self.assertEqual(tc["arguments"], {"q": "x"})
        self.assertEqual(tc["count"], 1)  # len(result["rows"])
        self.assertEqual(events[-1]["reply"], "done")
        self.assertEqual(len(self.tool_runs), 1)
        # messages was mutated in place: system, user, assistant(tool_call),
        # tool result, assistant(final).
        self.assertEqual(messages[-1]["content"], "done")
        self.assertEqual(messages[-2]["role"], "tool")

    def test_duplicate_call_guard_runs_tool_once(self):
        self._turns = [_tool_call_turn("fake_tool", {"a": 1}),
                       _tool_call_turn("fake_tool", {"a": 1}),
                       _content_chunks("answer")]
        events, _ = self._run()
        # Tool executed once; the second identical call hit the guard.
        self.assertEqual(len(self.tool_runs), 1)
        self.assertEqual([e["type"] for e in events],
                         ["tool_call", "tool_call", "token", "done"])

    def test_tools_withheld_on_final_turn(self):
        # Model keeps calling tools until the budget is exhausted; the final
        # request must withhold tools (tools=None).
        self._turns = [_tool_call_turn("fake_tool", {"n": i}) for i in range(3)]
        self._turns.append(_content_chunks("forced answer"))
        events, _ = self._run()
        self.assertEqual(events[-1], {"type": "done", "reply": "forced answer"})
        # Last stream call is the tools-withheld final request.
        self.assertIsNone(self._stream_calls[-1]["tools"])
        self.assertEqual(self._stream_calls[-1]["tools"], None)
        for call in self._stream_calls[:-1]:
            self.assertEqual(call["tools"], chat.OLLAMA_TOOLS)

    # --- wellbeing floor ---------------------------------------------------

    def test_wellbeing_turn_buffers_and_sanitizes(self):
        raw = "You seem quieter. This really looks like depression to me."
        self._turns = [_tool_call_turn("get_language_trajectory", {"contact": "X"}),
                       _content_chunks("You seem quieter. ",
                                       "This really looks like depression to me.")]
        events, _ = self._run()
        types = [e["type"] for e in events]
        # Exactly one token event on a wellbeing turn (the buffered whole reply),
        # then the epilogue, then done -- no partial, unsanitized tokens.
        self.assertEqual(types, ["tool_call", "token", "epilogue", "done"])
        expected_clean, redacted = chat_floor.sanitize_reply(raw)
        self.assertTrue(redacted)
        self.assertNotIn("depression", events[1]["text"])
        self.assertEqual(events[1]["text"], expected_clean)
        # The epilogue is emitted verbatim and support_block is never leaked.
        self.assertEqual(events[2]["text"], "If you need support, reach out.")
        # done.reply == sanitized reply with the epilogue folded in.
        self.assertEqual(events[-1]["reply"],
                         f"{expected_clean}\n\nIf you need support, reach out.")


if __name__ == "__main__":
    unittest.main()
