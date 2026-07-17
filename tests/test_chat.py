"""Smoke tests for chat.run_turn's tool-budget fallback and the tool-result
size cap (chat._cap_tool_result).

No Ollama and no real chat.db required -- chat.ollama_chat_stream is swapped
for a stub, so nothing opens a socket or a database. Covers the two
load-bearing behaviors of the fallback, both of which regress silently:
  - the tool schemas are actually withheld from the final request (prompting
    alone does not stop the model from calling a tool)
  - FINAL_TURN_NOTE is folded into the FIRST system message rather than
    appended as its own turn (gemma4's Ollama renderer drops later ones)

Run from the repo root:  python3 -m unittest discover tests
"""

import copy
import io
import json
import unittest

from rich.console import Console

import chat

# An unknown tool name on purpose: it routes run_turn into its "unknown tool"
# branch, so no real query_tools function -- and therefore no DB -- is touched.
TOOL_CALL_REPLY = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"function": {"name": "not_a_real_tool", "arguments": {}}}],
}
FINAL_REPLY = {"role": "assistant", "content": "the final answer"}


class RunTurnFallbackTest(unittest.TestCase):
    def setUp(self):
        # from-import binds MAX_TOOL_ITERATIONS into chat's namespace, so the
        # patch has to land on chat, not config. Two keeps the test quiet/fast.
        self._saved_iterations = chat.MAX_TOOL_ITERATIONS
        chat.MAX_TOOL_ITERATIONS = 2
        self.addCleanup(setattr, chat, "MAX_TOOL_ITERATIONS", self._saved_iterations)

        # Swallow the "[tool call] ..." progress lines.
        self._saved_console = chat.console
        chat.console = Console(file=io.StringIO())
        self.addCleanup(setattr, chat, "console", self._saved_console)

        self.calls = []
        self._saved_stream = chat.ollama_chat_stream
        chat.ollama_chat_stream = self._fake_stream
        self.addCleanup(setattr, chat, "ollama_chat_stream", self._saved_stream)

    def _fake_stream(self, messages, tools=chat.OLLAMA_TOOLS, show_model_header=True, model=None):
        """Stand in for the one network entry point. Mirrors the real defaults
        so a call that omits `tools` or `model` records what it would have sent.
        Deep-copies because run_turn mutates the caller's list in place."""
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "model": model})
        # Answer only once the tools are withheld -- otherwise keep burning
        # the budget, which is what drives run_turn into the fallback.
        if tools is None:
            return dict(FINAL_REPLY)
        return copy.deepcopy(TOOL_CALL_REPLY)

    def test_fallback_withholds_tools_from_the_final_request(self):
        chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        self.assertEqual(len(self.calls), 3)  # 2 budgeted iterations + 1 final
        self.assertIsNone(self.calls[-1]["tools"])
        for call in self.calls[:-1]:
            self.assertEqual(call["tools"], chat.OLLAMA_TOOLS)

    def test_fallback_folds_the_note_into_the_first_system_message(self):
        chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        final = self.calls[-1]["messages"]
        self.assertEqual(final[0]["role"], "system")
        self.assertIn(chat.SYSTEM_PROMPT, final[0]["content"])
        self.assertIn(chat.FINAL_TURN_NOTE, final[0]["content"])
        # A trailing/mid-conversation system message would be silently dropped
        # by the gemma4 renderer, so there must be exactly one.
        self.assertEqual([m for m in final if m["role"] == "system"], [final[0]])

    def test_fallback_does_not_persist_the_note_into_the_conversation(self):
        messages = [{"role": "system", "content": chat.SYSTEM_PROMPT}]
        chat.run_turn(messages)
        self.assertEqual(messages[0]["content"], chat.SYSTEM_PROMPT)
        self.assertNotIn(chat.FINAL_TURN_NOTE, messages[0]["content"])

    def test_fallback_returns_the_final_content(self):
        reply = chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        self.assertEqual(reply, "the final answer")

    def test_fallback_appends_the_final_answer_to_the_conversation(self):
        messages = [{"role": "system", "content": chat.SYSTEM_PROMPT}]
        chat.run_turn(messages)
        self.assertEqual(messages[-1]["content"], "the final answer")

    def test_fallback_synthesizes_a_system_message_when_none_is_present(self):
        # Defensive: main() always seeds a system message, so this state is not
        # currently reachable -- the gate only guards against a future caller.
        messages = [{"role": "user", "content": "hi"}]
        chat.run_turn(messages)
        final = self.calls[-1]["messages"]
        self.assertEqual(final[0]["role"], "system")
        self.assertIn(chat.SYSTEM_PROMPT, final[0]["content"])
        self.assertIn(chat.FINAL_TURN_NOTE, final[0]["content"])
        # The user turn survives rather than being consumed as the prompt.
        self.assertEqual(final[1], {"role": "user", "content": "hi"})

    def test_model_override_reaches_every_request(self):
        chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}], model="other-model")
        self.assertEqual(len(self.calls), 3)  # 2 budgeted iterations + 1 final
        for call in self.calls:
            self.assertEqual(call["model"], "other-model")

    def test_run_turn_returns_early_when_no_tool_calls(self):
        def immediate_reply(messages, tools=chat.OLLAMA_TOOLS, show_model_header=True, model=None):
            return dict(FINAL_REPLY)

        chat.ollama_chat_stream = immediate_reply
        reply = chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        self.assertEqual(reply, "the final answer")


class EmptyReplyRetryTest(RunTurnFallbackTest):
    """A reply with neither content nor tool calls (the model stalling out)
    must trigger the same tools-withheld final request as budget exhaustion,
    not be returned as an empty answer."""

    def _fake_stream(self, messages, tools=chat.OLLAMA_TOOLS, show_model_header=True, model=None):
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools, "model": model})
        if tools is None:
            return dict(FINAL_REPLY)
        # First budgeted iteration: an empty, tool-call-free stall.
        return {"role": "assistant", "content": ""}

    def test_empty_reply_falls_through_to_final_request(self):
        reply = chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        self.assertEqual(reply, "the final answer")
        # One stalled iteration + the final request -- the budget (2) is NOT
        # exhausted first.
        self.assertEqual(len(self.calls), 2)
        self.assertIsNone(self.calls[-1]["tools"])
        self.assertIn(chat.FINAL_TURN_NOTE, self.calls[-1]["messages"][0]["content"])

    def test_empty_assistant_message_is_dropped_from_the_conversation(self):
        messages = [{"role": "system", "content": chat.SYSTEM_PROMPT}]
        chat.run_turn(messages)
        empties = [m for m in messages
                   if m["role"] == "assistant" and not m.get("content")]
        self.assertEqual(empties, [])
        self.assertEqual(messages[-1]["content"], "the final answer")

    # The inherited fallback tests exercise the empty-reply path too (the
    # stub returns an empty reply on every budgeted iteration but the first
    # one already breaks out), except the ones premised on exhausting the
    # budget -- re-pin their expected call counts.
    def test_fallback_withholds_tools_from_the_final_request(self):
        chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        self.assertEqual(len(self.calls), 2)  # 1 stalled iteration + 1 final
        self.assertIsNone(self.calls[-1]["tools"])

    def test_model_override_reaches_every_request(self):
        chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}], model="other-model")
        self.assertEqual(len(self.calls), 2)
        for call in self.calls:
            self.assertEqual(call["model"], "other-model")


class DedupGuardTest(unittest.TestCase):
    """An exact repeat of a (tool, args) call within one turn must not
    re-execute the tool -- it burns an iteration for zero new information."""

    def setUp(self):
        self._saved_iterations = chat.MAX_TOOL_ITERATIONS
        chat.MAX_TOOL_ITERATIONS = 4
        self.addCleanup(setattr, chat, "MAX_TOOL_ITERATIONS", self._saved_iterations)

        self._saved_console = chat.console
        chat.console = Console(file=io.StringIO())
        self.addCleanup(setattr, chat, "console", self._saved_console)

        self.executions = []

        def counting_tool(**kwargs):
            self.executions.append(kwargs)
            return [{"id": 1}]

        chat.TOOL_FUNCTIONS["counting_tool"] = counting_tool
        self.addCleanup(chat.TOOL_FUNCTIONS.pop, "counting_tool", None)

        self._saved_stream = chat.ollama_chat_stream
        self.addCleanup(setattr, chat, "ollama_chat_stream", self._saved_stream)

    def _run_with_scripted_calls(self, arg_script):
        """Each entry in arg_script becomes one iteration's tool call; then
        the stub answers."""
        script = list(arg_script)

        def fake_stream(messages, tools=chat.OLLAMA_TOOLS, show_model_header=True, model=None):
            if script:
                args = script.pop(0)
                return {
                    "role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "counting_tool",
                                                 "arguments": args}}],
                }
            return dict(FINAL_REPLY)

        chat.ollama_chat_stream = fake_stream
        messages = [{"role": "system", "content": chat.SYSTEM_PROMPT}]
        reply = chat.run_turn(messages)
        return reply, messages

    def test_identical_repeat_is_not_reexecuted(self):
        reply, messages = self._run_with_scripted_calls([{"a": 1}, {"a": 1}])
        self.assertEqual(reply, "the final answer")
        self.assertEqual(len(self.executions), 1)
        duplicate_notes = [m for m in messages if m["role"] == "tool"
                           and "duplicate_call" in m["content"]]
        self.assertEqual(len(duplicate_notes), 1)
        self.assertIn("counting_tool", duplicate_notes[0]["content"])

    def test_different_arguments_still_execute(self):
        self._run_with_scripted_calls([{"a": 1}, {"a": 2}])
        self.assertEqual(len(self.executions), 2)

    def test_key_is_stable_under_dict_ordering(self):
        self._run_with_scripted_calls([{"a": 1, "b": 2}, {"b": 2, "a": 1}])
        self.assertEqual(len(self.executions), 1)


class CapToolResultTest(unittest.TestCase):
    """Whatever the shape or size of a tool result, _cap_tool_result must hand
    Ollama valid JSON no larger than TOOL_RESULT_MAX_CHARS -- an oversized
    tool message makes Ollama silently drop the whole conversation."""

    def setUp(self):
        # from-import binding again: patch on chat, not config. 500 keeps the
        # test fixtures small.
        self._saved_cap = chat.TOOL_RESULT_MAX_CHARS
        chat.TOOL_RESULT_MAX_CHARS = 500
        self.addCleanup(setattr, chat, "TOOL_RESULT_MAX_CHARS", self._saved_cap)

    def cap(self, result):
        return chat._cap_tool_result(result, json.dumps(result))

    def test_small_result_passes_through_unchanged(self):
        result = [{"id": 1, "text": "hi"}]
        self.assertEqual(self.cap(result), json.dumps(result))

    def test_oversized_list_is_cut_at_whole_rows(self):
        rows = [{"id": i, "text": "x" * 40} for i in range(50)]
        capped = json.loads(self.cap(rows))
        self.assertIs(capped["truncated"], True)
        self.assertEqual(capped["rows_total"], 50)
        self.assertGreaterEqual(capped["rows_returned"], 1)
        # every kept row is complete and identical to the original
        self.assertEqual(capped["rows"], rows[: capped["rows_returned"]])
        self.assertLessEqual(len(json.dumps(capped)), 500)

    def test_oversized_non_list_keeps_a_labeled_fragment(self):
        capped = json.loads(self.cap({"blob": "y" * 2000}))
        self.assertIs(capped["truncated"], True)
        self.assertIn("partial_json", capped)
        self.assertLessEqual(len(json.dumps(capped)), 500)

    def test_single_row_busting_the_cap_falls_back_to_a_fragment(self):
        capped = json.loads(self.cap([{"text": "z" * 2000}]))
        self.assertIs(capped["truncated"], True)
        self.assertIn("partial_json", capped)
        self.assertLessEqual(len(json.dumps(capped)), 500)

    def test_oversized_envelope_is_cut_at_whole_rows(self):
        envelope = {
            "rows": [{"id": i, "text": "x" * 40} for i in range(50)],
            "total_matching": 900, "returned": 50,
            "order": "newest-first", "truncated": False,
        }
        capped = json.loads(self.cap(envelope))
        self.assertIs(capped["truncated"], True)
        self.assertEqual(capped["rows_total"], 50)
        # kept rows are whole and the envelope context survives the cut
        self.assertEqual(capped["rows"], envelope["rows"][: capped["rows_returned"]])
        self.assertEqual(capped["total_matching"], 900)
        self.assertEqual(capped["order"], "newest-first")
        self.assertEqual(capped["returned"], capped["rows_returned"])
        self.assertLessEqual(len(json.dumps(capped)), 500)


if __name__ == "__main__":
    unittest.main()
