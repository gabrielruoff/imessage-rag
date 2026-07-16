"""Smoke tests for chat.run_turn's tool-budget fallback.

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

    def _fake_stream(self, messages, tools=chat.OLLAMA_TOOLS, show_model_header=True):
        """Stand in for the one network entry point. Mirrors the real default
        so a call that omits `tools` records the schemas it would have sent.
        Deep-copies because run_turn mutates the caller's list in place."""
        self.calls.append({"messages": copy.deepcopy(messages), "tools": tools})
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

    def test_run_turn_returns_early_when_no_tool_calls(self):
        chat.ollama_chat_stream = (
            lambda messages, tools=chat.OLLAMA_TOOLS, show_model_header=True: dict(FINAL_REPLY)
        )
        reply = chat.run_turn([{"role": "system", "content": chat.SYSTEM_PROMPT}])
        self.assertEqual(reply, "the final answer")


if __name__ == "__main__":
    unittest.main()
