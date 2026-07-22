"""The wellbeing floor, single-sourced.

`chat.run_turn`, `chat.run_turn_events` (the SSE generator), and
`chat_apple.run_turn` each run one user turn to completion and each must enforce
the SAME wellbeing floor:

- **whole-reply sanitize**: once ``get_language_trajectory`` has run this turn,
  the assembled reply is scrubbed by ``wellbeing.sanitize_condition_names`` before
  it reaches the user (streaming is held back so no unsanitized condition name --
  in any sentence -- is ever shown);
- **the ``_epilogue`` channel**: any tool result may carry a reserved
  ``_epilogue`` string (the code-emitted support block). It is collected, printed
  verbatim AFTER the reply, and folded into the returned string; both ``_epilogue``
  and the duplicate ``support_block`` are stripped from the dict the model sees
  (they are code-delivery directives, not data to narrate from).

This module owns the *collection, arming, dedupe, and string-folding* of that
floor so the three entry paths cannot drift. It deliberately does NOT print: the
CLI prints to a rich console, the SSE path emits an event, and the apple path
prints to ``chat.console`` -- each renders the block its own way. It imports only
``wellbeing`` (stdlib-only), so ``chat_apple`` can adopt it without new deps.
"""

import wellbeing

# The one tool whose result arms whole-reply sanitize for the rest of the turn.
_SANITIZE_ARMING_TOOL = "get_language_trajectory"


def sanitize_reply(content):
    """Scrub clinical-condition names from a whole reply.

    Thin pass-through to ``wellbeing.sanitize_condition_names`` so the three entry
    paths name one function. Returns ``(clean_text, was_redacted)``.
    """
    return wellbeing.sanitize_condition_names(content)


class TurnFloor:
    """Per-turn wellbeing-floor state. Construct one per user turn.

    ``sanitize`` starts False and is armed once ``note_tool`` sees the trajectory
    tool run (by NAME -- armed even for a duplicate/error result, matching the
    pre-refactor behavior). ``epilogues`` accumulates the verbatim support text to
    append after the reply.
    """

    def __init__(self):
        self.sanitize = False
        self.epilogues = []

    def note_tool(self, name, result):
        """Register one executed tool call and return the result the MODEL should
        see.

        (a) If ``name`` is the trajectory tool, arm whole-reply sanitize for every
            later stream this turn -- regardless of ``result`` shape (a
            duplicate-call marker or an error dict still arms it, as the original
            inline logic did).
        (b) If ``result`` is a dict carrying a non-blank string ``_epilogue``,
            collect it and return a copy with both ``_epilogue`` and the duplicate
            ``support_block`` stripped. Otherwise return ``result`` unchanged.
        """
        if name == _SANITIZE_ARMING_TOOL:
            self.sanitize = True
        if isinstance(result, dict) and isinstance(result.get("_epilogue"), str) \
                and result["_epilogue"].strip():
            self.epilogues.append(result["_epilogue"])
            return {k: v for k, v in result.items()
                    if k not in ("_epilogue", "support_block")}
        return result

    def epilogue_block(self):
        """The collected epilogues as one printable block, or None if none.

        Identical epilogues are de-duplicated (order preserved) -- a turn that
        calls several conflict tools, each returning the same disclaimer, yields it
        once, not once per tool.
        """
        if not self.epilogues:
            return None
        return "\n\n".join(dict.fromkeys(self.epilogues))

    def fold_epilogues(self, reply):
        """Fold the epilogue block into the returned reply string.

        Reproduces the return value of the old ``_emit_epilogues`` exactly: a no-op
        when nothing was collected (so non-wellbeing turns are byte-for-byte
        unchanged); otherwise the block is appended after a blank line, or returned
        alone when the reply itself is empty.
        """
        block = self.epilogue_block()
        if not block:
            return reply
        return f"{reply}\n\n{block}" if reply else block
