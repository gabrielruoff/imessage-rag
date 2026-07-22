"""Hermetic test for `evalkit selfcheck`'s tool-sync check under feature flags.

`cases.ALL_TOOLS` lists only the UNCONDITIONAL tools; `tool_schemas.TOOL_DEFINITIONS`
adds the flag-gated tools when DRIFT_ENABLED / CONFLICT_TRACE_ENABLED /
PERSONA_SIM_ENABLED are on. selfcheck must reconcile the two in EITHER flag state
-- it previously false-failed whenever any flag was enabled (the owner's own .env).

Mirrors the reload pattern of test_voiceprint.SimulateRegistrationGateTest: toggle
the flags on the live `config`, reload `tool_schemas` (rebuilds TOOL_DEFINITIONS +
GATED_TOOLS) and `chat`, restore in cleanup. No Ollama, no DB."""

import contextlib
import importlib
import io
import unittest

import chat
import config
import tool_schemas
from eval.evalkit import __main__ as evalkit_main

_FLAGS = ("DRIFT_ENABLED", "CONFLICT_TRACE_ENABLED", "PERSONA_SIM_ENABLED")


class SelfcheckFlagGateTest(unittest.TestCase):
    def setUp(self):
        self._saved = {f: getattr(config, f) for f in _FLAGS}
        self.addCleanup(self._set, self._saved)

    @staticmethod
    def _set(values):
        for flag, value in values.items():
            setattr(config, flag, value)
        importlib.reload(tool_schemas)  # rebuilds TOOL_DEFINITIONS + GATED_TOOLS
        importlib.reload(chat)          # keeps dispatch/prompt consistent for siblings

    @staticmethod
    def _run_selfcheck():
        with contextlib.redirect_stdout(io.StringIO()):
            return evalkit_main.cmd_selfcheck(None)

    def test_selfcheck_passes_with_flags_off(self):
        self._set({f: False for f in _FLAGS})
        self.assertEqual(self._run_selfcheck(), 0)

    def test_selfcheck_passes_with_flags_on(self):
        self._set({f: True for f in _FLAGS})
        # The gated tools ARE now registered...
        names = {d["name"] for d in tool_schemas.TOOL_DEFINITIONS}
        self.assertIn("get_relationship_drift", names)
        self.assertIn("trace_conflict_across_chats", names)
        self.assertIn("simulate_contact", names)
        # ...and selfcheck still reconciles to in-sync (this returned 1 before the fix).
        self.assertEqual(self._run_selfcheck(), 0)


if __name__ == "__main__":
    unittest.main()
