"""Hermetic tests for config.py -- the PARSING of the env knobs, not their
consumption (~40 `mock.patch.object(config, ...)` sites elsewhere already cover
consumption).

THE governing fact: every value in config.py resolves AT IMPORT. There are no
lazy accessors, so a `patch.dict(os.environ, ...)` applied AFTER import has no
effect whatsoever on the module constants. Two strategies are used here:

  1. PREFERRED -- call the pure helpers directly (`_think`,
     `_parse_json_dict_env`, `_nudge_kind_weights`, `_resolve`,
     `has_effort_levels`, `is_apple_model`) under a patched environment. Zero
     side effects, no reload, no bleed into sibling test modules.
  2. RELOAD (`_reloaded_config`) -- only where a module-level CONSTANT is
     genuinely the subject. The trap, copied from tests/test_nudges.py: patch
     `dotenv.load_dotenv` AT THE SOURCE, because reloading config re-runs its
     `from dotenv import load_dotenv` and would clobber a patch applied to
     config's own attribute -- and without the stub the test reads the
     DEVELOPER's .env and is not hermetic. Every reload restores the real
     environment and reloads again in a `finally`, or sibling test modules would
     see mutated values (both `unittest tests.test_config tests.test_nudges` and
     the reverse order must pass).

A second fact worth carrying, for anyone reaching for `mock.patch.object`: the
repo binds config two different ways, and only one of them is patchable on
`config` itself. Measured over the tree (app + tests, excluding .venv):

  * 22 modules do `from config import X`, which BINDS THE VALUE at their own
    import time -- 20 of them app modules. For those,
    `mock.patch.object(config, "X")` is a NO-OP; the patch must land on the
    CONSUMING module (e.g. `ollama.CHAT_THINK`, `chat.MAX_TOOL_ITERATIONS`).
  * 42 modules read `config.X` per call and ARE patchable on `config` directly.
    (5 modules do both, for different knobs.)

The counts move as the tree does; the two-shapes distinction does not.
"""

import contextlib
import importlib
import os
import pathlib
import re
import unittest

from unittest import mock

import config
import dotenv


def _env_without(*names):
    """A `patch.dict` target holding the real environment MINUS `names`, so an
    "unset" assertion cannot be broken by whatever the developer exports."""
    env = {k: v for k, v in os.environ.items() if k not in names}
    return mock.patch.dict(os.environ, env, clear=True)


@contextlib.contextmanager
def _reloaded_config(**overrides):
    """Reload config.py with `overrides` applied to os.environ (a None value
    UNSETS the var) and .env loading stubbed at the dotenv source, then restore
    the environment and reload again so nothing leaks to sibling tests."""
    saved = dict(os.environ)
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        with mock.patch.object(dotenv, "load_dotenv", lambda *a, **k: False):
            importlib.reload(config)
        yield config
    finally:
        os.environ.clear()
        os.environ.update(saved)
        importlib.reload(config)  # re-reads the real .env, restores real values


class ThinkTriStateTest(unittest.TestCase):
    """config._think (config.py:71-78). `think` is TRI-STATE and the state has
    to survive to the request body: None OMITS the flag entirely (some models
    reject the option), False SENDS `think: false`, True forces a pass."""

    def test_an_unset_think_is_none_not_false(self):
        # assertIs, never assertFalse: assertFalse passes for BOTH None and
        # False and would hide exactly the bug this pins.
        with _env_without("CHAT_THINK"):
            self.assertIs(config._think("CHAT_THINK"), None)
        # ...and an explicit default is what an unset var falls back to.
        with _env_without("TOPIC_THINK"):
            self.assertIs(config._think("TOPIC_THINK", False), False)

    def test_an_empty_think_falls_back_to_the_inherited_default(self):
        # Set-but-empty and whitespace-only are both "unset" here, so
        # TOPIC_THINK= in a .env inherits CHAT_THINK rather than meaning false.
        for raw in ("", "   ", "\t\n"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"TOPIC_THINK": raw}, clear=False):
                    self.assertIs(config._think("TOPIC_THINK", True), True)
                    self.assertIs(config._think("TOPIC_THINK"), None)

    def test_on_is_truthy_for_flags_but_false_for_think(self):
        # The highest-value asymmetry in the file: _think accepts only
        # ("1", "true", "yes") -- no "on" -- while every other bool in config.py
        # accepts "on" too. So one .env written in a consistent style gets two
        # different answers: DRIFT_ENABLED=on turns the feature ON, while
        # CHAT_THINK=on ACTIVELY SENDS think:false (it does not even omit the
        # flag). Both halves asserted together, on the real constants.
        with _reloaded_config(CHAT_THINK="on", DRIFT_ENABLED="on") as cfg:
            self.assertIs(cfg.DRIFT_ENABLED, True)
            self.assertIs(cfg.CHAT_THINK, False)
        # The words _think does accept, for contrast.
        for raw in ("1", "true", "yes", "TRUE", " Yes "):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"CHAT_THINK": raw}, clear=False):
                    self.assertIs(config._think("CHAT_THINK"), True)

    def test_an_unrecognized_value_sends_think_false_rather_than_omitting_it(self):
        # Anything non-empty and unrecognized collapses to False (send
        # think:false), NOT to None (omit) and NOT to the passed default.
        for raw in ("banana", "0", "false", "no", "off", "low"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"CHAT_THINK": raw}, clear=False):
                    self.assertIs(config._think("CHAT_THINK"), False)
                    self.assertIs(config._think("CHAT_THINK", True), False)


class EnvParsingTest(unittest.TestCase):
    def test_a_whitespace_only_quality_is_not_the_default(self):
        # config.py:222 is (os.getenv(...) or "distilled").strip().lower(): the
        # `or` fires BEFORE the .strip(), so a whitespace-only value is TRUTHY,
        # survives the fallback, and strips down to "" -- not "distilled".
        with _reloaded_config(ANNOTATE_QUALITY="   ") as cfg:
            self.assertEqual(cfg.ANNOTATE_QUALITY, "")
            self.assertNotEqual(cfg.ANNOTATE_QUALITY, "distilled")
        # A set-but-EMPTY value is falsy and does reach the default.
        with _reloaded_config(ANNOTATE_QUALITY="") as cfg:
            self.assertEqual(cfg.ANNOTATE_QUALITY, "distilled")
        with _reloaded_config(ANNOTATE_QUALITY=None) as cfg:
            self.assertEqual(cfg.ANNOTATE_QUALITY, "distilled")

    def test_the_tolerant_int_form_reads_empty_as_unset(self):
        # int(os.getenv("X") or "default"): a set-but-empty var is falsy, so the
        # default is used and the import survives.
        with _reloaded_config(EMBED_DIM="", ANNOTATE_KNN_INDEX_CAP="") as cfg:
            self.assertEqual(cfg.EMBED_DIM, 0)
            self.assertEqual(cfg.ANNOTATE_KNN_INDEX_CAP, 800)
        with mock.patch.dict(os.environ, {"EMBED_DIM": ""}, clear=False):
            self.assertEqual(int(os.getenv("EMBED_DIM") or "0"), 0)

    def test_the_strict_int_form_raises_on_an_empty_value(self):
        # int(os.getenv("X", "default")): a var that is SET BUT EMPTY returns ""
        # (the fallback never fires), int("") raises ValueError AT IMPORT, and
        # every CLI in the repo dies on `import config`. Demonstrated with the
        # actual parsing expressions rather than by breaking the real import.
        with mock.patch.dict(os.environ, {"OLLAMA_PORT": ""}, clear=False):
            with self.assertRaises(ValueError):
                int(os.getenv("OLLAMA_PORT", "11434"))
            # The tolerant form over the SAME env survives -- the two forms
            # genuinely differ, and which one a knob got is not cosmetic.
            self.assertEqual(int(os.getenv("OLLAMA_PORT") or "11434"), 11434)
        # Both forms are live in config.py (measured: 42 strict, 61 tolerant).
        src = pathlib.Path(config.__file__).resolve().read_text()
        strict = re.findall(r'(?:int|float)\(os\.getenv\(\s*"[A-Z][A-Z0-9_]*"\s*,', src)
        tolerant = re.findall(r'(?:int|float)\(os\.getenv\(\s*"[A-Z][A-Z0-9_]*"\s*\)\s*or\s*"', src)
        self.assertGreater(len(strict), 0)
        self.assertGreater(len(tolerant), 0)

    def test_parse_json_dict_env_raises_naming_the_variable(self):
        # config.py:496-509 fails LOUDLY at import on a malformed drift knob,
        # and the message names the var so the .env line is findable.
        with mock.patch.dict(os.environ, {"DRIFT_WEIGHTS": "{not json"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                config._parse_json_dict_env("DRIFT_WEIGHTS", "{}")
        self.assertIn("DRIFT_WEIGHTS", str(ctx.exception))
        # Valid JSON that is not an OBJECT is refused the same way.
        with mock.patch.dict(os.environ, {"DRIFT_PHASE_THRESHOLDS": "[1, 2]"}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                config._parse_json_dict_env("DRIFT_PHASE_THRESHOLDS", "{}")
        self.assertIn("DRIFT_PHASE_THRESHOLDS", str(ctx.exception))
        # An unset var parses the shipped default string, which must be an object.
        with _env_without("DRIFT_WEIGHTS"):
            parsed = config._parse_json_dict_env(
                "DRIFT_WEIGHTS", config._DRIFT_WEIGHTS_DEFAULT)
        self.assertIsInstance(parsed, dict)
        self.assertIn("volume", parsed)

    def test_nudge_kind_weights_falls_back_to_defaults_on_malformed_json(self):
        # config.py:420-448 takes the OPPOSITE policy to _parse_json_dict_env on
        # purpose: a bad urgency weight must not take the whole import down.
        with _env_without("NUDGE_KIND_WEIGHTS"):
            defaults = config._nudge_kind_weights()
        for raw in ("{not json", "[1, 2]", '"a string"', "null", "7"):
            with self.subTest(raw=raw):
                with mock.patch.dict(os.environ, {"NUDGE_KIND_WEIGHTS": raw}, clear=False):
                    self.assertEqual(config._nudge_kind_weights(), defaults)
        # A whitespace-only value is stripped to empty and also means "unset".
        with mock.patch.dict(os.environ, {"NUDGE_KIND_WEIGHTS": "   "}, clear=False):
            self.assertEqual(config._nudge_kind_weights(), defaults)

    def test_nudge_kind_weights_merges_a_partial_override(self):
        with _env_without("NUDGE_KIND_WEIGHTS"):
            defaults = config._nudge_kind_weights()
        raw = '{"dormant_tie": 0.1, "unknown_kind": 2, "dropped_plan": "nope"}'
        with mock.patch.dict(os.environ, {"NUDGE_KIND_WEIGHTS": raw}, clear=False):
            merged = config._nudge_kind_weights()
        # Overridden, coerced to float.
        self.assertEqual(merged["dormant_tie"], 0.1)
        self.assertIsInstance(merged["dormant_tie"], float)
        # Untouched keys keep their defaults -- a partial override is partial.
        self.assertEqual(merged["unanswered_question"], defaults["unanswered_question"])
        self.assertEqual(merged["faded_initiator"], defaults["faded_initiator"])
        # An unknown kind is ADDED (harmless: nothing reads it)...
        self.assertEqual(merged["unknown_kind"], 2.0)
        # ...but a non-numeric value is skipped, leaving that kind's default.
        self.assertEqual(merged["dropped_plan"], defaults["dropped_plan"])


class SafetyDefaultTest(unittest.TestCase):
    def test_the_dv_screen_stays_on_for_an_unrecognized_value(self):
        # config.py:622-623 is INVERSE-default -- `not in ("0","off","false","no")`
        # -- so a typo, a nonsense word, or a value from another convention
        # leaves the DV screen ON. That is the correct direction for a safety
        # gate and the opposite of every other bool in the file; pinned here so
        # nobody "normalizes" it into the `in (...)` form used elsewhere.
        for raw in ("banana", "", "   ", "on", "true", "1", "yes", "ON"):
            with self.subTest(raw=raw):
                with _reloaded_config(CONFLICT_DV_SCREEN=raw) as cfg:
                    self.assertIs(cfg.CONFLICT_DV_SCREEN, True)
        # Only the four explicit off-words turn it off.
        for raw in ("0", "off", "false", "no", " OFF "):
            with self.subTest(raw=raw):
                with _reloaded_config(CONFLICT_DV_SCREEN=raw) as cfg:
                    self.assertIs(cfg.CONFLICT_DV_SCREEN, False)
        # Unset is on: the shipped default.
        with _reloaded_config(CONFLICT_DV_SCREEN=None) as cfg:
            self.assertIs(cfg.CONFLICT_DV_SCREEN, True)


class PredicateTest(unittest.TestCase):
    def test_has_effort_levels_is_true_only_for_the_named_family(self):
        # _EFFORT_LEVEL_MODELS is the small NAMED set; everything else is
        # assumed to merely toggle, so an effort string buys it a full pass.
        self.assertEqual(config._EFFORT_LEVEL_MODELS, ("gpt-oss",))
        for name in ("gpt-oss", "gpt-oss:20b", "GPT-OSS:120b", "ollama/gpt-oss"):
            with self.subTest(name=name):
                self.assertTrue(config.has_effort_levels(name))
        for name in ("qwen3.5:4b", "gemma4", "apple", "", "gptoss", None, 7):
            with self.subTest(name=name):
                self.assertFalse(config.has_effort_levels(name))

    def test_is_apple_model_matches_the_sentinel_and_its_prefix(self):
        self.assertTrue(config.is_apple_model("apple"))
        self.assertTrue(config.is_apple_model("apple:"))
        self.assertTrue(config.is_apple_model("apple:on-device"))
        # A near-miss is NOT the sentinel: no substring or case folding.
        for name in ("Apple", "apple-fm", "applefm", "my-apple", " apple", "apples"):
            with self.subTest(name=name):
                self.assertFalse(config.is_apple_model(name))

    def test_is_apple_model_is_false_for_a_non_string(self):
        # Every dispatch seam gates on this, so a None/unset *_MODEL must not
        # raise its way through ollama.py / embeddings.py / chat.py.
        for value in (None, 0, 7, [], {}, object()):
            with self.subTest(value=value):
                self.assertFalse(config.is_apple_model(value))


class DotenvAnchoringTest(unittest.TestCase):
    def test_dotenv_is_loaded_from_the_config_directory_only(self):
        # config.py:19-22. A BARE load_dotenv() walks UP the tree, so a checkout
        # under .claude/worktrees/<name>/ would silently inherit the parent
        # checkout's flags, models and DB paths. This repo literally runs from a
        # worktree, so the invariant is live. Record the call and pin the path.
        expected = pathlib.Path(config.__file__).resolve().parent / ".env"
        recorder = mock.Mock(return_value=False)
        saved = dict(os.environ)
        try:
            with mock.patch.object(dotenv, "load_dotenv", recorder):
                importlib.reload(config)
        finally:
            os.environ.clear()
            os.environ.update(saved)
            importlib.reload(config)
        self.assertEqual(recorder.call_count, 1)
        self.assertEqual(recorder.call_args, mock.call(expected))
        # Explicitly NOT the bare, tree-walking call.
        self.assertNotEqual(recorder.call_args, mock.call())

    def test_a_relative_db_path_anchors_to_the_config_directory(self):
        # config._resolve (config.py:25-27): relative values hang off config.py's
        # own directory, never the CWD, so every CLI opens the same DB whatever
        # it is launched from. An ABSOLUTE env value still wins.
        root = pathlib.Path(config.__file__).resolve().parent
        self.assertEqual(config._resolve("imessage_analysis.db"),
                         root / "imessage_analysis.db")
        self.assertEqual(config._resolve("source_data/chat.db"),
                         root / "source_data" / "chat.db")
        self.assertEqual(config._resolve("/tmp/elsewhere.db"),
                         pathlib.Path("/tmp/elsewhere.db"))
        # And the shipped constants really are anchored.
        self.assertEqual(config.TARGET_DB.parent, root)
        self.assertEqual(config.APP_DB.parent, root)


class EnvExampleParityTest(unittest.TestCase):
    def test_every_knob_is_documented_in_env_example(self):
        # Parity is EXACT in both directions today: a knob nobody can discover
        # is a knob nobody uses, and a documented line for a knob that no longer
        # exists is a lie that `make env` copies verbatim.
        #
        # The scanner must recognize the HELPER forms too. A naive
        # `os.getenv("NAME"` regex reports 7 false positives -- CHAT_THINK,
        # CONFLICT_THINK, NUDGES_THINK, TOPIC_THINK, WRAPPED_THINK,
        # DRIFT_WEIGHTS, DRIFT_PHASE_THRESHOLDS -- because those are read
        # through _think(...) / _parse_json_dict_env(...) instead.
        root = pathlib.Path(config.__file__).resolve().parent
        src = (root / "config.py").read_text()
        declared = set(re.findall(
            r'(?:os\.getenv|_think|_parse_json_dict_env)\(\s*"([A-Z][A-Z0-9_]*)"', src))
        naive = set(re.findall(r'os\.getenv\(\s*"([A-Z][A-Z0-9_]*)"', src))
        self.assertEqual(
            sorted(declared - naive),
            ["CHAT_THINK", "CONFLICT_THINK", "DRIFT_PHASE_THRESHOLDS",
             "DRIFT_WEIGHTS", "NUDGES_THINK", "TOPIC_THINK", "WRAPPED_THINK"])

        # .env.example is a bare commented inventory: `# VAR=default` lines,
        # every one of which must STAY commented (make env copies it verbatim).
        example = (root / ".env.example").read_text()
        documented = set(re.findall(r'^#\s*([A-Z][A-Z0-9_]*)=', example, re.M))
        self.assertEqual(sorted(declared - documented), [],
                         "knob(s) read by config.py but absent from .env.example")
        self.assertEqual(sorted(documented - declared), [],
                         "knob(s) documented in .env.example but not read by config.py")
        self.assertGreater(len(declared), 100)

    def test_no_env_example_line_is_left_uncommented(self):
        root = pathlib.Path(config.__file__).resolve().parent
        live = [ln for ln in (root / ".env.example").read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        self.assertEqual(live, [], "make env copies verbatim: an uncommented "
                                   "line becomes a live override")


if __name__ == "__main__":
    unittest.main()
